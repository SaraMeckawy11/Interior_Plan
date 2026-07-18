"""
AI Interior Design Generator using FLUX.2 [klein] 4B (Black Forest Labs).

Replaces the old Stable-Diffusion + ControlNet pipeline with the same local,
offline FLUX.2-klein-4B image-editing model used in Gen_klein.py. The model
takes the actual 3D screenshot as its input image and redesigns it into a
furnished room while keeping the photo's geometry (walls, windows, doors),
so we no longer need Canny / depth / seg ControlNets to hold the layout.

Memory strategy (this machine has a 6 GB RTX 3050, so both models never sit
on the GPU together):
  stage 1: Qwen3 text encoder 4-bit on GPU -> encode every room prompt -> free
  stage 2: transformer 4-bit + VAE on GPU -> denoise each room with the embeds
Weights quantize shard-by-shard straight onto the GPU (never fp32 in RAM).

The public entry point `generate_furnished_plan(...)` keeps the exact same
signature and return shape as before, so plan2.py needs no changes (though it
now also forwards the chosen design style).
"""

import gc
import glob
import os
import re

# Reuse the model snapshot already downloaded by the sibling Gen_klein project
# when it is present, otherwise keep a repo-local git-ignored cache. Either way
# HF_HOME points inside a cache folder so runs are reproducible offline.
_SIBLING_DIR = r"C:\SIA\Interior_design\hf_cache"
_LOCAL_CACHE = os.path.join(os.path.dirname(__file__), "hf_cache")
os.environ.setdefault(
    "HF_HOME",
    os.path.join(
        _SIBLING_DIR if os.path.isdir(_SIBLING_DIR) else _LOCAL_CACHE,
        "huggingface",
    ),
)
# Plain-HTTPS downloads (the default Xet backend stalled on this network).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import threading

import numpy as np
import torch
from PIL import Image

# Import the diffusers/transformers symbols eagerly, at module load, while we
# are still single-threaded. diffusers uses a lazy (_LazyModule) import system
# that is NOT thread-safe: if several worker threads trigger the first-ever
# `from diffusers import Flux2KleinPipeline` at the same moment, one of them can
# lose the race and raise a bogus "cannot import name" error. Resolving the
# names here once removes that race entirely.
from diffusers import BitsAndBytesConfig as DiffusersBnb4bit
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig as TransformersBnb4bit

# Only one generation may touch the GPU at a time. The desktop app can queue
# several rooms at once (e.g. rapid screenshots), and this 6 GB GPU cannot hold
# even two model loads simultaneously — so serialize them.
_GEN_LOCK = threading.Lock()

# ===========================================================
# CONFIG
# ===========================================================

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"  # Apache 2.0, commercial OK
# klein is step-distilled: 4 steps is the intended operating point (~10 s on a
# 4060). guidance_scale is ignored by the distilled model (keep it at 1.0).
STEPS = 4
GUIDANCE = 1.0
SEED = 7

if not torch.cuda.is_available():
    print("[AI] WARNING: no CUDA GPU found — FLUX.2-klein 4-bit needs a GPU.")

# Candidate locations for the ~15 GB local snapshot (sibling project first).
_MODEL_DIR_CANDIDATES = [
    os.path.join(_SIBLING_DIR, "models", "FLUX.2-klein-4B"),
    os.path.join(_LOCAL_CACHE, "models", "FLUX.2-klein-4B"),
]

# Files that must all exist for a snapshot to count as "complete" (so we skip
# the download entirely and load straight from disk).
_REQUIRED_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "transformer/config.json",
    "vae/config.json",
]
_WEIGHT_SIZES = {
    "text_encoder/model-00001-of-00002.safetensors": 4_967_215_360,
    "text_encoder/model-00002-of-00002.safetensors": 3_077_766_632,
    "transformer/diffusion_pytorch_model.safetensors": 7_751_109_744,
    "vae/diffusion_pytorch_model.safetensors": 168_120_878,
}


def _snapshot_complete(model_dir):
    """True if every required config + weight file is present at full size."""
    files_ok = all(
        os.path.isfile(os.path.join(model_dir, rel)) for rel in _REQUIRED_FILES
    )
    weights_ok = all(
        os.path.isfile(os.path.join(model_dir, rel))
        and os.path.getsize(os.path.join(model_dir, rel)) == size
        for rel, size in _WEIGHT_SIZES.items()
    )
    return files_ok and weights_ok


def _resolve_model_dir():
    """Return a local model folder, reusing a complete snapshot or downloading."""
    for candidate in _MODEL_DIR_CANDIDATES:
        if _snapshot_complete(candidate):
            print(f"[AI] Using complete local model snapshot: {candidate}")
            return candidate

    # Nothing complete on disk: download into the repo-local cache (resumable).
    from huggingface_hub import snapshot_download

    target = os.path.join(_LOCAL_CACHE, "models", "FLUX.2-klein-4B")
    print("[AI] Downloading FLUX.2-klein-4B (resumable, ~15 GB first run)...")
    local_dir = snapshot_download(
        MODEL_ID,
        local_dir=target,
        allow_patterns=[
            "model_index.json", "scheduler/*", "tokenizer/*",
            "text_encoder/*", "transformer/*", "vae/*",
        ],
    )
    print(f"[AI] Download complete: {local_dir}")
    return local_dir


# ===========================================================
# PROMPT SYSTEM (ported from Gen_klein.py)
# ===========================================================
# Models draw what you NAME, so each style defines its own shapes, materials,
# floor, curtains, art, plants, lamp and textures.

STYLE_SPECS = {
    "modern": dict(
        sofa="a sculptural curved sofa with a velvet back and boucle seat",
        table="a round travertine pedestal coffee table",
        floor="wide-plank warm honey oak laid straight",
        rug="a LARGE chunky-woven jute rug",
        curtains="cream double-layer drapery, sheer plus linen panels",
        art="an oversized abstract artwork",
        plants="tall olive trees in matte travertine planters",
        lamp="a brass floor lamp with tapered fabric shade",
        ceiling="ONE wide brass disc pendant close under the ceiling",
        textures=("boucle, velvet, travertine, jute and warm oak, subtle "
                  "brass; warm golden ambience"),
    ),
    "classic": dict(
        sofa="a tailored roll-arm sofa with carved wooden legs",
        table="a rectangular marble-top coffee table with carved legs",
        floor="herringbone oak parquet",
        rug="a LARGE bordered wool rug",
        curtains="heavy pleated drapery with elegant tiebacks",
        art="a large framed classical painting",
        plants="sculpted plants in ceramic urns",
        lamp="a column floor lamp with a pleated shade",
        ceiling="ONE crystal chandelier on a short chain, close to the ceiling",
        textures=("rich deeper accents; silk, velvet, marble and dark "
                  "polished wood, antique gold details; stately warm mood"),
    ),
    "scandinavian": dict(
        sofa="a clean-lined fabric sofa on tapered wooden legs",
        table="a round pale-wood coffee table",
        floor="pale matte oak boards",
        rug="a LARGE soft wool rug",
        curtains="airy white linen curtains",
        art="simple framed line-art prints",
        plants="a leafy plant in a simple white pot",
        lamp="a minimalist tripod floor lamp",
        ceiling="ONE small white dome pendant close to the ceiling",
        textures=("muted tone-on-tone accents; wool, linen, pale birch and "
                  "sheepskin, matte black details; bright airy calm"),
    ),
    "boho": dict(
        sofa="a relaxed low sofa with layered patterned cushions",
        table="a round carved-wood or rattan coffee table",
        floor="warm rustic wood boards",
        rug="LAYERED patterned rugs",
        curtains="light flowing natural-cotton curtains",
        art="an eclectic mix of woven and framed wall pieces",
        plants="abundant potted and trailing plants in terracotta and baskets",
        lamp="a woven rattan floor lamp",
        ceiling="ONE woven rattan pendant close to the ceiling",
        textures=("earthy playful accents; rattan, macrame, layered woven "
                  "textiles, jute and terracotta; relaxed sunlit warmth"),
    ),
    "japandi": dict(
        sofa="a low clean-lined sofa in natural linen",
        table="a low round dark-wood coffee table",
        floor="light matte wood boards",
        rug="a LARGE flat-woven neutral rug",
        curtains="plain linen panels",
        art="one minimal ink-brush artwork",
        plants="a single sculptural branch arrangement in a stone vessel",
        lamp="a paper-lantern floor lamp",
        ceiling="ONE round paper lantern close to the ceiling",
        textures=("quiet deeper accents; linen, pale and dark wood, stone "
                  "and paper, matte black; serene zen calm"),
    ),
    "industrial": dict(
        sofa="a cognac leather sofa",
        table="a rectangular reclaimed-wood and black steel coffee table",
        floor="wide dark wood boards",
        rug="a LARGE worn-look neutral rug",
        curtains="simple dark linen panels",
        art="large monochrome photography prints",
        plants="a tall plant in a black metal planter",
        lamp="a black tripod spotlight floor lamp",
        ceiling="ONE black metal ceiling light close to the ceiling",
        textures=("bold contrast accents; leather, black steel, reclaimed "
                  "wood and aged brass; moody warm light"),
    ),
    "minimalist": dict(
        sofa="a low straight-lined sofa in soft neutral fabric",
        table="a low rectangular seamless coffee table",
        floor="seamless pale oak boards",
        rug="a LARGE plain low-pile rug",
        curtains="plain full-height panels near the wall tone",
        art="one single large calm artwork",
        plants="one sculptural plant in a plain pot",
        lamp="a slim unobtrusive floor lamp",
        ceiling="ONE discreet flush ceiling light",
        textures=("subtle tone-on-tone accents; smooth plaster, pale wood "
                  "and soft matte fabric; serene uncluttered light"),
    ),
}

# Map the gallery UI's style labels onto the STYLE_SPECS keys above.
_STYLE_ALIASES = {
    "modern minimalist": "minimalist",
    "modern": "modern",
    "contemporary": "modern",
    "mid-century modern": "modern",
    "mid century modern": "modern",
    "scandinavian": "scandinavian",
    "industrial": "industrial",
    "bohemian": "boho",
    "boho": "boho",
    "traditional": "classic",
    "classic": "classic",
    "japandi": "japandi",
    "minimalist": "minimalist",
}

FURNITURE_BY_ROOM = {
    "living room": (
        "{sofa}, two matching armchairs beside the sofa, {table}, and a media "
        "console with a TV above it"
    ),
    "bedroom": (
        "an upholstered bed with layered premium bedding, two nightstands with "
        "warm lamps, and a bench at the foot of the bed"
    ),
    "dining room": (
        "a solid-wood dining table with sculptural chairs and a styled sideboard"
    ),
    "kitchen": (
        "fitted cabinetry with stone countertops, a breakfast counter with "
        "designer stools, and integrated appliances"
    ),
    "home office": (
        "a wide desk with a refined chair, full bookshelves, and a reading "
        "armchair"
    ),
    "office": (
        "a wide desk with a refined chair, full bookshelves, and a reading "
        "armchair"
    ),
    "kids room": (
        "a cozy bed with playful bedding, a study desk, a soft rug, and "
        "generous storage"
    ),
    "bathroom": (
        "a floating stone-top vanity with a backlit mirror, a glass shower, and "
        "premium tile"
    ),
    "guest room": (
        "an upholstered bed with layered bedding, a nightstand, a dresser with "
        "a mirror, and a small seat"
    ),
    "studio": (
        "a compact sofa, a bed zone, a small dining set and smart storage, "
        "arranged into clear zones"
    ),
}

DEFAULT_COLOR_TONE = "warm vanilla latte"


def _style_spec(style_label):
    """Resolve a UI style label to a STYLE_SPECS entry (with a graceful default)."""
    key = _STYLE_ALIASES.get((style_label or "").lower().strip())
    if key and key in STYLE_SPECS:
        return STYLE_SPECS[key], key
    # Unknown style: build a generic spec that names the style so the model
    # still commits to it.
    label = style_label or "modern"
    return (
        dict(
            sofa=f"an authentic {label} statement sofa",
            table=f"a coffee table in authentic {label} design",
            floor=f"premium flooring true to {label} style",
            rug="a LARGE area rug true to the style",
            curtains="full-height drapery true to the style",
            art="one large artwork matching the style",
            plants="plants in style-matching pots",
            lamp="a floor lamp matching the style",
            ceiling="ONE style-matched ceiling light close to the ceiling",
            textures=(f"deeper tone-on-tone accents; materials and textures "
                      f"authentic to {label} style"),
        ),
        label,
    )


def build_prompt(room_type, style_label, color_tone=DEFAULT_COLOR_TONE,
                 has_windows=True):
    """Compose the FLUX.2-klein redesign prompt for one room."""
    st, style_key = _style_spec(style_label)
    room_key = (room_type or "living room").lower().strip()
    furniture = FURNITURE_BY_ROOM.get(
        room_key,
        f"the essential furniture of a premium {room_type}, beautifully styled",
    ).format(sofa=st["sofa"], table=st["table"])

    # Only ask for curtains / a window wall when the room actually has windows;
    # otherwise the model invents a window that the floor plan does not have.
    if has_windows:
        window_line = (
            f"- Curtains: {st['curtains']}, from a recessed ceiling slot, NO "
            "rod or gap, spanning the window wall to the floor.\n"
        )
    else:
        window_line = (
            "- This room has NO window: keep every wall solid, do not add, "
            "paint or imply any window, glazing or curtain.\n"
        )

    prompt = (
        f"Redesign this room as a {room_type} in {style_key} style.\n\n"
        "HARD CONSTRAINTS:\n"
        "- Keep the photo's GEOMETRY: every wall, window, door and ceiling "
        "keeps its exact position, size, shape and SILL HEIGHT; never add, "
        "remove, move, resize or convert any window or door; balcony doors "
        "stay. Finishes MAY change; geometry may not.\n"
        "- Keep the same camera position and angle.\n\n"
        "DESIGN BRIEF - senior interior designer:\n"
        "- Furnish if empty; else replace everything.\n"
        f"- FULLY FINISH: {st['floor']} floor, smooth painted walls, clean "
        "ceiling; no dust, stains or bare concrete.\n"
        f"- Furnish with: {furniture}; {st['rug']} under all main furniture, "
        f"{st['art']} on the main wall, {st['plants']}, {st['lamp']}, "
        f"{st['ceiling']}, layered cushions, books and ceramics ONLY on the "
        "table.\n"
        f"{window_line}"
        "- PLACEMENT: few high-quality decor pieces, generous open space; "
        "corners MAY stay empty; never crowd two items into one spot; plants "
        "NEVER stand in front of or overlap furniture; nothing blocks windows, "
        "doors or walkways; furniture square to walls; floor and rug CLEAR of "
        "books and papers.\n"
        f"- COLOR 60/30/10: DOMINANT 60% {color_tone} on walls, ceiling, "
        "curtains and rug; SECONDARY 30% one deeper harmonizing shade on sofa, "
        "armchairs and large upholstery; ACCENT 10% ONE bold contrasting color "
        f"ONLY on cushions, art and small decor; {st['textures']}.\n"
        "- Editorial photo, soft natural light, contact shadows."
    )
    return prompt


# ===========================================================
# OUTPUT SIZE
# ===========================================================

def get_target_size(image):
    """Pick output dims (divisible by 32) that keep the photo's orientation.

    On <7.5 GB GPUs we shrink to keep activation memory in budget, exactly like
    Gen_klein.py does for this 6 GB RTX 3050.
    """
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    landscape = w >= h
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        vram_gb = 0.0

    if vram_gb and vram_gb < 7.5:
        return (768, 512) if landscape else (512, 768)
    return (1024, 768) if landscape else (768, 1024)


# ===========================================================
# MODEL PIPELINE (two-stage, memory-safe)
# ===========================================================

_state = {"model_dir": None}


def _get_model_dir():
    if _state["model_dir"] is None:
        _state["model_dir"] = _resolve_model_dir()
    return _state["model_dir"]


def _encode_prompts(model_dir, prompts):
    """Stage 1: load the 4-bit text encoder, embed every prompt, then free it."""
    print("[AI] Stage 1/2: loading text encoder (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    # The encoder silently truncates past 512 tokens, which would drop the tail
    # constraints — warn if any prompt is too long.
    for i, prompt in enumerate(prompts):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ntok = len(tokenizer(chat)["input_ids"])
        if ntok > 512:
            print(f"[AI] *** WARNING: prompt {i} = {ntok} tokens > 512 — the "
                  "end WILL BE CUT OFF. Shorten it! ***")
        else:
            print(f"[AI] Prompt {i} tokens: {ntok}/512")

    text_encoder = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, "text_encoder"),
        torch_dtype=torch.bfloat16,
        quantization_config=TransformersBnb4bit(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )

    embeds = []
    with torch.no_grad():
        for prompt in prompts:
            emb = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=torch.bfloat16,
                device="cuda",
            ).cpu()  # park off-GPU while models swap
            embeds.append(emb)

    del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("[AI] Prompts encoded, text encoder freed.")
    return embeds


def _load_transformer_pipe(model_dir):
    """Stage 2: load the 4-bit transformer + VAE onto the GPU (no text encoder)."""
    print("[AI] Stage 2/2: loading transformer (4-bit) + VAE...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=DiffusersBnb4bit(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        transformer=transformer,
        text_encoder=None,   # already used and freed in stage 1
        tokenizer=None,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Activation-memory savers (the 4-bit load already handles the weights).
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    return pipe


def generate_asset_references(jobs, output_dir):
    """Generate isolated furniture reference images for local image-to-3D.

    ``jobs`` contains ``key`` and ``prompt`` values. Existing PNGs are reused,
    making the expensive FLUX pass a one-time operation for each style/asset.
    The generated images use a neutral background and a clear three-quarter
    product view, which is the input format TripoSR reconstructs best.
    """
    os.makedirs(output_dir, exist_ok=True)
    pending = []
    ready = {}
    for job in jobs:
        path = os.path.join(output_dir, f"{job['key']}.png")
        if os.path.isfile(path) and os.path.getsize(path) > 10_000:
            ready[job["key"]] = path
        else:
            pending.append(dict(job, path=path))

    if not pending:
        return ready

    with _GEN_LOCK:
        model_dir = _get_model_dir()
        prompts = [
            (
                f"{job['prompt']}. A single freestanding furniture object, "
                "complete object fully visible and centered, three-quarter front "
                "product view, accurate construction and realistic proportions, "
                "soft even studio lighting, plain mid-gray background, no room, "
                "no floor, no people, no text, no watermark, no cropped edges."
            )
            for job in pending
        ]
        prompt_embeds = _encode_prompts(model_dir, prompts)
        pipe = _load_transformer_pipe(model_dir)
        blank = Image.new("RGB", (512, 512), (128, 128, 128))
        try:
            for index, (job, embeds) in enumerate(zip(pending, prompt_embeds)):
                result = pipe(
                    prompt=None,
                    prompt_embeds=embeds.to("cuda"),
                    image=[blank],
                    width=512,
                    height=512,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=torch.Generator(device="cuda").manual_seed(
                        SEED + index + 100
                    ),
                ).images[0]
                result.convert("RGB").save(job["path"])
                ready[job["key"]] = job["path"]
                print(f"[AI-3D] Furniture reference ready: {job['key']}")
        finally:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()

    return ready


# ===========================================================
# GEOMETRY SCORING (recall of the photo's structural edges)
# ===========================================================

def geometry_score(input_rgb, candidate_pil):
    """Fraction of the photo's structural edges kept in the redesign."""
    import cv2

    w, h = candidate_pil.size
    inp = cv2.resize(input_rgb, (w, h))
    edges_in = cv2.Canny(
        cv2.GaussianBlur(cv2.cvtColor(inp, cv2.COLOR_RGB2GRAY), (5, 5), 0),
        40, 120,
    )
    edges_cand = cv2.Canny(cv2.cvtColor(np.array(candidate_pil), cv2.COLOR_RGB2GRAY), 40, 120)
    edges_cand = cv2.dilate(edges_cand, np.ones((7, 7), np.uint8))
    kept = ((edges_in > 0) & (edges_cand > 0)).sum()
    return kept / max((edges_in > 0).sum(), 1)


# ===========================================================
# MAIN GENERATION FUNCTION
# ===========================================================

def generate_furnished_plan(base_rgb, room_masks, doors, windows, room_types,
                            px_per_m=100, has_doors=None, has_windows=None,
                            visible_openings=None, camera_params=None,
                            styles=None, color_tone=DEFAULT_COLOR_TONE):
    """Redesign each room screenshot with FLUX.2-klein-4B (image editing).

    The model receives the actual 3D screenshot as its input image, so the
    room's geometry (walls, windows, doors, camera angle) is preserved by the
    image itself — no ControlNet conditioning needed. `visible_openings` and
    `camera_params` are accepted for backward compatibility but are no longer
    used for projection.

    Only one call runs at a time (see `_GEN_LOCK`): the 6 GB GPU cannot hold two
    model loads at once, so concurrent requests are serialized here.

    Returns a list of {"furnished": PIL.Image, "room_type": str, "index": int}.
    """
    with _GEN_LOCK:
        return _generate_furnished_plan_impl(
            base_rgb, room_masks, doors, windows, room_types,
            px_per_m=px_per_m, has_doors=has_doors, has_windows=has_windows,
            visible_openings=visible_openings, camera_params=camera_params,
            styles=styles, color_tone=color_tone,
        )


def _generate_furnished_plan_impl(base_rgb, room_masks, doors, windows, room_types,
                                  px_per_m=100, has_doors=None, has_windows=None,
                                  visible_openings=None, camera_params=None,
                                  styles=None, color_tone=DEFAULT_COLOR_TONE):
    base_image = base_rgb if isinstance(base_rgb, np.ndarray) else np.array(base_rgb)

    if not room_masks:
        room_masks = [None]
    if not room_types:
        room_types = ["Living Room"]

    # ---- gather one (room image, prompt, size) per room -------------------
    jobs = []
    for idx, mask in enumerate(room_masks):
        room_type = room_types[idx] if idx < len(room_types) else "Living Room"
        style = styles[idx] if styles and idx < len(styles) else "Modern"

        if mask is not None:
            ys, xs = np.where(mask == 255)
            if xs.size == 0 or ys.size == 0:
                print(f"[AI] Skipping room {idx + 1}: empty mask")
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            room = base_image[y1:y2 + 1, x1:x2 + 1]
        else:
            room = base_image

        room_has_windows = bool(has_windows) if has_windows is not None else (len(windows) > 0)
        target_w, target_h = get_target_size(room)
        room_pil = Image.fromarray(room).convert("RGB")
        prompt = build_prompt(room_type, style, color_tone, has_windows=room_has_windows)

        print(f"\n[AI] ==== ROOM {idx + 1} - {room_type} / {style} ====")
        print(f"[AI] {room.shape[1]}x{room.shape[0]} -> generating {target_w}x{target_h}")
        jobs.append(dict(index=idx, room_type=room_type, room_pil=room_pil,
                         room_rgb=room, width=target_w, height=target_h,
                         prompt=prompt))

    if not jobs:
        return []

    model_dir = _get_model_dir()

    # ---- stage 1: encode every prompt, then free the encoder --------------
    prompt_embeds = _encode_prompts(model_dir, [j["prompt"] for j in jobs])

    # ---- stage 2: load the transformer once, denoise every room ----------
    pipe = _load_transformer_pipe(model_dir)

    output = []
    try:
        for job, embeds in zip(jobs, prompt_embeds):
            try:
                result = pipe(
                    prompt=None,
                    prompt_embeds=embeds.to("cuda"),
                    image=[job["room_pil"]],
                    width=job["width"],
                    height=job["height"],
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE,
                    generator=torch.Generator(device="cuda").manual_seed(SEED),
                ).images[0]

                try:
                    score = geometry_score(job["room_rgb"], result)
                    print(f"[AI] Room {job['index'] + 1}: geometry score {score:.3f}")
                except Exception:
                    pass

                output.append({
                    "furnished": result,
                    "room_type": job["room_type"],
                    "index": job["index"],
                })
                print(f"[AI] Room {job['index'] + 1} generated successfully!")
            except Exception as e:
                print(f"[AI ERROR] Failed to generate room {job['index'] + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    finally:
        # Free the GPU so the desktop app stays responsive between generations.
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return output


# ===========================================================
# STANDALONE TEST
# ===========================================================

if __name__ == "__main__":
    import cv2

    # Prefer a captured room screenshot; fall back to a bundled floor-plan image.
    candidates = sorted(glob.glob(os.path.join(
        os.path.dirname(__file__), "room_designs", "screenshots", "*.png")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(
            os.path.dirname(__file__), "plan", "*.jp*g")))

    if not candidates:
        raise SystemExit("No test image found under room_designs/screenshots or plan/")

    img_path = candidates[0]
    print(f"[TEST] Using image: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise SystemExit(f"Could not read {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = generate_furnished_plan(
        base_rgb=img_rgb,
        room_masks=[],
        doors=[],
        windows=[],
        room_types=["Living Room"],
        has_doors=True,
        has_windows=False,
        styles=["Modern"],
    )

    if results:
        out = os.path.join(os.path.dirname(__file__), "test_output.png")
        results[0]["furnished"].save(out)
        print(f"[OK] Test successful! Saved {out}")
    else:
        print("[ERROR] Test failed - no results")
