"""Offline AI furniture generation and walkthrough asset loading.

Pipeline:
  1. The existing local FLUX.2-klein model creates isolated furniture images
     from the user's room style and design-density preference.
  2. The local TripoSR model reconstructs those images into real GLB meshes.
  3. Meshes are cached by style and reused in every later walkthrough.

The measured apartment shell always comes from the floor plan; generative AI
is used only for furniture assets, so it cannot move walls or invent openings.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "room_designs" / "local_3d_assets"
REFERENCE_ROOT = ASSET_ROOT / "references"
RUN_ROOT = ASSET_ROOT / "_runs"
TRIPOSR_ROOT = ROOT / "vendor" / "TripoSR"
TRIPOSR_PYTHON = ROOT / ".triposr_venv" / "Scripts" / "python.exe"
TRIPOSR_MODEL = ROOT / "models" / "TripoSR"


ASSET_SPECS = {
    "sofa": ("three-seat upholstered sofa with separate seat and back cushions", 0.90),
    "armchair": ("upholstered lounge armchair with supportive arms", 0.86),
    "coffee_table": ("low rectangular designer coffee table", 0.44),
    "tv_unit": ("low media console with clean cabinet doors, without a television", 0.58),
    "bed": ("queen bed with upholstered headboard, mattress, duvet and pillows", 1.15),
    "nightstand": ("compact bedside table with one drawer", 0.58),
    "wardrobe": ("two-door freestanding wardrobe with refined handles", 2.20),
    "kitchen_island": ("freestanding kitchen island with stone top and cabinet base", 0.94),
    "fridge": ("modern full-height refrigerator with two clean doors", 1.92),
    "dining_table": ("six-seat rectangular dining table without chairs", 0.77),
    "dining_chair": ("comfortable dining chair with four legs", 0.92),
    "sideboard": ("low dining-room sideboard with cabinet doors", 0.86),
    "desk": ("freestanding home office desk with slim drawers", 0.78),
    "office_chair": ("ergonomic upholstered office chair on a five-star base", 1.08),
    "bookshelf": ("tall open bookcase with multiple shelves", 2.00),
    "vanity": ("bathroom vanity cabinet with basin and stone counter", 0.92),
    "toilet": ("realistic modern ceramic toilet", 0.78),
    "shower": ("glass corner shower enclosure with tray and metal frame", 2.05),
}


ROOM_ASSETS = {
    "living": ("sofa", "armchair", "coffee_table", "tv_unit"),
    "bed": ("bed", "nightstand", "wardrobe"),
    "kitchen": ("kitchen_island", "fridge"),
    "dining": ("dining_table", "dining_chair", "sideboard"),
    "office": ("desk", "office_chair", "bookshelf"),
    "bath": ("vanity", "toilet", "shower"),
}


class Local3DError(RuntimeError):
    pass


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", (value or "modern").lower()).strip("_")
    return value or "modern"


def _room_asset_keys(room_type: str) -> tuple[str, ...]:
    room_type = (room_type or "").lower()
    for marker, keys in ROOM_ASSETS.items():
        if marker in room_type:
            return keys
    return ()


def preference_key(config) -> str:
    """Stable cache key for the user's complete coordinated design direction."""
    config = config or {}
    style = config.get("style", "Modern")
    direction = {
        "style": style,
        "profile": config.get("design_profile", "Curated"),
        "color_mood": config.get("color_mood", "Warm neutral"),
        "design_notes": str(config.get("design_notes", "")).strip().lower(),
        "floor_finish": config.get("floor_finish", "Auto by style"),
        "wall_finish": config.get("wall_finish", "Auto by style"),
    }
    signature = hashlib.sha1(
        json.dumps(direction, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{_slug(style)}_{signature}"


def _uses_default_direction(config) -> bool:
    config = config or {}
    return (
        config.get("design_profile", "Curated") == "Curated"
        and config.get("color_mood", "Warm neutral") == "Warm neutral"
        and not str(config.get("design_notes", "")).strip()
        and config.get("floor_finish", "Auto by style") == "Auto by style"
        and config.get("wall_finish", "Auto by style") == "Auto by style"
    )


def _asset_folder(style: str, asset_key: str, design_key: str | None = None) -> Path:
    style_root = ASSET_ROOT / _slug(style)
    if design_key:
        return style_root / design_key / asset_key
    return style_root / asset_key


def asset_path(style: str, asset_key: str, design_key: str | None = None) -> Path:
    return _asset_folder(style, asset_key, design_key) / "mesh.glb"


def _resolved_asset_path(
    style: str,
    asset_key: str,
    design_key: str | None = None,
) -> Path | None:
    """Return a cached mesh, including assets made by older design-key runs."""
    if design_key:
        exact = asset_path(style, asset_key, design_key)
        if exact.is_file() and exact.stat().st_size > 10_000:
            return exact

    canonical = asset_path(style, asset_key)
    if canonical.is_file() and canonical.stat().st_size > 10_000:
        return canonical

    style_root = ASSET_ROOT / _slug(style)
    candidates = []
    if style_root.is_dir():
        for candidate in style_root.glob(f"*/{asset_key}/mesh.glb"):
            try:
                if candidate.stat().st_size > 10_000:
                    candidates.append(candidate)
            except OSError:
                continue
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def runtime_status() -> tuple[bool, str]:
    if not TRIPOSR_PYTHON.is_file():
        return False, "The isolated TripoSR runtime is not installed."
    if not (TRIPOSR_ROOT / "run.py").is_file():
        return False, "The TripoSR source is not installed."
    if not (TRIPOSR_MODEL / "config.yaml").is_file():
        return False, "The TripoSR configuration is missing."
    if not (TRIPOSR_MODEL / "dino_config.json").is_file():
        return False, "The local DINO configuration is missing."
    checkpoint = TRIPOSR_MODEL / "model.ckpt"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
        return False, "The one-time TripoSR model download is not complete."

    site_packages = TRIPOSR_PYTHON.parents[1] / "Lib" / "site-packages"
    package_roots = [site_packages]
    if site_packages.is_dir():
        for pth_file in site_packages.glob("*.pth"):
            try:
                for line in pth_file.read_text(encoding="utf-8").splitlines():
                    extra = Path(line.strip())
                    if extra.is_dir():
                        package_roots.append(extra)
            except (OSError, UnicodeError):
                continue

    required_modules = (
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("transformers", "transformers"),
        ("trimesh", "trimesh"),
    )
    missing = [
        label for label, module in required_modules
        if not any((root / module).exists() for root in package_roots)
    ]
    if missing:
        return False, "TripoSR runtime repair needed: missing " + ", ".join(missing) + "."
    return True, "Local TripoSR furniture generation is ready."


def _requested_jobs(room_configs):
    jobs = {}
    for config in room_configs or []:
        style = config.get("style", "Modern")
        profile = config.get("design_profile", "Curated")
        design_key = preference_key(config)
        color_mood = config.get("color_mood", "Warm neutral")
        design_notes = str(config.get("design_notes", "")).strip()
        floor_finish = config.get("floor_finish", "Auto by style")
        wall_finish = config.get("wall_finish", "Auto by style")
        for asset_key in _room_asset_keys(config.get("room_type", "")):
            final_path = asset_path(style, asset_key, design_key)
            if final_path.is_file() and final_path.stat().st_size > 10_000:
                continue
            legacy_path = asset_path(style, asset_key)
            if (_uses_default_direction(config) and legacy_path.is_file()
                    and legacy_path.stat().st_size > 10_000):
                continue
            description, _height = ASSET_SPECS[asset_key]
            ref_key = f"{design_key}__{asset_key}"
            jobs[ref_key] = {
                "key": ref_key,
                "style": style,
                "profile": profile,
                "design_key": design_key,
                "asset_key": asset_key,
                "prompt": (
                    f"{description}, designed in a cohesive {style} interior style, "
                    f"{profile.lower()} level of visual detail, premium believable "
                    f"materials and construction, {color_mood.lower()} color direction, "
                    f"coordinated with {floor_finish.lower()} flooring and "
                    f"{wall_finish.lower()} walls"
                    + (f", respecting this client brief: {design_notes}" if design_notes else "")
                ),
            }
    return list(jobs.values())


def prepare_local_assets(room_configs, progress=None):
    """Generate every missing primary furniture asset and return a summary."""
    ready, reason = runtime_status()
    if not ready:
        raise Local3DError(reason)

    jobs = _requested_jobs(room_configs)
    if not jobs:
        if progress:
            progress("Local AI furniture is already cached.")
        return {"generated": 0, "cached": True}

    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"Creating {len(jobs)} furniture reference images with local FLUX…")
    import planAI

    references = planAI.generate_asset_references(jobs, str(REFERENCE_ROOT))
    image_paths = [references[job["key"]] for job in jobs]

    signature = hashlib.sha1(
        "|".join(job["key"] for job in jobs).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = RUN_ROOT / signature
    run_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(f"Converting {len(jobs)} designs into real 3D meshes with TripoSR…")

    command = [
        str(TRIPOSR_PYTHON),
        str(TRIPOSR_ROOT / "run.py"),
        *image_paths,
        "--pretrained-model-name-or-path",
        str(TRIPOSR_MODEL),
        "--output-dir",
        str(run_dir),
        "--model-save-format",
        "glb",
        "--mc-resolution",
        "192",
        "--chunk-size",
        "2048",
        "--no-remove-bg",
    ]
    env = os.environ.copy()
    env["HF_HOME"] = str(ROOT / "hf_cache")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TRIPOSR_DINO_CONFIG"] = str(TRIPOSR_MODEL / "dino_config.json")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            creationflags=creationflags,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = detail[-1200:]
        else:
            detail = "The local conversion process exited before producing a mesh."
        raise Local3DError(
            "TripoSR could not generate the selected furniture.\n\n" + detail
        ) from exc

    generated = 0
    for index, job in enumerate(jobs):
        source = run_dir / str(index) / "mesh.glb"
        if not source.is_file():
            raise Local3DError(f"TripoSR did not produce {job['asset_key']}.")
        destination = asset_path(
            job["style"], job["asset_key"], job["design_key"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        metadata = {
            "asset": job["asset_key"],
            "style": job["style"],
            "profile": job["profile"],
            "design_key": job["design_key"],
            "source_reference": str(Path(image_paths[index]).relative_to(ROOT)),
            "generator": "FLUX.2-klein-4B + TripoSR",
        }
        with open(destination.parent / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        generated += 1

    if progress:
        progress(f"Local AI furniture ready — {generated} new 3D assets generated.")
    return {"generated": generated, "cached": False}


_MESH_CACHE = {}


def load_asset_mesh(asset_key, style, width, depth, height=None, design_key=None):
    """Load and normalize a generated GLB to the requested real-world size."""
    path = _resolved_asset_path(style, asset_key, design_key)
    if path is None:
        return None
    height = float(height or ASSET_SPECS.get(asset_key, ("", 1.0))[1])
    cache_key = (str(path), round(width, 3), round(depth, 3), round(height, 3))
    if cache_key in _MESH_CACHE:
        return [copy.deepcopy(_MESH_CACHE[cache_key])]

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        return None

    vertices = np.asarray(mesh.vertices).copy()
    source_extents = np.ptp(vertices, axis=0)
    if np.any(source_extents < 1e-6):
        return None

    target_extents = np.array([float(width), float(depth), height])
    source_rank = np.argsort(source_extents)[::-1]
    target_rank = np.argsort(target_extents)[::-1]
    aligned = np.zeros_like(vertices)
    for source_axis, target_axis in zip(source_rank, target_rank):
        aligned[:, target_axis] = vertices[:, source_axis]

    aligned -= aligned.min(axis=0)
    aligned *= target_extents / np.maximum(np.ptp(aligned, axis=0), 1e-6)
    aligned[:, 0] -= (aligned[:, 0].min() + aligned[:, 0].max()) / 2
    aligned[:, 1] -= (aligned[:, 1].min() + aligned[:, 1].max()) / 2
    aligned[:, 2] -= aligned[:, 2].min()
    mesh.vertices = o3d.utility.Vector3dVector(aligned)
    mesh.compute_vertex_normals()
    _MESH_CACHE[cache_key] = copy.deepcopy(mesh)
    return [mesh]
