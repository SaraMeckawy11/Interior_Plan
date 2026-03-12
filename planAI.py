"""
AI Interior Design Generator using Stable Diffusion with ControlNet (Canny) + Img2Img

Uses img2img so the AI STARTS from the actual 3D screenshot, naturally preserving
door/window positions. Canny ControlNet provides additional edge guidance for
wall structure.

Key insight: txt2img + Canny can't preserve doors because edge lines alone don't
tell the AI what a door looks like. Img2img lets the AI SEE the actual room and
transform it into a furnished version while keeping the layout intact.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# ===========================================================
# CONFIG
# ===========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[AI] Using device: {DEVICE}")

MODEL_SD = "Lykon/dreamshaper-8"
MODEL_CANNY = "lllyasviel/control_v11p_sd15_canny"
MODEL_DEPTH = "lllyasviel/control_v11f1p_sd15_depth"
MODEL_SEG = "lllyasviel/control_v11p_sd15_seg"

# ===========================================================
# LAZY MODEL LOADING
# ===========================================================

_models = {
    "pipe": None,
    "loaded": False
}

def load_models():
    """Load 3 ControlNets + Stable Diffusion Img2Img pipeline (called once)"""
    global _models

    if _models["loaded"]:
        return

    from transformers import pipeline

    print("[AI] Loading ControlNet models (Canny, Depth, Seg)...")
    controlnet_canny = ControlNetModel.from_pretrained(MODEL_CANNY, torch_dtype=torch.float16)
    controlnet_depth = ControlNetModel.from_pretrained(MODEL_DEPTH, torch_dtype=torch.float16)
    controlnet_seg = ControlNetModel.from_pretrained(MODEL_SEG, torch_dtype=torch.float16)

    print("[AI] Loading Stable Diffusion Img2Img + ControlNet pipeline...")
    _models["pipe"] = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        MODEL_SD,
        controlnet=[controlnet_canny, controlnet_depth, controlnet_seg],
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(DEVICE)
    
    _models["pipe"].scheduler = UniPCMultistepScheduler.from_config(_models["pipe"].scheduler.config)
    
    print("[AI] Loading Depth Estimator...")
    try:
        _models["depth_estimator"] = pipeline("depth-estimation", model="Intel/dpt-large")
    except Exception as e:
        print(f"[AI] Warning: Depth estimator failed to load: {e}")
        _models["depth_estimator"] = None

    _models["loaded"] = True
    print("[AI] All models loaded successfully!")

# ===========================================================
# IMAGE PROCESSING
# ===========================================================

def get_target_size(image):
    """Determine target size (must be divisible by 8 for SD)"""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    if w > h:
        return 1024, 768   # Landscape
    else:
        return 768, 1024   # Portrait


def get_canny_image(image, target_width, target_height):
    """
    Generate CLEAN Canny edge map: white edges on black background.

    ControlNet Canny model was trained on pure white-on-black edge maps.
    Do NOT blend with original image — that confuses the model.
    """
    resized = cv2.resize(image, (target_width, target_height),
                         interpolation=cv2.INTER_CUBIC)

    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Standard Canny — clean thresholds for structural edges
    canny = cv2.Canny(gray, 80, 160)

    # Convert to 3-channel RGB PIL Image (ControlNet expects RGB)
    canny_rgb = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_rgb), canny


def get_depth_image(image, target_width, target_height):
    """Generate depth map using transformers pipeline or fallback"""
    global _models
    if not _models.get("depth_estimator"):
        # Fallback to simple grayscale representation if not available
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_width, target_height))
        return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        
    i_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(i_rgb)
    
    depth_pil = _models["depth_estimator"](pil_image)["depth"]
    depth_rgb = Image.fromarray(cv2.cvtColor(np.array(depth_pil), cv2.COLOR_GRAY2RGB))
    return depth_rgb.resize((target_width, target_height), Image.Resampling.LANCZOS)


def get_seg_image(
    image,
    target_width,
    target_height,
    visible_openings,
    camera_params,
    src_w,
    src_h,
    crop_box=None
):
    """Build a semantic segmentation mask using ADE20K color palette for ControlNet"""
    # ADE20K colors: Wall is [120, 120, 120], Floor is [80, 50, 50]
    seg = np.full((target_height, target_width, 3), [120, 120, 120], dtype=np.uint8)
    
    # Simple floor approximation: bottom 40% is floor
    floor_start = int(target_height * 0.6)
    seg[floor_start:, :] = [80, 50, 50]
    
    if not visible_openings or not camera_params:
        return Image.fromarray(seg)
        
    intrinsic = camera_params.get("intrinsic")
    extrinsic = camera_params.get("extrinsic")
    if intrinsic is None or extrinsic is None:
        return Image.fromarray(seg)

    if crop_box is not None:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        source_w = max(1, crop_x2 - crop_x1)
        source_h = max(1, crop_y2 - crop_y1)
    else:
        crop_x1, crop_y1 = 0, 0
        source_w, source_h = src_w, src_h

    sx = target_width / source_w
    sy = target_height / source_h

    for op in visible_openings:
        corners_3d = op.get("corners_3d")
        op_type = op.get("type", "door")

        if not corners_3d:
            continue

        pts_2d = []
        for corner in corners_3d:
            px = _project_point(corner, intrinsic, extrinsic)
            if px is None:
                break

            u = px[0] - crop_x1
            v = px[1] - crop_y1
            pts_2d.append((u * sx, v * sy))

        if len(pts_2d) < 4:
            continue

        pts_np = np.array(pts_2d, dtype=np.float32)
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, target_width - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, target_height - 1)
        pts_int = pts_np.astype(np.int32)
        
        # ADE20K colors (RGB): Door is [8, 255, 51], Windowpane is [230, 230, 230]
        color = (8, 255, 51) if op_type == "door" else (230, 230, 230)
        cv2.fillPoly(seg, [pts_int], color=color)

    return Image.fromarray(seg)


# ===========================================================
# 3D → 2D OPENING PROJECTION
# ===========================================================

def _project_point(xyz, intrinsic, extrinsic):
    """Project a single 3D world point [x, y, z] to 2D pixel [u, v].
    
    Open3D world coordinates: X = right (floor plan x),
                               Y = forward (floor plan y),
                               Z = up (wall height)
    Extrinsic is the 4×4 world-to-camera transform from Open3D.
    
    Returns (u, v) pixel coords, or None if point is behind the camera.
    """
    K = np.array(intrinsic)   # 3×3
    E = np.array(extrinsic)   # 4×4  (world → camera)

    # Homogeneous world point
    world_h = np.array([xyz[0], xyz[1], xyz[2], 1.0])

    # Apply extrinsic: get camera-space coordinates
    cam = E @ world_h          # shape (4,)

    # Open3D extrinsic gives a right-hand camera: Z points away from camera
    # (into the scene). Point is visible only if cam[2] > 0.
    if cam[2] <= 0:
        return None

    # Apply intrinsic (K acts on (X_c, Y_c, Z_c))
    uv = K @ cam[:3]
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    return (u, v)


def _project_opening_points(corners_3d, intrinsic, extrinsic, crop_box=None):
    """Project 3D opening corners and optionally remap them into a cropped source view."""
    points = []
    crop_x1, crop_y1 = (crop_box[0], crop_box[1]) if crop_box is not None else (0, 0)

    for corner in corners_3d:
        px = _project_point(corner, intrinsic, extrinsic)
        if px is None:
            return None
        points.append((px[0] - crop_x1, px[1] - crop_y1))

    return points


def draw_openings_on_canny(canny_pil, visible_openings, camera_params,
                            src_w, src_h, target_w, target_h, crop_box=None):
    """
    Project 3D door/window corners to 2D and draw bright white outlines
    (plus a lighter fill) on the Canny conditioning image.

    This tells ControlNet EXACTLY where the door/window is — even if
    it's only a thin sliver seen at a sharp perspective angle.

    Args:
        canny_pil:        PIL Image (RGB), the Canny edge map at target size
        visible_openings: list of dicts with keys: type, corners_3d
        camera_params:    dict with intrinsic, extrinsic, width, height
        src_w, src_h:     original screenshot pixel dimensions
        target_w, target_h: AI target dimensions (e.g. 1024×768)

    Returns:
        PIL Image (RGB) with door/window shapes drawn in
    """
    if not visible_openings or not camera_params:
        return canny_pil

    intrinsic = camera_params.get("intrinsic")
    extrinsic = camera_params.get("extrinsic")
    cam_w     = camera_params.get("width",  src_w)
    cam_h     = camera_params.get("height", src_h)

    if intrinsic is None or extrinsic is None:
        return canny_pil

    if crop_box is not None:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        source_w = max(1, crop_x2 - crop_x1)
        source_h = max(1, crop_y2 - crop_y1)
    else:
        source_w, source_h = src_w, src_h

    # Scale factor: current source view → target AI size
    sx = target_w / source_w
    sy = target_h / source_h

    canny_np = np.array(canny_pil)   # (H, W, 3) uint8

    for op in visible_openings:
        corners_3d = op.get("corners_3d")
        op_type    = op.get("type", "door")

        if not corners_3d:
            continue

        # Project all 4 corners
        pts_2d = _project_opening_points(corners_3d, intrinsic, extrinsic, crop_box=crop_box)
        if pts_2d is not None:
            pts_2d = [(u * sx, v * sy) for u, v in pts_2d]

        if not pts_2d or len(pts_2d) < 4:
            print(f"[AI] Opening {op_type} has corner(s) behind camera, skipping projection")
            continue

        # Clamp to image bounds
        pts_np = np.array(pts_2d, dtype=np.float32)
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, target_w - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, target_h - 1)

        pts_int = pts_np.astype(np.int32)

        # Compute pixel-space bounding box for diagnostics
        x_min, y_min = pts_int[:, 0].min(), pts_int[:, 1].min()
        x_max, y_max = pts_int[:, 0].max(), pts_int[:, 1].max()
        w_px = x_max - x_min
        h_px = y_max - y_min
        print(f"[AI] Projecting {op_type}: 2D box [{x_min},{y_min}]->[{x_max},{y_max}]  ({w_px}x{h_px}px)")

        if w_px < 3 and h_px < 3:
            print("[AI]   too small after projection, skipping")
            continue

        # Draw clear outlines only — windows use a simple rectangle to avoid being
        # interpreted as a TV or wall panel.
        thickness = max(3, min(w_px, h_px) // 5 + 2)
        cv2.polylines(canny_np, [pts_int], isClosed=True,
                      color=(255, 255, 255), thickness=thickness)

        if op_type == "door":
            cx     = int((pts_int[0, 0] + pts_int[1, 0]) / 2)
            cy_bot = int((pts_int[0, 1] + pts_int[1, 1]) / 2)
            cy_top = int((pts_int[2, 1] + pts_int[3, 1]) / 2)
            cv2.line(canny_np, (cx, cy_bot), (cx, cy_top),
                     (255, 255, 255), max(2, thickness // 2))
        else:  # window
            sill_y = int(max(y_min, y_max - max(2, thickness // 2)))
            cv2.line(canny_np, (x_min, sill_y), (x_max, sill_y),
                     (255, 255, 255), max(2, thickness // 3))

    return Image.fromarray(canny_np)


def draw_openings_on_init_image(init_pil, visible_openings, camera_params,
                                src_w, src_h, target_w, target_h, crop_box=None):
    """Overlay clear door/window cues on the init image so img2img preserves them."""
    if not visible_openings or not camera_params:
        return init_pil

    intrinsic = camera_params.get("intrinsic")
    extrinsic = camera_params.get("extrinsic")
    if intrinsic is None or extrinsic is None:
        return init_pil

    if crop_box is not None:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        source_w = max(1, crop_x2 - crop_x1)
        source_h = max(1, crop_y2 - crop_y1)
    else:
        source_w, source_h = src_w, src_h

    sx = target_w / source_w
    sy = target_h / source_h

    init_np = np.array(init_pil).copy()
    overlay = init_np.copy()

    for op in visible_openings:
        corners_3d = op.get("corners_3d")
        op_type = op.get("type", "door")
        if not corners_3d:
            continue

        pts_2d = _project_opening_points(corners_3d, intrinsic, extrinsic, crop_box=crop_box)
        if not pts_2d or len(pts_2d) < 4:
            continue

        pts_np = np.array([(u * sx, v * sy) for u, v in pts_2d], dtype=np.float32)
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, target_w - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, target_h - 1)
        pts_int = pts_np.astype(np.int32)

        if op_type == "window":
            # Light outline only — the 3D render already has the bright glass/sky
            cv2.polylines(overlay, [pts_int], isClosed=True, color=(255, 255, 255), thickness=3)
        else:
            cv2.fillPoly(overlay, [pts_int], color=(135, 92, 60))
            cv2.polylines(overlay, [pts_int], isClosed=True, color=(78, 50, 30), thickness=5)
            x_min, y_min = pts_int[:, 0].min(), pts_int[:, 1].min()
            x_max, y_max = pts_int[:, 0].max(), pts_int[:, 1].max()
            cx = int((x_min + x_max) / 2)
            cv2.line(overlay, (cx, y_min), (cx, y_max), color=(95, 62, 38), thickness=2)

    init_np = cv2.addWeighted(overlay, 0.38, init_np, 0.62, 0)
    return Image.fromarray(init_np)


def prepare_init_image(image, target_width, target_height):
    """
    Prepare the screenshot as the img2img starting image.
    The AI starts from this and transforms it, so doors/windows are preserved.
    """
    resized = cv2.resize(image, (target_width, target_height),
                         interpolation=cv2.INTER_CUBIC)

    if len(resized.shape) == 3 and resized.shape[2] == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return Image.fromarray(resized)

# ===========================================================
# PROMPTS
# ===========================================================

ROOM_PROMPTS = {
    "Living Room": (
        "(beautifully furnished living room:1.6), modern interior design, photorealistic, "
        "real furniture, sofa, coffee table, rug, shelving, lamps, plants, wall art, "
        "tasteful decor, balanced composition, cozy lighting"
    ),
    "Bedroom": (
        "(beautifully furnished bedroom:1.6), modern interior design, photorealistic, "
        "real furniture, bed, nightstands, dresser, rug, lamps, decor, plants, cozy lighting"
    ),
    "Kitchen": (
        "(beautifully furnished kitchen:1.6), modern interior design, photorealistic, "
        "real cabinets, countertops, island, stools, appliances, pendant lights, styled decor"
    ),
    "Bathroom": (
        "(beautifully furnished bathroom:1.5), modern interior design, photorealistic, "
        "vanity, mirror, towels, accessories, fixtures, styled decor, bright lighting"
    ),
    "Office": (
        "(beautifully furnished home office:1.6), modern interior design, photorealistic, "
        "desk, office chair, shelves, computer, lamp, decor, plants, balanced layout"
    ),
    "Dining Room": (
        "(beautifully furnished dining room:1.6), modern interior design, photorealistic, "
        "dining table, chairs, sideboard, pendant light, rug, decor, warm lighting"
    ),
    "Kids Room": (
        "(beautifully furnished kids room:1.5), modern interior design, photorealistic, "
        "kids bed, toy storage, desk, playful decor, rug, cheerful lighting"
    ),
    "Guest Room": (
        "(beautifully furnished guest room:1.6), modern interior design, photorealistic, "
        "bed, nightstand, dresser, mirror, seating, decor, warm lighting"
    ),
    "Studio": (
        "(beautifully furnished studio apartment:1.6), modern interior design, photorealistic, "
        "sofa, bed zone, dining area, storage, decor, balanced furniture layout"
    )
}

BASE_NEGATIVE_PROMPT = (
    "blurry, lowres, distorted, bad lighting, wrong perspective, "
    "changing room structure, moving walls, deformed architecture, "
    "empty room, unfurnished, bare walls, missing furniture, sparse furniture, "
    "television, tv, monitor, wall panel, wood panel, giant window, oversized window, "
    "full glass wall, floor-to-ceiling glazing, moved window, moved door, low detail"
)

# ===========================================================
# MAIN GENERATION FUNCTION - IMG2IMG + CANNY CONTROLNET
# ===========================================================

def generate_furnished_plan(base_rgb, room_masks, doors, windows, room_types,
                            px_per_m=100, has_doors=None, has_windows=None,
                            visible_openings=None, camera_params=None):
    """
    Generate furnished room images using img2img + Canny ControlNet.

    The img2img approach starts from the actual 3D screenshot, so the AI
    naturally preserves the room layout including door/window positions.
    Canny ControlNet provides additional edge guidance for wall structure.
    
    visible_openings: list of dicts with 3D corner geometry for projection
    camera_params:    dict with intrinsic/extrinsic matrices from Open3D
    """
    load_models()
    output = []
    base_image = base_rgb if isinstance(base_rgb, np.ndarray) else np.array(base_rgb)

    if not room_masks or len(room_masks) == 0:
        room_masks = [None]
        if not room_types:
            room_types = ["Living Room"]

    for idx, mask in enumerate(room_masks):
        room_type = room_types[idx] if idx < len(room_types) else "Living Room"
        print(f"\n==== ROOM {idx+1} - {room_type} ====\n")

        try:
            # Get room image region
            if mask is not None:
                ys, xs = np.where(mask == 255)
                if xs.size == 0 or ys.size == 0:
                    print(f"[AI] Skipping room {idx+1}: empty mask")
                    continue
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                room = base_image[y1:y2+1, x1:x2+1]
                crop_box = (x1, y1, x2 + 1, y2 + 1)
            else:
                room = base_image
                crop_box = None

            # Determine target size
            target_w, target_h = get_target_size(room)
            src_h, src_w = room.shape[:2]
            print(f"[AI] Room size: {src_w}x{src_h}")
            print(f"[AI] Target size: {target_w}x{target_h}")

            # Prepare the screenshot as img2img starting image
            init_image = prepare_init_image(room, target_w, target_h)
            init_image = draw_openings_on_init_image(
                init_image, visible_openings, camera_params,
                src_w, src_h, target_w, target_h, crop_box=crop_box
            )

            # Generate clean Canny edge map (white on black only)
            canny_image, canny_raw = get_canny_image(room, target_w, target_h)
            edge_ratio = np.sum(canny_raw > 0) / canny_raw.size
            print(f"[AI] Canny edge ratio: {edge_ratio:.4f}")

            # Project 3D door/window positions onto Canny image
            if visible_openings and camera_params:
                print(f"[AI] Drawing {len(visible_openings)} opening(s) onto Canny map...")
                canny_image = draw_openings_on_canny(
                    canny_image, visible_openings, camera_params,
                    src_w, src_h, target_w, target_h, crop_box=crop_box
                )
            else:
                print("[AI] No opening geometry available for Canny projection")

            # Generate Depth and Seg images
            depth_image = get_depth_image(room, target_w, target_h)
            seg_image = get_seg_image(
                room, target_w, target_h, visible_openings, camera_params,
                src_w, src_h, crop_box=crop_box
            )

            # Determine door/window info from floor plan data
            if has_doors is not None:
                actual_has_doors = has_doors
            else:
                actual_has_doors = len(doors) > 0

            if has_windows is not None:
                actual_has_windows = has_windows
            else:
                actual_has_windows = False

            print(f"[AI] Layout: has_doors={actual_has_doors}, has_windows={actual_has_windows}")

            # Build prompt — only mention windows/curtains if they actually exist
            prompt = ROOM_PROMPTS.get(room_type, ROOM_PROMPTS["Living Room"])
            negative_prompt = BASE_NEGATIVE_PROMPT

            if actual_has_windows:
                # Windows exist — preserve them, but keep prompt conservative to avoid scene drift
                prompt += ", keep existing window exactly where shown, same size and wall position, natural daylight, optional light sheer curtains"
                negative_prompt += ", no windows, blocked windows, moved window, oversized window, giant window, tv on wall"
            else:
                # NO windows in this room — block all window/curtain hallucinations
                negative_prompt += (
                    ", window, windows, windowpane, glass pane, curtain, curtains, "
                    "drapes, blinds, shutters, daylight from window, sunlight through window"
                )

            if actual_has_doors:
                prompt += ", keep existing door exactly where shown, same size and wall position, preserve the visible doorway and door frame"
                negative_prompt += ", no door, missing door, doorless, covered doorway, wall where door should be, furniture blocking doorway"
            else:
                negative_prompt += ", door, doorway, door frame, entrance"

            prompt += ", realistic furnished room, balanced furniture, keep room architecture unchanged, preserve all openings and wall positions"

            print(f"[AI] Prompt: {prompt[:120]}...")
            print(f"[AI] Negative: {negative_prompt[:100]}...")

            # ── Img2Img + ControlNet call ──
            print("[AI] Generating furnished room (img2img + Multi-ControlNet)...")
            result = _models["pipe"](
                prompt=prompt,
                image=init_image,
                control_image=[canny_image, depth_image, seg_image],
                controlnet_conditioning_scale=[0.01, 0.40, 0.15],
                strength=0.9,
                num_inference_steps=36,
                guidance_scale=9,
                negative_prompt=negative_prompt,
                generator=torch.manual_seed(576906284)
            )

            furnished_img = result.images[0]

            output.append({
                "furnished": furnished_img,
                "room_type": room_type,
                "index": idx
            })

            print(f"[AI] Room {idx+1} generated successfully!")

        except Exception as e:
            print(f"[AI ERROR] Failed to generate room {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return output


# ===========================================================
# STANDALONE TEST
# ===========================================================

if __name__ == "__main__":
    test_dir = r"C:\Users\Lenovo\Desktop\Interior_plan\room_designs\screenshots"

    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        if files:
            img_path = os.path.join(test_dir, files[0])
            print(f"Testing with: {img_path}")

            img = cv2.imread(img_path)
            print(f"[INFO] Image loaded: {img.shape}")

            results = generate_furnished_plan(
                base_rgb=img,
                room_masks=[],
                doors=[],
                windows=[],
                room_types=["Living Room"],
                has_doors=True,
                has_windows=False
            )

            if results:
                results[0]["furnished"].save("test_output.png")
                results[0]["furnished"].show()
                print("[OK] Test successful!")
            else:
                print("[ERROR] Test failed - no results")
        else:
            print("No screenshot files found for testing")
    else:
        print("Test directory not found")
