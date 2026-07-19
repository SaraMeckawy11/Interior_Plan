"""
2D plan -> furnished 3D walkthrough.

Pipeline (2D-to-3D-first, then furnish — a deterministic approach that gives a
true walkable model and respects the drawn plan exactly):

  1. Convert the drawn room polygons / doors / windows to REAL-WORLD meters
     (100 px = 1 m, no artistic scaling), and build walls with door and
     window openings cut out (reusing plan3d's realistic door/window meshes).
  2. Apply the user's style, color mood, flooring, wall finish and personal
     brief to architecture, material palette, focal wall, ceiling detail,
     window treatment, lighting, furniture and decor.
  3. First-person walkthrough: WASD to walk, arrow keys to look around.
     Collision keeps you inside the apartment and lets you pass between
     rooms only through doors.

Public entry point:  launch_walkthrough(...)
Standalone demo:     python plan_walkthrough.py          (interactive)
                     python plan_walkthrough.py --capture out_dir  (renders
                     verification screenshots and exits)
"""

import hashlib
import math
import os
import random
import statistics
import time

import numpy as np
import open3d as o3d
from shapely.geometry import LineString, Point, Polygon, box as shp_box
from shapely.ops import unary_union
from shapely import affinity
from shapely.prepared import prep

from plan3d import (
    create_window_geometry,
    floor_mesh,
    merge_edges,
    wall_segment,
    DOOR_HEIGHT,
    WINDOW_SILL,
    WINDOW_HEIGHT,
    WALL_THICKNESS,
)
from archviz_materials import (
    apply_archviz_material,
    floor_material,
    material_record_for_mesh,
    wall_material,
)

# ================= WALKTHROUGH CONFIG (real-world scale) =================
WALL_H = 2.8               # ceiling height (m)
MIN_DOOR_W = 1.0           # every doorway is at least this wide (m) so the
                           # player always fits through — real interior doors
                           # are ~0.9 m; we give a little extra clearance
MIN_WINDOW_W = 0.7
OPENING_EDGE_TOL = 0.6     # door/window is assigned to every wall closer than this
WALL_GAP = 0.16            # furniture stand-off from the wall line

SCALE_BOOST = 1.15         # enlarge the building shell relative to the walker
                           # (rooms feel bigger; furniture stays real-size)
CAMERA_FOV = 68.0          # professional interior lens without fisheye stretch
GHOST_MARGIN = 3.0         # keep free-explore mode close to the apartment

EYE_HEIGHT = 1.62
WALK_SPEED = 2.8           # m/s (a touch faster to suit the larger rooms)
RUN_MULT = 2.0             # hold Shift to move faster
TURN_SPEED = 1.9           # rad/s (arrow-key look)
MOUSE_SENS = 0.0042        # rad per pixel (drag to look)
BODY_RADIUS = 0.16         # collision inset (slim so doorways feel roomy)

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "room_designs", "walkthrough_captures")

CEILING_COLOR = [0.96, 0.96, 0.95]

# ---- Baked lighting rig -------------------------------------------------
# Realtime lighting in Open3D's legacy view produces a "flashlight" headlight
# and specular hot-spots on flat walls. Instead we BAKE soft directional
# shading into per-face vertex colours (light stays OFF), which gives clean,
# even, artefact-free interior lighting that reads as proper 3D form.
def _norm(v):
    v = np.array(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-9)


_KEY_DIR = _norm([0.35, 0.55, 0.75])    # warm key light (window/upper-front)
_FILL_DIR = _norm([-0.6, -0.35, 0.45])  # cool fill from the opposite side
_AMBIENT = 0.66                          # base fill so shadows never go black
_KEY_I = 0.42
_FILL_I = 0.16
_SKY_I = 0.14                            # extra light on up-facing surfaces


# ================= STYLE PALETTES =================
# Same style vocabulary as the AI gallery (planAI.STYLE_SPECS), expressed as
# 3D material colors. Every furnisher picks its colors from here, so the
# walkthrough reflects the user's chosen design style per room.

PALETTES = {
    "modern": dict(
        wall=[0.93, 0.90, 0.85], floor=[0.70, 0.53, 0.35],
        rug=[0.82, 0.74, 0.60], sofa=[0.87, 0.83, 0.76],
        cushion=[0.75, 0.45, 0.28], wood=[0.63, 0.48, 0.33],
        wood_dark=[0.42, 0.30, 0.20], metal=[0.78, 0.65, 0.35],
        accent=[0.75, 0.45, 0.28], table=[0.88, 0.85, 0.78],
        cabinet=[0.90, 0.88, 0.83], counter=[0.92, 0.90, 0.86],
        shade=[0.93, 0.88, 0.75],
    ),
    "classic": dict(
        wall=[0.91, 0.88, 0.81], floor=[0.55, 0.38, 0.24],
        rug=[0.58, 0.38, 0.33], sofa=[0.72, 0.64, 0.52],
        cushion=[0.45, 0.28, 0.25], wood=[0.45, 0.30, 0.18],
        wood_dark=[0.32, 0.20, 0.12], metal=[0.80, 0.66, 0.35],
        accent=[0.45, 0.28, 0.25], table=[0.90, 0.89, 0.86],
        cabinet=[0.85, 0.80, 0.70], counter=[0.90, 0.89, 0.86],
        shade=[0.92, 0.86, 0.72],
    ),
    "scandinavian": dict(
        wall=[0.96, 0.96, 0.94], floor=[0.86, 0.79, 0.66],
        rug=[0.89, 0.87, 0.83], sofa=[0.72, 0.73, 0.72],
        cushion=[0.55, 0.60, 0.62], wood=[0.83, 0.73, 0.56],
        wood_dark=[0.60, 0.50, 0.38], metal=[0.18, 0.18, 0.18],
        accent=[0.45, 0.55, 0.58], table=[0.85, 0.76, 0.60],
        cabinet=[0.94, 0.94, 0.92], counter=[0.90, 0.88, 0.84],
        shade=[0.95, 0.94, 0.90],
    ),
    "boho": dict(
        wall=[0.95, 0.90, 0.81], floor=[0.62, 0.45, 0.29],
        rug=[0.72, 0.50, 0.34], sofa=[0.70, 0.42, 0.27],
        cushion=[0.85, 0.68, 0.40], wood=[0.68, 0.52, 0.32],
        wood_dark=[0.48, 0.34, 0.20], metal=[0.55, 0.40, 0.25],
        accent=[0.55, 0.58, 0.35], table=[0.72, 0.57, 0.36],
        cabinet=[0.80, 0.66, 0.46], counter=[0.85, 0.78, 0.64],
        shade=[0.88, 0.74, 0.52],
    ),
    "japandi": dict(
        wall=[0.93, 0.91, 0.86], floor=[0.82, 0.74, 0.60],
        rug=[0.85, 0.82, 0.75], sofa=[0.79, 0.75, 0.67],
        cushion=[0.35, 0.32, 0.28], wood=[0.72, 0.62, 0.48],
        wood_dark=[0.28, 0.23, 0.18], metal=[0.18, 0.17, 0.16],
        accent=[0.40, 0.36, 0.30], table=[0.32, 0.26, 0.20],
        cabinet=[0.88, 0.84, 0.76], counter=[0.84, 0.80, 0.72],
        shade=[0.94, 0.91, 0.84],
    ),
    "industrial": dict(
        wall=[0.78, 0.76, 0.73], floor=[0.42, 0.31, 0.21],
        rug=[0.60, 0.56, 0.50], sofa=[0.56, 0.33, 0.16],
        cushion=[0.30, 0.30, 0.32], wood=[0.55, 0.42, 0.28],
        wood_dark=[0.35, 0.27, 0.18], metal=[0.13, 0.13, 0.14],
        accent=[0.65, 0.50, 0.30], table=[0.50, 0.38, 0.25],
        cabinet=[0.35, 0.35, 0.37], counter=[0.70, 0.70, 0.68],
        shade=[0.55, 0.52, 0.46],
    ),
    "minimalist": dict(
        wall=[0.95, 0.94, 0.92], floor=[0.87, 0.83, 0.75],
        rug=[0.90, 0.88, 0.84], sofa=[0.83, 0.81, 0.77],
        cushion=[0.68, 0.66, 0.62], wood=[0.80, 0.73, 0.60],
        wood_dark=[0.55, 0.50, 0.42], metal=[0.55, 0.55, 0.55],
        accent=[0.60, 0.58, 0.54], table=[0.82, 0.78, 0.70],
        cabinet=[0.93, 0.92, 0.90], counter=[0.90, 0.89, 0.86],
        shade=[0.94, 0.93, 0.90],
    ),
}

# Map the gallery UI's style labels onto palette keys (same aliases planAI uses).
STYLE_ALIASES = {
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

# A style is resolved as one coordinated room kit. This prevents independently
# chosen furniture, rugs, curtains and wall decor from drifting into different
# visual languages while still allowing every part to be overridden in the UI.
STYLE_DESIGN_PRESETS = {
    "modern": dict(
        rug_design="bordered",
        curtain_design="linen drapes",
        decor_set="sculptural",
        wall_treatment="accent color",
    ),
    "classic": dict(
        rug_design="vintage pattern",
        curtain_design="layered sheers + drapes",
        decor_set="layered",
        wall_treatment="panel moulding",
    ),
    "scandinavian": dict(
        rug_design="plain woven",
        curtain_design="sheer panels",
        decor_set="art & greenery",
        wall_treatment="warm paint",
    ),
    "boho": dict(
        rug_design="vintage pattern",
        curtain_design="layered sheers + drapes",
        decor_set="art & greenery",
        wall_treatment="limewash",
    ),
    "japandi": dict(
        rug_design="plain woven",
        curtain_design="linen drapes",
        decor_set="sculptural",
        wall_treatment="wood slats",
    ),
    "industrial": dict(
        rug_design="geometric",
        curtain_design="linen drapes",
        decor_set="sculptural",
        wall_treatment="concrete",
    ),
    "minimalist": dict(
        rug_design="plain woven",
        curtain_design="sheer panels",
        decor_set="minimal",
        wall_treatment="warm paint",
    ),
}


def room_design_choices(config=None, room_type=""):
    """Resolve Auto selections into a coherent, room-appropriate style kit."""
    config = config or {}
    style_key = STYLE_ALIASES.get(
        str(config.get("style", "Modern")).lower().strip(), "modern"
    )
    choices = dict(STYLE_DESIGN_PRESETS[style_key])
    automatic = {}
    for key in ("rug_design", "curtain_design", "decor_set"):
        selected = str(config.get(key, "Auto by style")).lower().strip()
        automatic[key] = selected == "auto by style"
        if not automatic[key]:
            choices[key] = selected

    room_key = (room_type or config.get("room_type", "")).lower()
    if automatic["rug_design"] and any(
        word in room_key
        for word in (
            "kitchen", "laundry", "utility", "hall", "corridor", "entry"
        )
    ):
        choices["rug_design"] = "none"
    if automatic["rug_design"] and "bath" in room_key:
        choices["rug_design"] = "plain woven"
    if automatic["curtain_design"] and "bath" in room_key:
        choices["curtain_design"] = "none"
    elif automatic["curtain_design"] and any(
        word in room_key for word in ("hall", "corridor", "entry", "utility")
    ):
        choices["curtain_design"] = "none"
    elif automatic["curtain_design"] and "kitchen" in room_key:
        choices["curtain_design"] = "sheer panels"
    if automatic["decor_set"] and any(
        word in room_key
        for word in (
            "kitchen", "bath", "laundry", "utility", "hall", "corridor"
        )
    ):
        choices["decor_set"] = "minimal"
    if "bath" in room_key:
        choices["wall_treatment"] = "large tile"
    elif any(word in room_key for word in ("kitchen", "laundry", "utility")):
        choices["wall_treatment"] = "warm paint"

    profile = str(config.get("design_profile", "Curated")).lower()
    if automatic["curtain_design"] and profile == "airy":
        choices["curtain_design"] = "sheer panels"
    elif automatic["curtain_design"] and profile == "layered":
        choices["curtain_design"] = "layered sheers + drapes"
    if automatic["decor_set"] and profile == "airy":
        choices["decor_set"] = "minimal"
    elif automatic["decor_set"] and profile == "layered":
        choices["decor_set"] = "layered"
    choices["style_key"] = style_key
    return choices


GREEN_FOLIAGE = [0.25, 0.42, 0.22]
TERRACOTTA = [0.72, 0.44, 0.30]
WHITE_SOFT = [0.93, 0.92, 0.90]
SCREEN_DARK = [0.05, 0.05, 0.07]


def _mix_color(base, generated, generated_weight):
    base = np.asarray(base, dtype=float)
    generated = np.asarray(generated, dtype=float)
    return np.clip(
        base * (1.0 - generated_weight) + generated * generated_weight,
        0.03,
        0.98,
    ).tolist()


MOOD_COLORS = {
    "warm neutral": dict(
        wall=[0.91, 0.86, 0.77], accent=[0.66, 0.43, 0.28],
        secondary=[0.76, 0.69, 0.58],
    ),
    "cool neutral": dict(
        wall=[0.88, 0.91, 0.92], accent=[0.35, 0.48, 0.56],
        secondary=[0.67, 0.72, 0.74],
    ),
    "earthy natural": dict(
        wall=[0.87, 0.82, 0.70], accent=[0.35, 0.46, 0.28],
        secondary=[0.67, 0.54, 0.38],
    ),
    "light and airy": dict(
        wall=[0.96, 0.95, 0.91], accent=[0.58, 0.67, 0.68],
        secondary=[0.84, 0.82, 0.75],
    ),
    "monochrome": dict(
        wall=[0.90, 0.90, 0.89], accent=[0.24, 0.25, 0.26],
        secondary=[0.60, 0.60, 0.59],
    ),
    "bold accents": dict(
        wall=[0.91, 0.89, 0.84], accent=[0.16, 0.36, 0.52],
        secondary=[0.72, 0.47, 0.25],
    ),
}

FLOOR_FINISH_COLORS = {
    "light oak": [0.82, 0.72, 0.56],
    "warm oak": [0.66, 0.47, 0.29],
    "dark walnut": [0.31, 0.22, 0.16],
    "natural stone": [0.69, 0.66, 0.59],
    "polished concrete": [0.48, 0.49, 0.49],
    "terrazzo": [0.76, 0.73, 0.67],
    "large tile": [0.72, 0.70, 0.66],
}

WALL_FINISH_COLORS = {
    "warm paint": [0.92, 0.86, 0.76],
    "cool paint": [0.89, 0.92, 0.93],
    "limewash": [0.84, 0.79, 0.69],
    "wallpaper": [0.88, 0.84, 0.76],
    "wood slats": [0.78, 0.72, 0.62],
    "panel moulding": [0.89, 0.87, 0.81],
    "concrete": [0.63, 0.63, 0.61],
    "accent color": [0.79, 0.72, 0.64],
}

BRIEF_COLOR_HINTS = {
    "green": [0.29, 0.43, 0.29],
    "olive": [0.39, 0.43, 0.23],
    "blue": [0.25, 0.42, 0.58],
    "navy": [0.14, 0.23, 0.37],
    "terracotta": [0.68, 0.34, 0.22],
    "rust": [0.58, 0.28, 0.17],
    "burgundy": [0.39, 0.13, 0.18],
    "mustard": [0.72, 0.52, 0.14],
    "black": [0.11, 0.11, 0.12],
}


def get_palette(style_label, config=None):
    key = STYLE_ALIASES.get((style_label or "").lower().strip(), "modern")
    palette = {
        name: list(color)
        for name, color in PALETTES.get(key, PALETTES["modern"]).items()
    }
    config = config or {}
    mood = MOOD_COLORS.get(
        str(config.get("color_mood", "Warm neutral")).lower()
    )
    if mood:
        palette["wall"] = _mix_color(palette["wall"], mood["wall"], 0.34)
        palette["accent"] = _mix_color(
            palette["accent"], mood["accent"], 0.58
        )
        palette["cushion"] = _mix_color(
            palette["cushion"], mood["accent"], 0.42
        )
        for name in ("sofa", "rug", "cabinet", "shade"):
            palette[name] = _mix_color(
                palette[name], mood["secondary"], 0.22
            )

    floor_finish = str(config.get("floor_finish", "Auto by style")).lower()
    if floor_finish in FLOOR_FINISH_COLORS:
        floor_color = FLOOR_FINISH_COLORS[floor_finish]
        palette["floor"] = list(floor_color)
        palette["wood"] = _mix_color(palette["wood"], floor_color, 0.72)
        palette["wood_dark"] = _shade(floor_color, 0.58)

    wall_finish = str(config.get("wall_finish", "Auto by style")).lower()
    if wall_finish in WALL_FINISH_COLORS:
        palette["wall"] = _mix_color(
            palette["wall"], WALL_FINISH_COLORS[wall_finish], 0.72
        )

    brief = str(config.get("design_notes", "")).lower()
    for word, color in BRIEF_COLOR_HINTS.items():
        if word in brief:
            palette["accent"] = _mix_color(palette["accent"], color, 0.76)
            palette["cushion"] = _mix_color(palette["cushion"], color, 0.55)
            break
    if "walnut" in brief:
        palette["wood"] = [0.36, 0.24, 0.16]
        palette["wood_dark"] = [0.22, 0.14, 0.10]
        if floor_finish == "auto by style":
            palette["floor"] = [0.41, 0.29, 0.20]
    elif "light oak" in brief:
        palette["wood"] = [0.82, 0.72, 0.56]
        palette["floor"] = [0.84, 0.76, 0.63]
    elif "oak" in brief:
        palette["wood"] = [0.67, 0.50, 0.32]
    if "concrete" in brief:
        palette["floor"] = _mix_color(palette["floor"], [0.48, 0.49, 0.49], 0.75)
    if "marble" in brief or "stone" in brief:
        palette["counter"] = [0.88, 0.87, 0.83]
        palette["table"] = _mix_color(palette["table"], [0.82, 0.80, 0.75], 0.60)
    if "brass" in brief or "gold" in brief:
        palette["metal"] = [0.72, 0.55, 0.24]

    palette["ceiling"] = _mix_color(CEILING_COLOR, palette["wall"], 0.10)
    palette["floor_finish"] = floor_finish
    palette["wall_finish"] = wall_finish
    return palette


# ================= SCALE CALIBRATION =================
STANDARD_DOOR_M = 0.9          # doors on floor plans are ~0.9 m wide
TYPICAL_ROOM_AREA_M2 = 12.0    # median room (baths included) ≈ 12 m²


def estimate_px_per_m(rooms_px, doors_px, default=100.0):
    """Deduce the plan's pixel scale so the 3D comes out at real-world size.

    Two anchors: door lengths (standard door ≈ 0.9 m) and median room area
    (≈ 12 m²). When they agree, doors win — they are the more precise
    anchor. When they disagree strongly the plan's door symbols are not to
    scale (common in decorative plans), so room sizes win: rooms are what
    the user actually experiences, and doors get widened to a minimum
    walkable width in 3D anyway.
    """
    areas_px = [abs(Polygon([(p[0], p[1]) for p in poly]).area)
                for poly in rooms_px if len(poly) >= 3]
    areas_px = [a for a in areas_px if a > 100]
    med_area_px = statistics.median(areas_px) if areas_px else None
    ppm_area = (math.sqrt(med_area_px / TYPICAL_ROOM_AREA_M2)
                if med_area_px else None)

    door_px = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in doors_px]
    door_px = [d for d in door_px if d > 5]
    ppm_doors = (statistics.median(door_px) / STANDARD_DOOR_M
                 if door_px else None)

    if ppm_doors and ppm_area:
        ratio = ppm_doors / ppm_area
        if 0.8 <= ratio <= 1.25:
            print(f"[WALK] Scale from doors: {ppm_doors:.1f} px/m "
                  f"(median room {med_area_px / ppm_doors ** 2:.1f} m^2)")
            return ppm_doors
        print(f"[WALK] Door symbols not to scale (door {ppm_doors:.1f} vs "
              f"room {ppm_area:.1f} px/m) -> using room sizes: "
              f"{ppm_area:.1f} px/m")
        return ppm_area
    if ppm_doors:
        print(f"[WALK] Scale from doors: {ppm_doors:.1f} px/m")
        return ppm_doors
    if ppm_area:
        print(f"[WALK] Scale from room sizes: {ppm_area:.1f} px/m "
              f"(median room set to {TYPICAL_ROOM_AREA_M2:.0f} m^2)")
        return ppm_area
    print(f"[WALK] No scale anchors; using default {default:.0f} px/m")
    return default


# ================= SMALL GEOMETRY HELPERS =================
def px_to_m_real(p, px_per_m):
    """Plan pixels -> real meters (y flipped so the plan isn't mirrored)."""
    return np.array([p[0], -p[1]], dtype=float) / float(px_per_m)


def bake_lighting(mesh):
    """Bake soft multi-light shading into per-face colours.

    Splits shared vertices so each triangle is flat-shaded by its own normal,
    then tints by (ambient + key + fill + sky) directional terms. Produces a
    clean, evenly lit look with no realtime specular. Meshes with intentional
    colour gradients (window glass, sky, light spill) or near-white emissive
    surfaces are returned untouched so their glow survives.
    """
    if mesh.has_triangle_uvs() and len(mesh.textures):
        return mesh
    tris = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)
    cols = np.asarray(mesh.vertex_colors)
    if len(tris) == 0 or len(cols) != len(verts):
        return mesh
    if np.ptp(cols, axis=0).max() > 0.03:      # keep gradients (glass/spill)
        return mesh
    base = cols[0].astype(float)
    if base.min() > 0.9:                        # keep bright emissive (sky/glass)
        return mesh

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    shade = (_AMBIENT
             + _KEY_I * np.clip(n @ _KEY_DIR, 0, None)
             + _FILL_I * np.clip(n @ _FILL_DIR, 0, None)
             + _SKY_I * np.clip(n[:, 2], 0, None))
    tri_col = np.clip(base[None, :] * shade[:, None], 0, 1)

    new_v = verts[tris].reshape(-1, 3)
    new_c = np.repeat(tri_col, 3, axis=0)
    new_t = np.arange(len(new_v), dtype=np.int32).reshape(-1, 3)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(new_v)
    out.triangles = o3d.utility.Vector3iVector(new_t)
    out.vertex_colors = o3d.utility.Vector3dVector(new_c)
    out.compute_vertex_normals()
    return out


def _paint(m, color):
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def _bx(w, d, h, color, cx=0.0, cy=0.0, z=0.0):
    """Box centered at (cx, cy) in the XY footprint, sitting on z."""
    m = o3d.geometry.TriangleMesh.create_box(width=w, height=d, depth=h)
    m.translate((cx - w / 2, cy - d / 2, z))
    return _paint(m, color)


def _cyl(r, h, color, cx=0.0, cy=0.0, z=0.0, res=24):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h, resolution=res)
    m.translate((cx, cy, z + h / 2))
    return _paint(m, color)


def _cylinder_between(start, end, radius, color, resolution=24):
    """Create a cylinder aligned between arbitrary 3D endpoints."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-5:
        return None
    direction = vector / length
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution
    )
    axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    axis_length = float(np.linalg.norm(axis))
    if axis_length > 1e-8:
        axis /= axis_length
        angle = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
        mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle),
            center=(0.0, 0.0, 0.0),
        )
    elif direction[2] < 0:
        mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz((math.pi, 0.0, 0.0)),
            center=(0.0, 0.0, 0.0),
        )
    mesh.translate((start + end) / 2)
    return _paint(mesh, color)


def _orient_horizontal_surface(mesh, upward=True):
    """Give horizontal PBR surfaces a predictable front face and tangent basis."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        return mesh
    points = vertices[triangles]
    signed_z = float(np.sum(np.cross(
        points[:, 1] - points[:, 0],
        points[:, 2] - points[:, 0],
    )[:, 2]))
    should_flip = (upward and signed_z < 0) or (not upward and signed_z > 0)
    if should_flip:
        mesh.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(triangles[:, [0, 2, 1]])
        )
        if mesh.has_triangle_uvs():
            uvs = np.asarray(mesh.triangle_uvs).reshape((-1, 3, 2))
            mesh.triangle_uvs = o3d.utility.Vector2dVector(
                np.ascontiguousarray(uvs[:, [0, 2, 1], :].reshape((-1, 2)))
            )
    mesh.compute_vertex_normals()
    return mesh


def _sph(r, color, cx=0.0, cy=0.0, z=0.0):
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=14)
    m.translate((cx, cy, z))
    return _paint(m, color)


def _rounded_cuboid(w, d, h, color, cx=0.0, cy=0.0, z=0.0,
                    roundness=0.24, resolution=22):
    """Smooth superellipsoid component used for upholstered furnishings."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0, resolution=resolution
    )
    vertices = np.asarray(mesh.vertices)
    shaped = np.sign(vertices) * np.power(
        np.abs(vertices), roundness
    )
    shaped *= np.array([w / 2, d / 2, h / 2])
    shaped += np.array([cx, cy, z + h / 2])
    mesh.vertices = o3d.utility.Vector3dVector(shaped)
    return _paint(mesh, color)


def _shade(base, k):
    """Lighten/darken a color."""
    return [min(1.0, c * k) for c in base]


def yaw_facing(n):
    """Yaw so that the furniture's local +Y axis points along normal n."""
    return math.atan2(-n[0], n[1])


def _rotz(yaw):
    return o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))


def place_meshes(meshes, pos, yaw):
    """Rotate around Z then translate a list of local-frame meshes."""
    R = _rotz(yaw)
    for m in meshes:
        m.rotate(R, center=(0, 0, 0))
        m.translate((pos[0], pos[1], 0.0))
    return meshes


def rotate_furniture_object(furniture, delta):
    """Rotate one editable component group around its stable floor pivot."""
    center = (
        float(furniture["position"][0]),
        float(furniture["position"][1]),
        0.0,
    )
    rotation = _rotz(delta)
    for mesh in furniture["meshes"]:
        mesh.rotate(rotation, center=center)
    furniture["yaw"] += delta
    return furniture


def footprint_poly(pos, yaw, w, d):
    fp = shp_box(-w / 2, -d / 2, w / 2, d / 2)
    fp = affinity.rotate(fp, yaw, origin=(0, 0), use_radians=True)
    return affinity.translate(fp, pos[0], pos[1])


# ================= FURNITURE BUILDERS (local frame) =================
# Every builder returns (meshes, w, d). Local frame: footprint centered on
# the origin, w along X, d along Y, furniture FACES +Y, floor at z=0.

def build_sofa(P, w=2.2, d=0.95):
    fab = P["sofa"]
    lift = 0.12                                                    # raised on feet
    ms = [
        _bx(w - 0.04, d - 0.04, 0.34, fab, z=lift),                # seat base
        _bx(w, 0.20, 0.78, _shade(fab, 0.92),
            cy=-(d / 2 - 0.10), z=lift + 0.06),                    # backrest
        _bx(0.20, d, 0.56, _shade(fab, 0.97), cx=-(w / 2 - 0.10),
            z=lift),                                               # arms
        _bx(0.20, d, 0.56, _shade(fab, 0.97), cx=(w / 2 - 0.10), z=lift),
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):                                         # wooden feet
            ms.append(_cyl(0.03, lift, P["wood_dark"],
                           cx=sx * (w / 2 - 0.10), cy=sy * (d / 2 - 0.10)))
    cw = (w - 0.52) / 2
    for sx in (-1, 1):
        ms.append(_bx(cw - 0.04, d - 0.42, 0.15, _shade(fab, 1.07),
                      cx=sx * (cw / 2 + 0.02), cy=0.08, z=lift + 0.34))  # seat cushions
        ms.append(_bx(cw - 0.06, 0.16, 0.42, _shade(fab, 1.03),
                      cx=sx * (cw / 2 + 0.02), cy=-(d / 2 - 0.19),
                      z=lift + 0.40))                              # back cushions
        ms.append(_bx(0.42, 0.15, 0.38, P["cushion"],
                      cx=sx * (cw / 2), cy=-(d / 2 - 0.34),
                      z=lift + 0.44))                              # throw pillows
    return ms, w, d


def build_armchair(P, w=0.92, d=0.85):
    fab = P["sofa"]
    lift = 0.12
    ms = [
        _bx(w - 0.04, d - 0.04, 0.34, fab, z=lift),
        _bx(w, 0.18, 0.74, _shade(fab, 0.92), cy=-(d / 2 - 0.09), z=lift + 0.06),
        _bx(0.16, d, 0.52, _shade(fab, 0.97), cx=-(w / 2 - 0.08), z=lift),
        _bx(0.16, d, 0.52, _shade(fab, 0.97), cx=(w / 2 - 0.08), z=lift),
        _bx(w - 0.36, d - 0.34, 0.15, _shade(fab, 1.07), cy=0.06, z=lift + 0.34),
        _bx(0.38, 0.14, 0.34, P["cushion"], cy=-(d / 2 - 0.26), z=lift + 0.42),
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_cyl(0.028, lift, P["wood_dark"],
                           cx=sx * (w / 2 - 0.09), cy=sy * (d / 2 - 0.09)))
    return ms, w, d


def build_coffee_table(P, w=1.1, d=0.6):
    ms = [_bx(w, d, 0.06, P["table"], z=0.36)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_bx(0.05, 0.05, 0.36, P["wood_dark"],
                          cx=sx * (w / 2 - 0.06), cy=sy * (d / 2 - 0.06)))
    ms.append(_bx(0.3, 0.22, 0.03, P["accent"], cx=-w * 0.18, z=0.42))  # books
    return ms, w, d


def build_side_table(P, diameter=0.46):
    """Sculptural side table with a stone top and fluted timber pedestal."""
    top = _mix_color(P["table"], WHITE_SOFT, 0.22)
    stem = _mix_color(P["wood"], P["wood_dark"], 0.28)
    meshes = [
        _cyl(diameter * 0.48, 0.055, top, z=0.47, res=40),
        _cyl(diameter * 0.23, 0.45, stem, z=0.03, res=32),
        _cyl(
            diameter * 0.34,
            0.035,
            _mix_color(P["wood_dark"], P["metal"], 0.10),
            z=0.0,
            res=32,
        ),
    ]
    apply_archviz_material(
        meshes[0],
        "marble",
        tint=top,
        tint_strength=0.14,
        repeat_m=0.7,
    )
    apply_archviz_material(
        meshes[1],
        "warm_oak",
        tint=stem,
        tint_strength=0.28,
        repeat_m=0.55,
    )
    # A restrained hand-thrown ceramic object avoids repeating the angular
    # vase asset at every styling point.
    ceramic = _mix_color(P["accent"], P["wall"], 0.34)
    meshes.extend([
        _cyl(0.052, 0.13, ceramic, z=0.53, res=32),
        _sph(0.065, ceramic, z=0.62),
        _cyl(0.027, 0.055, _shade(ceramic, 0.88), z=0.675, res=24),
    ])
    return meshes, diameter, diameter


def build_rug(P, w=2.6, d=1.8, design="plain woven"):
    """Build one low-profile rounded rug without overlapping slab layers."""
    design = str(design or "plain woven").lower()
    radius = min(0.14, w * 0.08, d * 0.12)
    rounded = shp_box(
        -w / 2 + radius,
        -d / 2 + radius,
        w / 2 - radius,
        d / 2 - radius,
    ).buffer(radius, resolution=6)
    base = _orient_horizontal_surface(
        floor_mesh(list(rounded.exterior.coords)[:-1], P["rug"]),
        upward=True,
    )
    base.translate((0.0, 0.0, 0.014))
    apply_archviz_material(
        base, "carpet", tint=P["rug"], tint_strength=0.48, repeat_m=0.85
    )
    meshes = [base]

    def rug_layer(width, depth, color, cx=0.0, cy=0.0, z=0.015):
        layer = _bx(
            max(0.025, width), max(0.025, depth), 0.003,
            color, cx=cx, cy=cy, z=z,
        )
        apply_archviz_material(
            layer, "carpet", tint=color, tint_strength=0.40, repeat_m=0.72
        )
        meshes.append(layer)

    if design == "bordered":
        border = 0.065
        border_color = _mix_color(P["rug"], P["accent"], 0.28)
        rug_layer(w - 0.18, border, border_color, cy=d / 2 - 0.10)
        rug_layer(w - 0.18, border, border_color, cy=-d / 2 + 0.10)
        rug_layer(border, d - 0.31, border_color, cx=w / 2 - 0.10)
        rug_layer(border, d - 0.31, border_color, cx=-w / 2 + 0.10)
    elif design == "geometric":
        accent = _mix_color(P["rug"], P["accent"], 0.58)
        stripe = max(0.055, min(0.12, w * 0.045))
        for fraction in (-0.28, 0.0, 0.28):
            rug_layer(stripe, d - 0.12, accent, cx=w * fraction)
        rug_layer(
            w - 0.14, stripe, _shade(accent, 0.88), cy=d * 0.24, z=0.019
        )
        rug_layer(
            w - 0.14, stripe, _shade(accent, 0.88), cy=-d * 0.24, z=0.019
        )
    elif design == "vintage pattern":
        border = _mix_color(P["rug"], P["accent"], 0.64)
        line = 0.055
        rug_layer(w - 0.18, line, border, cy=d / 2 - 0.10)
        rug_layer(w - 0.18, line, border, cy=-d / 2 + 0.10)
        rug_layer(line, d - 0.31, border, cx=w / 2 - 0.10)
        rug_layer(line, d - 0.31, border, cx=-w / 2 + 0.10)
        rug_layer(w * 0.38, d * 0.12, border, z=0.019)
        rug_layer(w * 0.12, d * 0.38, border, z=0.019)
    else:
        band = _shade(P["rug"], 0.90)
        rug_layer(w - 0.10, 0.035, band, cy=d / 2 - 0.07)
        rug_layer(w - 0.10, 0.035, band, cy=-d / 2 + 0.07)
    return meshes, w, d


def build_tv_unit(P, w=1.7, d=0.42):
    ms = [
        _bx(w, d, 0.5, P["wood"]),                                  # console
        _bx(w, d, 0.02, _shade(P["wood"], 1.1), z=0.5),
        _bx(1.35, 0.05, 0.75, SCREEN_DARK, cy=-(d / 2 - 0.05), z=0.85),  # TV
        _bx(0.25, 0.18, 0.22, P["accent"], cx=w * 0.3, z=0.52),     # decor
    ]
    return ms, w, d


def build_bed(P, w=1.8, d=2.15):
    """Detailed contemporary upholstered bed built from smooth 3D components."""
    fabric = _mix_color(P["sofa"], P["wall"], 0.18)
    linen = _mix_color(WHITE_SOFT, P["cushion"], 0.18)
    ms = [
        _rounded_cuboid(
            w - 0.04, d - 0.12, 0.26, P["wood_dark"],
            cy=0.04, z=0.10, roundness=0.18,
        ),
        _rounded_cuboid(
            w, 0.18, 1.02, fabric,
            cy=-(d / 2 - 0.09), z=0.12, roundness=0.20,
        ),
        _rounded_cuboid(
            w - 0.12, d - 0.28, 0.28, linen,
            cy=0.04, z=0.34, roundness=0.20,
        ),
        _rounded_cuboid(
            w - 0.08, d * 0.59, 0.16, P["cushion"],
            cy=d * 0.17, z=0.57, roundness=0.26,
        ),
    ]
    for sx in (-1, 1):
        ms.append(_rounded_cuboid(
            0.66, 0.40, 0.20, linen,
            cx=sx * (w / 4 - 0.02),
            cy=-(d / 2 - 0.37),
            z=0.59,
            roundness=0.30,
        ))
    ms.append(_rounded_cuboid(
        0.72, 0.22, 0.18, _shade(P["cushion"], 0.88),
        cy=-(d / 2 - 0.52), z=0.70, roundness=0.32,
    ))
    apply_archviz_material(
        ms[0],
        "dark_wood",
        tint=P["wood_dark"],
        tint_strength=0.20,
        repeat_m=1.35,
    )
    for index, tint in (
        (1, fabric),
        (2, linen),
        (3, P["cushion"]),
        (4, linen),
        (5, linen),
        (6, _shade(P["cushion"], 0.88)),
    ):
        apply_archviz_material(
            ms[index],
            "curtain_fabric",
            tint=tint,
            tint_strength=0.68,
            repeat_m=0.46,
        )
    # Tufted headboard buttons and slim feet make this read as an authored
    # furniture component rather than a set of primitive boxes.
    for x in np.linspace(-w * 0.35, w * 0.35, 5):
        for button_z in (0.48, 0.76, 1.00):
            ms.append(_sph(
                0.026, _shade(fabric, 0.68),
                cx=float(x), cy=-(d / 2 - 0.19), z=button_z,
            ))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_cyl(
                0.035, 0.11, P["metal"],
                cx=sx * (w / 2 - 0.10),
                cy=sy * (d / 2 - 0.13),
            ))
    return ms, w, d


def build_nightstand(P, w=0.48, d=0.42):
    ms = [
        _bx(w, d, 0.50, P["wood"]),
        _bx(w - 0.06, 0.02, 0.12, _shade(P["wood"], 0.8), cy=-(0.0), z=0.30),
        _cyl(0.03, 0.30, P["metal"], z=0.50),
        _cyl(0.13, 0.17, P["shade"], z=0.80),                       # lamp shade
    ]
    apply_archviz_material(
        ms[0], "warm_oak", tint=P["wood"], tint_strength=0.24, repeat_m=0.72
    )
    return ms, w, d


def build_wardrobe(P, w=1.8, d=0.62):
    ms = [
        _bx(w, d, 2.2, P["cabinet"]),
        _bx(0.02, 0.03, 2.0, P["wood_dark"], cy=(d / 2 - 0.005), z=0.1),  # door seam
        _cyl(0.015, 0.25, P["metal"], cx=-0.10, cy=(d / 2 + 0.01), z=1.0),
        _cyl(0.015, 0.25, P["metal"], cx=0.10, cy=(d / 2 + 0.01), z=1.0),
    ]
    return ms, w, d


def build_dining_table(P, w=1.7, d=0.95):
    ms = [_rounded_cuboid(
        w, d, 0.07, P["wood"], z=0.70, roundness=0.12
    )]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_bx(0.07, 0.07, 0.72, P["wood_dark"],
                          cx=sx * (w / 2 - 0.09), cy=sy * (d / 2 - 0.09)))
    apply_archviz_material(
        ms[0], "warm_oak", tint=P["wood"], tint_strength=0.22, repeat_m=1.25
    )
    ms.extend([
        _cyl(0.14, 0.045, _mix_color(P["table"], P["metal"], 0.18), z=0.78),
        _cyl(0.048, 0.17, P["accent"], z=0.825, res=28),
    ])
    for endpoint, radius, color in (
        ((-0.09, 0.02, 1.13), 0.052, GREEN_FOLIAGE),
        ((0.08, -0.02, 1.08), 0.048, _shade(GREEN_FOLIAGE, 1.16)),
        ((0.02, 0.06, 1.18), 0.045, _shade(GREEN_FOLIAGE, 0.92)),
    ):
        stem = _cylinder_between(
            (0.0, 0.0, 0.98),
            endpoint,
            0.006,
            _shade(GREEN_FOLIAGE, 0.62),
            resolution=12,
        )
        if stem is not None:
            ms.append(stem)
        ms.append(_sph(
            radius,
            color,
            cx=endpoint[0],
            cy=endpoint[1],
            z=endpoint[2],
        ))
    return ms, w, d


def build_chair(P, w=0.46, d=0.48):
    ms = [
        _bx(w, d, 0.05, P["wood"], z=0.43),                         # seat
        _bx(w, 0.05, 0.45, P["wood"], cy=-(d / 2 - 0.03), z=0.48),  # back
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_bx(0.04, 0.04, 0.43, P["wood_dark"],
                          cx=sx * (w / 2 - 0.05), cy=sy * (d / 2 - 0.05)))
    return ms, w, d


def build_sideboard(P, w=1.6, d=0.45):
    meshes = [
        _bx(w, d, 0.8, P["wood"]),
        _bx(w, d, 0.02, _shade(P["wood"], 1.12), z=0.80),
        _bx(0.30, 0.20, 0.35, P["accent"], cx=-w * 0.25, z=0.82),
        _sph(0.16, GREEN_FOLIAGE, cx=w * 0.28, z=1.05),
        _cyl(0.09, 0.22, TERRACOTTA, cx=w * 0.28, z=0.82),
    ]
    apply_archviz_material(
        meshes[0], "warm_oak", tint=P["wood"], tint_strength=0.22, repeat_m=1.2
    )
    return meshes, w, d


def build_kitchen_run(P, w=3.0, d=0.64):
    ms = [
        _bx(w, d - 0.04, 0.88, P["cabinet"], cy=-0.02),             # base cabinets
        _bx(w, d, 0.04, P["counter"], z=0.88),                      # countertop
        _bx(w, 0.035, 0.52, _mix_color(P["counter"], P["wall"], 0.48),
            cy=-(d / 2 + 0.018), z=0.93),                           # backsplash
        _bx(w * 0.72, 0.35, 0.72, P["cabinet"],
            cx=-w * 0.12, cy=-(d / 2 - 0.175), z=1.50),             # upper cabinets
        _bx(0.60, 0.52, 0.015, SCREEN_DARK, cx=w * 0.22, z=0.92),   # cooktop
        _bx(0.50, 0.40, 0.02, [0.75, 0.77, 0.79], cx=-w * 0.25, z=0.90),  # sink
        _cyl(0.018, 0.30, P["metal"], cx=-w * 0.25,
             cy=-(d / 2 - 0.08), z=0.92),                           # faucet
        _cyl(0.10, 0.17, P["accent"], cx=w * 0.39, z=0.93),         # utensil pot
    ]
    apply_archviz_material(ms[1], "marble", tint=P["counter"], tint_strength=0.10)
    apply_archviz_material(
        ms[2], "bathroom_tile", tint=P["wall"], tint_strength=0.18
    )
    # Shallow individual door fronts and handles make the cabinetry read as a
    # fitted kitchen system rather than one uninterrupted box.
    n_doors = max(2, int(w / 0.6))
    base_panels = []
    base_panel_w = w / n_doors - 0.022
    for index in range(n_doors):
        x = -w / 2 + (index + 0.5) * (w / n_doors)
        base_panels.append(_bx(
            base_panel_w,
            0.025,
            0.76,
            _shade(P["cabinet"], 0.98),
            cx=x,
            cy=d / 2 - 0.008,
            z=0.06,
        ))
        ms.append(_bx(
            0.12, 0.035, 0.025, P["metal"],
            cx=x, cy=d / 2 + 0.012, z=0.48,
        ))
    base_fronts = base_panels[0]
    for panel in base_panels[1:]:
        base_fronts += panel
    ms.append(base_fronts)

    upper_w = w * 0.72
    n_upper = max(2, int(upper_w / 0.55))
    upper_left = -w * 0.12 - upper_w / 2
    upper_panels = []
    for index in range(n_upper):
        x = upper_left + (index + 0.5) * (upper_w / n_upper)
        upper_panels.append(_bx(
            upper_w / n_upper - 0.022,
            0.025,
            0.66,
            _shade(P["cabinet"], 0.98),
            cx=x,
            cy=0.042,
            z=1.53,
        ))
        ms.append(_bx(
            0.12, 0.035, 0.025, P["metal"],
            cx=x, cy=0.062, z=1.56,
        ))
    upper_fronts = upper_panels[0]
    for panel in upper_panels[1:]:
        upper_fronts += panel
    ms.append(upper_fronts)
    return ms, w, d


def build_fridge(P, w=0.72, d=0.70):
    return [
        _bx(w, d, 1.92, [0.76, 0.77, 0.79]),
        _bx(0.03, 0.04, 0.9, [0.4, 0.4, 0.42], cx=-(w / 2 - 0.10),
            cy=(d / 2 - 0.01), z=0.95),
    ], w, d


def build_island(P, w=1.7, d=0.9):
    ms = [
        _bx(w, d, 0.88, P["cabinet"]),
        _bx(w + 0.08, d + 0.08, 0.05, P["counter"], z=0.88),
        _cyl(
            0.16, 0.045, _mix_color(P["accent"], P["table"], 0.38),
            cx=-w * 0.18, z=0.95, res=32,
        ),
        _sph(0.055, [0.72, 0.28, 0.16], cx=-w * 0.23, z=1.03),
        _sph(0.052, [0.84, 0.60, 0.20], cx=-w * 0.14, cy=0.03, z=1.02),
        _sph(0.048, [0.48, 0.60, 0.26], cx=-w * 0.11, cy=-0.04, z=1.02),
    ]
    apply_archviz_material(
        ms[0],
        "warm_oak",
        tint=P["cabinet"],
        tint_strength=0.46,
        repeat_m=1.25,
        detail_maps=False,
    )
    apply_archviz_material(ms[1], "marble", tint=P["counter"], tint_strength=0.10)
    for sx in (-1, 1):
        ms.append(_cyl(0.17, 0.03, P["wood"], cx=sx * w * 0.22,
                       cy=(d / 2 + 0.32), z=0.62))                  # stool seats
        ms.append(_cyl(0.03, 0.62, P["metal"], cx=sx * w * 0.22,
                       cy=(d / 2 + 0.32), z=0.0))
    return ms, w, d + 0.65   # depth includes the stools


def build_desk(P, w=1.4, d=0.7):
    ms = [_bx(w, d, 0.04, P["wood"], z=0.72)]
    for sx in (-1, 1):
        ms.append(_bx(0.05, d - 0.08, 0.72, P["wood_dark"], cx=sx * (w / 2 - 0.05)))
    ms.append(_bx(0.55, 0.04, 0.34, SCREEN_DARK, cy=-(d / 2 - 0.12), z=0.86))  # monitor
    ms.append(_cyl(0.05, 0.10, SCREEN_DARK, cy=-(d / 2 - 0.12), z=0.76))
    ms.append(_bx(0.34, 0.13, 0.02, [0.35, 0.35, 0.37], cy=0.10, z=0.76))      # keyboard
    return ms, w, d


def build_office_chair(P, w=0.55, d=0.55):
    return [
        _cyl(0.26, 0.04, P["metal"], z=0.02),
        _cyl(0.035, 0.40, P["metal"], z=0.06),
        _bx(0.48, 0.46, 0.09, P["cushion"], z=0.46),
        _bx(0.46, 0.09, 0.55, P["cushion"], cy=-(d / 2 - 0.10), z=0.52),
    ], w, d


def build_bookshelf(P, w=1.6, d=0.34):
    ms = [_bx(w, d, 2.0, P["wood"])]
    book_cols = [P["accent"], P["cushion"], P["metal"], _shade(P["accent"], 0.7)]
    for i, zz in enumerate((0.45, 0.95, 1.45)):
        ms.append(_bx(w - 0.08, d - 0.03, 0.035, _shade(P["wood"], 1.15),
                      cy=0.015, z=zz))
        for b in range(4):
            ms.append(_bx(0.16, 0.05, 0.30, book_cols[(i + b) % len(book_cols)],
                          cx=-w / 2 + 0.25 + b * 0.34,
                          cy=(d / 2 - 0.02), z=zz + 0.04))
    return ms, w, d


def build_vanity(P, w=1.0, d=0.52):
    return [
        _bx(w, d, 0.82, P["cabinet"]),
        _bx(w, d + 0.03, 0.04, P["counter"], z=0.82),
        _cyl(0.17, 0.10, WHITE_SOFT, z=0.86),
        _cyl(0.015, 0.20, P["metal"], cy=-(d / 2 - 0.05), z=0.92),
        _bx(w - 0.2, 0.03, 0.9, [0.72, 0.82, 0.88], cy=-(d / 2 - 0.02), z=1.15),  # mirror
    ], w, d


def build_toilet(P, w=0.42, d=0.66):
    ceramic = _mix_color(WHITE_SOFT, P["wall"], 0.05)
    tank_y = -(d / 2 - 0.09)
    bowl_y = 0.08
    meshes = [
        _rounded_cuboid(
            w, 0.18, 0.68, ceramic,
            cy=tank_y, z=0.10, roundness=0.13,
        ),
        _rounded_cuboid(
            0.28, 0.38, 0.38, ceramic,
            cy=0.02, z=0.0, roundness=0.28,
        ),
        _rounded_cuboid(
            0.36, 0.46, 0.10, ceramic,
            cy=bowl_y, z=0.34, roundness=0.38,
        ),
        _rounded_cuboid(
            0.34, 0.43, 0.035, _shade(ceramic, 0.97),
            cy=bowl_y, z=0.44, roundness=0.40,
        ),
        _cyl(
            0.035, 0.018, P["metal"],
            cy=tank_y, z=0.80, res=28,
        ),
    ]
    seat = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=0.145,
        tube_radius=0.024,
        radial_resolution=40,
        tubular_resolution=18,
    )
    vertices = np.asarray(seat.vertices)
    vertices[:, 1] *= 1.28
    seat.vertices = o3d.utility.Vector3dVector(vertices)
    seat.translate((0.0, bowl_y, 0.47))
    meshes.append(_paint(seat, _shade(ceramic, 0.92)))
    for mesh in meshes[:4] + meshes[5:]:
        apply_archviz_material(
            mesh,
            "plaster",
            tint=ceramic,
            tint_strength=0.08,
            detail_maps=False,
        )
    return meshes, w, d


def build_shower(P, w=0.95, d=0.95):
    glass = [0.78, 0.88, 0.92]
    return [
        _bx(w, d, 0.06, WHITE_SOFT),                                # tray
        _bx(w, 0.025, 1.95, glass, cy=(d / 2 - 0.0125), z=0.06),    # front glass
        _bx(0.025, d, 1.95, glass, cx=(w / 2 - 0.0125), z=0.06),    # side glass
        _cyl(0.02, 1.1, P["metal"], cx=-(w / 2 - 0.12), cy=-(d / 2 - 0.12), z=0.9),
        _sph(0.07, P["metal"], cx=-(w / 2 - 0.12), cy=-(d / 2 - 0.12), z=2.05),
    ], w, d


def build_bathtub(P, w=1.65, d=0.78):
    """Freestanding bath fallback with a real hollow rim and metal fittings."""
    shell = _mix_color(WHITE_SOFT, P["wall"], 0.10)
    inner = _shade(shell, 0.92)
    rim = 0.09
    ms = [
        _bx(w - 0.12, d - 0.12, 0.16, inner, z=0.10),
        _bx(w, rim, 0.48, shell, cy=-(d / 2 - rim / 2), z=0.10),
        _bx(w, rim, 0.48, shell, cy=(d / 2 - rim / 2), z=0.10),
        _bx(rim, d - rim * 2, 0.48, shell,
            cx=-(w / 2 - rim / 2), z=0.10),
        _bx(rim, d - rim * 2, 0.48, shell,
            cx=(w / 2 - rim / 2), z=0.10),
        _cyl(0.018, 0.34, P["metal"], cx=w * 0.32,
             cy=-(d / 2 - 0.02), z=0.50),
        _cyl(0.045, 0.055, P["metal"], cx=w * 0.32,
             cy=-(d / 2 - 0.02), z=0.80),
    ]
    return ms, w, d


def _professional_detail(asset_key, P, w, d, h, z=0.0):
    """Load a local production-authored detail, retaining material colors."""
    try:
        from furniture_catalog import load_catalog_asset

        meshes = load_catalog_asset(
            asset_key,
            "Modern",
            w,
            d,
            height=h,
            palette=P,
        )
        if not meshes:
            return None
        if z:
            for mesh in meshes:
                mesh.translate((0.0, 0.0, z))
        return meshes
    except Exception as exc:
        print(f"[WALK] Professional detail '{asset_key}' unavailable: {exc}")
        return None


def _textured_art_canvas(w, h, z0):
    """Map the full local artwork texture onto a real inset canvas."""
    try:
        path = os.path.join(
            os.path.dirname(__file__),
            "assets",
            "furniture_catalog",
            "pro",
            "custom_wall_art",
            "abstract_earth_sage.png",
        )
        image = o3d.io.read_image(path)
        if np.asarray(image).size == 0:
            return None
        canvas_w, canvas_h = w * 0.86, h * 0.82
        x0, x1 = -canvas_w / 2, canvas_w / 2
        z1, z2 = z0 + h * 0.09, z0 + h * 0.09 + canvas_h
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray([
                (x0, 0.056, z1),
                (x1, 0.056, z1),
                (x1, 0.056, z2),
                (x0, 0.056, z2),
            ], dtype=float)),
            o3d.utility.Vector3iVector(np.asarray([
                (0, 2, 1),
                (0, 3, 2),
            ], dtype=np.int32)),
        )
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            np.asarray([
                (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
                (0.0, 1.0), (0.0, 0.0), (1.0, 0.0),
            ], dtype=float)
        )
        mesh.triangle_material_ids = o3d.utility.IntVector(
            np.zeros(2, dtype=np.int32)
        )
        mesh.textures = [image]
        mesh.paint_uniform_color([1.0, 1.0, 1.0])
        mesh.compute_vertex_normals()
        return mesh
    except Exception as exc:
        print(f"[WALK] Generated wall-art canvas unavailable: {exc}")
        return None


def build_plant(P, tall=True):
    h = 1.35 if tall else 0.8
    professional = _professional_detail("plant", P, 0.55, 0.55, h)
    if professional:
        return professional, 0.55, 0.55
    ms = [
        _cyl(0.16, 0.34, TERRACOTTA),
        _cyl(0.028, h - 0.5, P["wood_dark"], z=0.34),
        _sph(0.30 if tall else 0.24, GREEN_FOLIAGE, z=h),
        _sph(0.22 if tall else 0.16, _shade(GREEN_FOLIAGE, 1.25),
             cx=0.14, cy=0.06, z=h - 0.22),
        _sph(0.18, _shade(GREEN_FOLIAGE, 0.85), cx=-0.15, cy=-0.05, z=h - 0.12),
    ]
    return ms, 0.55, 0.55


def build_floor_lamp(P):
    """Slim arched reading lamp with a coordinated dark-bronze frame."""
    frame = _mix_color(P["metal"], [0.14, 0.13, 0.12], 0.68)
    meshes = [_cyl(0.18, 0.025, frame)]
    points = (
        (0.0, 0.0, 0.025),
        (0.0, 0.0, 1.28),
        (0.0, 0.08, 1.53),
        (0.0, 0.25, 1.66),
        (0.0, 0.40, 1.68),
    )
    for start, end in zip(points, points[1:]):
        section = _cylinder_between(start, end, 0.014, frame, resolution=18)
        if section is not None:
            meshes.append(section)
    shade = _mix_color(P["shade"], WHITE_SOFT, 0.35)
    meshes.extend([
        _cyl(0.21, 0.23, shade, cy=0.40, z=1.44, res=36),
        _sph(0.055, [1.0, 0.84, 0.57], cy=0.40, z=1.50),
    ])
    return meshes, 0.5, 0.65


def build_art(P, w=1.25, h=0.85, z0=1.25):
    """Textured framed wall art with a modeled procedural fallback."""
    professional = _professional_detail("wall_art", P, w, 0.10, h, z=z0)
    if professional:
        canvas = _textured_art_canvas(w, h, z0)
        if canvas is not None:
            professional.append(canvas)
        return professional, w, 0.10
    return [
        _bx(w, 0.05, h, P["wood_dark"], z=z0),
        _bx(w - 0.08, 0.052, h - 0.08, _shade(P["wall"], 1.03), z=z0 + 0.04),
        _bx(w * 0.45, 0.056, h * 0.45, P["accent"], cx=-w * 0.12, z=z0 + h * 0.25),
        _bx(w * 0.3, 0.056, h * 0.3, P["cushion"], cx=w * 0.22, z=z0 + h * 0.18),
    ], w, 0.08


def build_slat_feature(P, w=1.8, h=1.35, z0=0.72):
    """Architectural timber feature used behind key furniture anchors."""
    ms = []
    count = max(5, int(w / 0.14))
    gap = w / count
    for i in range(count):
        x = -w / 2 + gap * (i + 0.5)
        ms.append(_bx(gap * 0.56, 0.045, h, P["wood_dark"], cx=x, z=z0))
    ms.append(_bx(w + 0.08, 0.025, 0.055, P["metal"], z=z0 + h + 0.05))
    return ms, w, 0.07


def build_console_table(P, w=1.15, d=0.34):
    """Slim styled console for circulation zones and spare walls."""
    ms = [_bx(w, d, 0.055, P["wood"], z=0.74)]
    for sx in (-1, 1):
        ms.append(_bx(0.045, d - 0.04, 0.74, P["wood_dark"],
                      cx=sx * (w / 2 - 0.07)))
    vase = _professional_detail("decor_vase", P, 0.20, 0.20, 0.30, z=0.79)
    if vase:
        for mesh in vase:
            mesh.translate((-w * 0.24, 0.0, 0.0))
        ms.extend(vase)
    else:
        ms.append(_cyl(0.085, 0.20, P["accent"], cx=-w * 0.23, z=0.80))
    # A restrained pair of books replaces the former toy-like topiary sphere.
    ms.extend([
        _bx(0.28, 0.16, 0.025, _shade(P["accent"], 0.82),
            cx=w * 0.22, z=0.80),
        _bx(0.24, 0.14, 0.025, _shade(P["wall"], 0.82),
            cx=w * 0.22, z=0.825),
    ])
    return ms, w, d


def build_ottoman(P, w=0.72, d=0.52):
    ms = [
        _bx(w, d, 0.28, P["sofa"], z=0.11),
        _bx(w - 0.08, d - 0.08, 0.10, _shade(P["sofa"], 1.08), z=0.39),
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_cyl(0.025, 0.11, P["wood_dark"],
                           cx=sx * (w / 2 - 0.08), cy=sy * (d / 2 - 0.08)))
    return ms, w, d


def build_bench(P, w=1.15, d=0.42):
    """Upholstered bedroom/dressing bench with a slim timber frame."""
    ms = [
        _bx(w, d, 0.16, P["sofa"], z=0.38),
        _bx(w - 0.08, d - 0.08, 0.08, _shade(P["sofa"], 1.08), z=0.54),
    ]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ms.append(_bx(
                0.045, 0.045, 0.38, P["wood_dark"],
                cx=sx * (w / 2 - 0.08), cy=sy * (d / 2 - 0.07),
            ))
    return ms, w, d


def build_round_mirror(P, diameter=0.88, z=1.55, ornate=False):
    """Build a clean round mirror, using the ornate asset only for classics."""
    professional = (
        _professional_detail(
            "wall_mirror", P, diameter, 0.10, diameter, z=z
        )
        if ornate
        else None
    )
    if professional:
        return professional, diameter, 0.10
    frame = o3d.geometry.TriangleMesh.create_cylinder(
        radius=diameter / 2, height=0.035, resolution=40
    )
    frame.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((math.pi / 2, 0, 0)),
        center=(0, 0, 0),
    )
    frame.translate((0, 0, z))
    _paint(frame, P["metal"])
    glass = o3d.geometry.TriangleMesh.create_cylinder(
        radius=diameter / 2 - 0.045, height=0.038, resolution=40
    )
    glass.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((math.pi / 2, 0, 0)),
        center=(0, 0, 0),
    )
    glass.translate((0, 0.012, z))
    _paint(glass, [0.68, 0.78, 0.82])
    return [frame, glass], diameter, 0.07


def build_wall_clock(P, diameter=0.56, z=1.58):
    """Production-authored wall clock with readable modeled depth."""
    professional = _professional_detail(
        "wall_clock", P, diameter, 0.12, diameter, z=z
    )
    if professional:
        return professional, diameter, 0.12
    return build_round_mirror(P, diameter=diameter, z=z)


def build_wall_sconce(P, w=0.26, d=0.20, h=0.46, z=1.30):
    """Modeled wall sconce; fallback has a projecting shade and stem."""
    professional = _professional_detail(
        "wall_sconce", P, w, d, h, z=z
    )
    if professional:
        return professional, w, d
    return [
        _cyl(0.10, 0.035, P["metal"], z=z + h * 0.42),
        _cyl(0.018, d * 0.75, P["metal"], cy=d * 0.15,
             z=z + h * 0.46),
        _cyl(0.12, h * 0.42, P["shade"], cy=d * 0.35, z=z),
    ], w, d


def build_towel_rail(P, w=0.68, z=1.05):
    """Wall-mounted towel rail with folded fabric."""
    return [
        _bx(w, 0.045, 0.035, P["metal"], z=z),
        _bx(0.30, 0.055, 0.48, _shade(P["shade"], 0.96),
            cx=-w * 0.12, z=z - 0.42),
    ], w, 0.08


def build_bathroom_shelf(P, w=0.72, z=1.05):
    """Wall shelf with folded towels and bath accessories."""
    towel = _mix_color(WHITE_SOFT, P["shade"], 0.20)
    meshes = [
        _rounded_cuboid(
            w, 0.20, 0.045, P["wood"], cy=0.06, z=z, roundness=0.10
        ),
        _rounded_cuboid(
            0.30, 0.15, 0.075, towel,
            cx=-w * 0.18, cy=0.06, z=z + 0.055, roundness=0.18,
        ),
        _rounded_cuboid(
            0.27, 0.14, 0.065, _shade(towel, 0.96),
            cx=-w * 0.18, cy=0.06, z=z + 0.13, roundness=0.18,
        ),
        _cyl(0.038, 0.18, P["accent"], cx=w * 0.20, cy=0.06, z=z + 0.05),
        _cyl(0.030, 0.14, _shade(P["metal"], 0.86),
             cx=w * 0.32, cy=0.06, z=z + 0.05),
    ]
    apply_archviz_material(
        meshes[0], "warm_oak", tint=P["wood"], tint_strength=0.25, repeat_m=0.8
    )
    for mesh in meshes[1:3]:
        apply_archviz_material(
            mesh, "curtain_fabric", tint=towel, tint_strength=0.58, repeat_m=0.4
        )
    return meshes, w, 0.22


def build_pendant(P, ceiling_h=WALL_H, drop=0.6):
    professional = _professional_detail(
        "ceiling_light", P, 0.42, 0.42, 0.31, z=ceiling_h - 0.34
    )
    if professional:
        return professional, 0.42, 0.42
    return [
        _cyl(0.012, drop - 0.14, P["metal"], z=ceiling_h - drop + 0.14),
        _cyl(0.19, 0.14, P["shade"], z=ceiling_h - drop),
    ], 0.4, 0.4


# ================= WALLS / OPENINGS =================
def build_room_edges(room_m):
    """Merged wall edges for one room polygon (meters)."""
    raw = []
    n = len(room_m)
    for i in range(n):
        p1, p2 = np.array(room_m[i]), np.array(room_m[(i + 1) % n])
        if np.linalg.norm(p2 - p1) < 0.05:
            continue
        raw.append({"p1": p1, "p2": p2, "line": LineString([p1, p2]),
                    "length": float(LineString([p1, p2]).length), "openings": []})
    return merge_edges(raw)


def assign_openings(all_room_edges, doors_m, windows_m):
    """Cut door/window spans into EVERY wall edge they touch.

    Assigning to all edges within tolerance (not just the single closest)
    means a door drawn between two adjacent rooms opens BOTH coincident
    walls, so the walkthrough can actually pass through. Only the FIRST
    edge that receives a door gets the door frame ("door"); the other
    coincident walls just get the hole cut ("door_hole") so the doorway
    isn't drawn twice.
    """
    def _cut(e, a, b, min_w, require_along, seg_len):
        """Try to cut opening [a, b] into edge e. Returns True if cut.

        require_along rejects edges the segment does not run along — a
        segment ending near a corner is also 'close' to the PERPENDICULAR
        wall and would otherwise punch a spurious hole into it.
        """
        L = e["length"]
        if L < 0.4:
            return False
        t0 = e["line"].project(Point(a)) / L
        t1 = e["line"].project(Point(b)) / L
        if require_along and seg_len > 0.2 and abs(t1 - t0) * L < 0.5 * seg_len:
            return False
        tc = (t0 + t1) / 2
        half = max(abs(t1 - t0) * L, min_w) / 2
        lo = max(0.05, tc * L - half) / L              # absolute 5 cm margins
        hi = min(L - 0.05, tc * L + half) / L
        if (hi - lo) * L < 0.25:
            return False
        return (lo, hi)

    door_infos = []
    for a, b in doors_m:
        seg = LineString([a, b])
        seg_len = float(np.linalg.norm(np.array(b) - np.array(a)))
        placed_any = False
        for require_along in (True, False):    # fallback pass for doors
            for edges in all_room_edges:       # drawn ACROSS the wall
                for e in edges:
                    if e["line"].distance(seg) > OPENING_EDGE_TOL:
                        continue
                    cut = _cut(e, a, b, MIN_DOOR_W, require_along, seg_len)
                    if cut:
                        e["openings"].append(
                            ("door" if not placed_any else "door_hole",
                             cut[0], cut[1]))
                        placed_any = True
            if placed_any:
                break
        if placed_any:
            door_infos.append((np.array(a), np.array(b)))
    for item in windows_m:
        a, b = item[0], item[1]
        seg = LineString([a, b])
        seg_len = float(np.linalg.norm(np.array(b) - np.array(a)))
        placed_any = False
        for require_along in (True, False):
            for edges in all_room_edges:
                for e in edges:
                    if e["line"].distance(seg) > OPENING_EDGE_TOL:
                        continue
                    cut = _cut(e, a, b, MIN_WINDOW_W, require_along, seg_len)
                    if cut:
                        e["openings"].append(("window", cut[0], cut[1]))
                        placed_any = True
            if placed_any:
                break
    return door_infos


DOOR_FRAME_COLOR = [0.36, 0.25, 0.15]


def _door_frame(op1, op2, wall_angle, frame_color=None):
    """Open doorway trim with a slim, palette-coordinated architectural casing."""
    op1, op2 = np.array(op1), np.array(op2)
    L = float(np.linalg.norm(op2 - op1))
    if L < 1e-6:
        return []
    # Deep enough to close the jamb without creating the former bulky portal.
    depth = max(WALL_THICKNESS + 0.12, 0.22)
    casing_color = frame_color or DOOR_FRAME_COLOR

    def bar(x0, x1, z0, z1):
        m = o3d.geometry.TriangleMesh.create_box(width=x1 - x0, height=depth,
                                                 depth=z1 - z0)
        m.translate((x0, -depth / 2, z0))
        m.rotate(_rotz(wall_angle), center=(0, 0, 0))
        m.translate((op1[0], op1[1], 0))
        return _paint(m, casing_color)

    return [bar(-0.015, 0.055, 0.0, DOOR_HEIGHT + 0.055),
            bar(L - 0.055, L + 0.015, 0.0, DOOR_HEIGHT + 0.055),
            bar(-0.015, L + 0.015, DOOR_HEIGHT - 0.02, DOOR_HEIGHT + 0.055)]


def build_walls(edges, wall_color, material_name=None, trim_color=None):
    """Wall meshes with door/window cutouts, at walkthrough wall height.

    Wall pieces that end at a polygon corner are extended slightly so
    neighbouring walls overlap instead of leaving hairline see-through
    slits (important for auto-detected polygons, which are not exact).
    """
    meshes = []
    wall_meshes = []

    def add_wall(mesh):
        if mesh is not None:
            wall_meshes.append(mesh)

    for e in edges:
        p1, p2 = np.array(e["p1"]), np.array(e["p2"])
        if e["length"] < 0.1:
            continue
        u = (p2 - p1) / np.linalg.norm(p2 - p1)
        ext = u * (WALL_THICKNESS * 0.75)

        def seg(a, b, z0, z1, ext_a=False, ext_b=False):
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)
            if ext_a:
                a = a - ext
            if ext_b:
                b = b + ext
            return wall_segment(a, b, z0, z1, wall_color)

        wall_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        length = e["length"]
        last = 0.0
        for typ, t0, t1 in sorted(e.get("openings", []), key=lambda x: x[1]):
            t0, t1 = max(0.0, min(1.0, t0)), max(0.0, min(1.0, t1))
            if t1 <= last:
                continue
            t0 = max(t0, last)
            if (t0 - last) * length > 0.02:     # absolute 2 cm, not 1% of edge
                w = seg(p1 + (p2 - p1) * last, p1 + (p2 - p1) * t0,
                        0, WALL_H, ext_a=(last <= 0.001))
                add_wall(w)
            op1, op2 = p1 + (p2 - p1) * t0, p1 + (p2 - p1) * t1
            if typ in ("door", "door_hole"):
                w = wall_segment(op1, op2, DOOR_HEIGHT, WALL_H, wall_color)
                add_wall(w)
                if typ == "door":
                    meshes.extend(
                        _door_frame(
                            op1,
                            op2,
                            wall_angle,
                            frame_color=trim_color,
                        )
                    )
            else:
                w = wall_segment(op1, op2, 0, WINDOW_SILL, wall_color)
                add_wall(w)
                w = wall_segment(op1, op2, WINDOW_SILL + WINDOW_HEIGHT, WALL_H,
                                 wall_color)
                add_wall(w)
                for window_mesh in create_window_geometry(
                    op1, op2, wall_angle
                ):
                    vertices = np.asarray(window_mesh.vertices)
                    # plan3d's legacy renderer fakes sunlight with a large
                    # cream trapezoid laid over the floor. Under PBR it reads
                    # as a badly fitted rug and overlaps the real carpets.
                    # Keep the modeled glass, reveals and frame; real scene
                    # lighting supplies the daylight in the walkthrough.
                    is_fake_floor_spill = (
                        len(vertices)
                        and float(vertices[:, 2].max()) <= 0.02
                        and float(vertices[:, 2].min()) >= -0.001
                        and len(vertices) <= 4
                    )
                    if not is_fake_floor_spill:
                        meshes.append(window_mesh)
            last = t1
        if (1.0 - last) * length > 0.02:        # absolute 2 cm, not 1% of edge
            w = seg(p1 + (p2 - p1) * last, p2, 0, WALL_H,
                    ext_a=(last <= 0.001), ext_b=True)
            add_wall(w)
    if wall_meshes:
        combined = wall_meshes[0]
        for part in wall_meshes[1:]:
            combined += part
        if material_name:
            apply_archviz_material(
                combined,
                material_name,
                tint=wall_color,
                tint_strength=0.32 if material_name != "wallpaper" else 0.18,
            )
        meshes.insert(0, combined)
    return meshes


BASEBOARD_H = 0.10
CORNICE_H = 0.06


def _wall_strip(a, b, z0, z1, thick, color, inward, offset):
    """A thin box hugging the interior face of a wall between a and b."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    L = float(np.linalg.norm(b - a))
    # Decorative battens and vertical moulding rails are intentionally narrow.
    # The former 5 cm cutoff silently discarded them, leaving only the plain
    # backing panel even when Wood slats or Panel moulding was selected.
    if L < 0.015:
        return None
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    m = o3d.geometry.TriangleMesh.create_box(width=L, height=thick, depth=z1 - z0)
    m.translate((0, -thick / 2, 0))
    m.rotate(_rotz(ang), center=(0, 0, 0))
    base = a + inward * offset
    m.translate((base[0], base[1], z0))
    return _paint(m, color)


def build_wall_finish_skins(room_m, edges, wall_color, material_name):
    """Apply each room's wall finish only to its own interior wall face.

    Structural walls are shared by adjacent rooms. Mapping a bathroom tile
    texture around the whole structural box made that tile appear on the
    living-room side too. These thin opening-aware skins give both sides of a
    shared wall independent, correctly selected finishes.
    """
    poly = Polygon([(p[0], p[1]) for p in room_m])
    if not poly.is_valid:
        poly = poly.buffer(0)
    meshes = []
    tint_strength = {
        "wallpaper": 0.14,
        "limewash": 0.30,
        "bathroom_tile": 0.20,
        "concrete": 0.18,
    }.get(material_name, 0.30)
    surface_material = (
        "marble"
        if material_name == "bathroom_tile"
        else material_name
    )

    for edge in edges:
        p1 = np.asarray(edge["p1"], dtype=float)
        p2 = np.asarray(edge["p2"], dtype=float)
        length = float(edge["length"])
        if length < 0.10:
            continue
        direction = (p2 - p1) / length
        normal = np.array([-direction[1], direction[0]])
        middle = (p1 + p2) / 2
        inward = (
            normal
            if poly.contains(Point(*(middle + normal * 0.25)))
            else -normal
        )

        def add_skin(a, b, z0, z1):
            skin = _wall_strip(
                a, b, z0, z1, 0.008, wall_color, inward,
                WALL_THICKNESS / 2 + 0.008,
            )
            if skin is None:
                return
            apply_archviz_material(
                skin,
                surface_material,
                tint=wall_color,
                tint_strength=tint_strength,
                # Bathroom walls use the installed mineral slab map rather
                # than enlarging a small-mosaic texture into heavy grout.
                repeat_m=3.2 if material_name == "bathroom_tile" else None,
                detail_maps=material_name not in {"plaster", "limewash"},
            )
            meshes.append(skin)

        last = 0.0
        for typ, t0, t1 in sorted(
            edge.get("openings", []), key=lambda opening: opening[1]
        ):
            t0 = max(last, min(1.0, max(0.0, float(t0))))
            t1 = max(t0, min(1.0, max(0.0, float(t1))))
            if (t0 - last) * length > 0.018:
                add_skin(
                    p1 + (p2 - p1) * last,
                    p1 + (p2 - p1) * t0,
                    0.0,
                    WALL_H,
                )
            opening_a = p1 + (p2 - p1) * t0
            opening_b = p1 + (p2 - p1) * t1
            if typ in ("door", "door_hole"):
                add_skin(opening_a, opening_b, DOOR_HEIGHT, WALL_H)
            else:
                add_skin(opening_a, opening_b, 0.0, WINDOW_SILL)
                add_skin(
                    opening_a,
                    opening_b,
                    WINDOW_SILL + WINDOW_HEIGHT,
                    WALL_H,
                )
            last = t1
        if (1.0 - last) * length > 0.018:
            add_skin(
                p1 + (p2 - p1) * last,
                p2,
                0.0,
                WALL_H,
            )
    return meshes


def _pleated_curtain_panel(
    a, b, z0, z1, color, inward, offset, tint_strength=0.62
):
    """Build a gathered, double-sided corrugated fabric panel."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    width = float(np.linalg.norm(b - a))
    if width < 0.08:
        return None
    direction = (b - a) / width
    folds = max(4, int(round(width / 0.11)))
    columns = folds * 4
    rows = 6
    thickness = 0.010
    amplitude = min(0.065, width / (folds * 2.4))
    center = (a + b) / 2
    vertices = []
    grid_uv = []
    for surface in (1.0, -1.0):
        for row in range(rows):
            v = row / (rows - 1)
            z = z0 + (z1 - z0) * v
            gathered_width = width * (0.86 + 0.14 * v)
            fullness = 0.92 + 0.12 * v
            for column in range(columns + 1):
                u = column / columns
                along = direction * ((u - 0.5) * gathered_width)
                fold = (
                    math.sin(u * folds * math.tau)
                    * amplitude
                    * fullness
                )
                point = (
                    center
                    + along
                    + inward * (offset + fold + surface * thickness / 2)
                )
                vertices.append((point[0], point[1], z))
                grid_uv.append(
                    (u * max(width / 0.42, 1.0), v * (z1 - z0) / 0.42)
                )

    triangles = []
    triangle_uvs = []
    surface_size = rows * (columns + 1)
    for surface_index in range(2):
        base = surface_index * surface_size
        for row in range(rows - 1):
            for column in range(columns):
                p0 = base + row * (columns + 1) + column
                p1 = p0 + 1
                p2 = p0 + columns + 1
                p3 = p2 + 1
                faces = (
                    ((p0, p2, p1), (p1, p2, p3))
                    if surface_index == 0
                    else ((p0, p1, p2), (p1, p3, p2))
                )
                for triangle in faces:
                    triangles.append(triangle)
                    triangle_uvs.extend(grid_uv[index] for index in triangle)

    # Close both side hems so the panel reads as fabric volume from oblique
    # walkthrough angles instead of disappearing like a one-sided plane.
    for column in (0, columns):
        for row in range(rows - 1):
            front0 = row * (columns + 1) + column
            front1 = (row + 1) * (columns + 1) + column
            back0 = surface_size + front0
            back1 = surface_size + front1
            for triangle in (
                (front0, back0, front1),
                (front1, back0, back1),
            ):
                triangles.append(triangle)
                triangle_uvs.extend(grid_uv[index] for index in triangle)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    _paint(mesh, color)
    apply_archviz_material(
        mesh,
        "curtain_fabric",
        tint=color,
        tint_strength=tint_strength,
        triangle_uvs=np.asarray(triangle_uvs, dtype=float),
    )
    return mesh


def build_room_trim(room_m, edges, P, config=None):
    """Baseboards + cornice + window curtains — the touches that make a room
    read as *designed* rather than a bare box. Baseboards skip door openings
    (so there's no bar across a doorway); curtains hang inside each window."""
    poly = Polygon([(p[0], p[1]) for p in room_m])
    if not poly.is_valid:
        poly = poly.buffer(0)
    meshes = []
    trim_col = _shade(P["wall"], 0.82)
    drape = _shade(P["shade"], 0.94)
    choices = room_design_choices(config, (config or {}).get("room_type", ""))
    curtain_design = choices["curtain_design"]

    for e in edges:
        p1, p2 = np.array(e["p1"]), np.array(e["p2"])
        L = e["length"]
        if L < 0.3:
            continue
        dvec = (p2 - p1) / L
        n2 = np.array([-dvec[1], dvec[0]])
        mid = (p1 + p2) / 2
        inward = n2 if poly.contains(Point(*(mid + n2 * 0.25))) else -n2
        off = WALL_THICKNESS / 2

        # baseboard along solid floor spans (skip door gaps, keep under windows)
        door_spans = sorted((t0, t1) for typ, t0, t1 in e.get("openings", [])
                            if typ in ("door", "door_hole"))
        cur = 0.0
        spans = []
        for t0, t1 in door_spans:
            if t0 > cur:
                spans.append((cur, t0))
            cur = max(cur, t1)
        if cur < 1.0:
            spans.append((cur, 1.0))
        for s0, s1 in spans:
            a = p1 + (p2 - p1) * s0
            b = p1 + (p2 - p1) * s1
            bb = _wall_strip(a, b, 0.0, BASEBOARD_H, 0.03, trim_col, inward, off)
            if bb:
                meshes.append(bb)
        # cornice along the whole wall at the ceiling
        cor = _wall_strip(p1, p2, WALL_H - CORNICE_H, WALL_H, 0.04,
                          _shade(P["wall"], 1.02), inward, off)
        if cor:
            meshes.append(cor)

        # curtains at each window
        for typ, t0, t1 in e.get("openings", []):
            if typ != "window" or curtain_design == "none":
                continue
            wa = p1 + (p2 - p1) * t0
            wb = p1 + (p2 - p1) * t1
            wwidth = float(np.linalg.norm(wb - wa))
            top = WINDOW_SILL + WINDOW_HEIGHT + 0.22
            rod_color = _mix_color(
                P["metal"], [0.18, 0.16, 0.14], 0.58
            )
            rod_a_2d = wa - dvec * 0.14 + inward * (off + 0.10)
            rod_b_2d = wb + dvec * 0.14 + inward * (off + 0.10)
            rod = _cylinder_between(
                (rod_a_2d[0], rod_a_2d[1], top),
                (rod_b_2d[0], rod_b_2d[1], top),
                0.016,
                rod_color,
                resolution=18,
            )
            if rod:
                meshes.append(rod)
                for point in (rod_a_2d, rod_b_2d):
                    finial = _sph(
                        0.028, rod_color,
                        cx=point[0], cy=point[1], z=top,
                    )
                    meshes.append(finial)
            if curtain_design in ("sheer panels", "layered sheers + drapes"):
                sheer = _mix_color(WHITE_SOFT, P["shade"], 0.18)
                sheer_spans = (
                    ((wa, wb),)
                    if curtain_design == "layered sheers + drapes"
                    else (
                        (wa, wa + (wb - wa) * 0.48),
                        (wa + (wb - wa) * 0.52, wb),
                    )
                )
                for sheer_a, sheer_b in sheer_spans:
                    panel = _pleated_curtain_panel(
                        sheer_a, sheer_b, 0.10, top, sheer, inward,
                        off + 0.065, tint_strength=0.20,
                    )
                    if panel:
                        meshes.append(panel)
            if curtain_design in ("linen drapes", "layered sheers + drapes"):
                panel_w = max(0.20, wwidth * (
                    0.30 if curtain_design == "layered sheers + drapes" else 0.25
                ))
                outer_offset = off + (
                    0.135 if curtain_design == "layered sheers + drapes" else 0.10
                )
                for end, sgn in ((wa, 1), (wb, -1)):
                    pa = end + dvec * sgn * 0.02
                    pb = end + dvec * sgn * (0.02 + panel_w)
                    panel = _pleated_curtain_panel(
                        pa, pb, 0.10, top, drape, inward, outer_offset,
                        tint_strength=0.58,
                    )
                    if panel:
                        meshes.append(panel)
    return meshes


def _polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [
            part for part in geometry.geoms
            if part.geom_type == "Polygon" and part.area > 1e-5
        ]
    return []


def build_floor_finish(room_m, P, room_type, style, config=None):
    """Add clipped flooring joints so the floor reads as a designed material."""
    config = config or {}
    poly = Polygon([(p[0], p[1]) for p in room_m]).buffer(-0.025)
    if poly.is_empty:
        return []
    minx, miny, maxx, maxy = poly.bounds
    room_type = (room_type or "").lower()
    style = (style or "").lower()
    finish = str(config.get("floor_finish", "Auto by style")).lower()
    if finish == "polished concrete":
        return []
    if finish == "terrazzo":
        rng = random.Random(
            int(hashlib.sha1(str(poly.bounds).encode("utf-8")).hexdigest()[:8], 16)
        )
        specks = []
        colors = (
            _shade(P["floor"], 0.60),
            _shade(P["floor"], 1.18),
            _mix_color(P["floor"], P["accent"], 0.42),
        )
        attempts = max(45, int(poly.area * 5))
        for index in range(attempts):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if not poly.contains(Point(x, y)):
                continue
            size = rng.uniform(0.025, 0.065)
            chip = _bx(size, size * rng.uniform(0.45, 1.0), 0.004,
                       colors[index % len(colors)], z=0.002)
            chip.rotate(_rotz(rng.uniform(0, math.pi)), center=(0, 0, 0))
            chip.translate((x, y, 0))
            specks.append(chip)
        return specks

    tiled = (
        finish in ("natural stone", "large tile")
        or "bath" in room_type
        or ("kitchen" in room_type and finish == "auto by style")
        or ("industrial" in style and finish == "auto by style")
    )
    spacing = 0.82 if finish in ("natural stone", "large tile") else (
        0.62 if tiled else 0.24
    )
    seam_color = _shade(P["floor"], 0.76 if tiled else 0.82)
    lines = []

    if tiled:
        x = math.floor(minx / spacing) * spacing
        while x <= maxx:
            lines.append(LineString([(x, miny), (x, maxy)]))
            x += spacing
        y = math.floor(miny / spacing) * spacing
        while y <= maxy:
            lines.append(LineString([(minx, y), (maxx, y)]))
            y += spacing
    else:
        # Boards follow the room's longest direction for a calmer composition.
        horizontal = (maxx - minx) >= (maxy - miny)
        start, end = (miny, maxy) if horizontal else (minx, maxx)
        value = math.floor(start / spacing) * spacing
        while value <= end:
            if horizontal:
                lines.append(LineString([(minx, value), (maxx, value)]))
            else:
                lines.append(LineString([(value, miny), (value, maxy)]))
            value += spacing

        # Staggered end joints make the strips read as individual wood planks.
        long_start, long_end = (minx, maxx) if horizontal else (miny, maxy)
        board_length = 1.25
        row = 0
        across = math.floor(start / spacing) * spacing
        while across <= end:
            joint = long_start + (0.38 if row % 2 else 0.0)
            while joint <= long_end:
                if horizontal:
                    lines.append(LineString([
                        (joint, across - spacing / 2),
                        (joint, across + spacing / 2),
                    ]))
                else:
                    lines.append(LineString([
                        (across - spacing / 2, joint),
                        (across + spacing / 2, joint),
                    ]))
                joint += board_length
            across += spacing
            row += 1

    meshes = []
    width = 0.010 if tiled else 0.006
    for line in lines:
        clipped = line.buffer(width / 2, cap_style=2).intersection(poly)
        for part in _polygon_parts(clipped):
            coords = list(part.exterior.coords)[:-1]
            if len(coords) < 3:
                continue
            mesh = floor_mesh(coords, seam_color)
            mesh.translate((0, 0, 0.004))
            meshes.append(mesh)
    return meshes


def _free_wall_spans(edges):
    spans = []
    for edge in edges:
        p1, p2 = np.array(edge["p1"]), np.array(edge["p2"])
        length = float(edge["length"])
        if length < 0.8:
            continue
        blocked = sorted(
            (max(0.0, start), min(1.0, end))
            for _kind, start, end in edge.get("openings", [])
        )
        cursor = 0.0
        for start, end in blocked:
            if start > cursor:
                spans.append((length * (start - cursor),
                              p1 + (p2 - p1) * cursor,
                              p1 + (p2 - p1) * start))
            cursor = max(cursor, end)
        if cursor < 1.0:
            spans.append((length * (1.0 - cursor),
                          p1 + (p2 - p1) * cursor, p2))
    return sorted(spans, key=lambda item: -item[0])


def build_room_design_surfaces(room_m, edges, P, config):
    """Build architectural finishes directly from the user's preferences."""
    room_type = config.get("room_type", "Living Room")
    style = config.get("style", "Modern")
    profile = str(config.get("design_profile", "Curated")).lower()
    wall_finish = str(config.get("wall_finish", "Auto by style")).lower()
    choices = room_design_choices(config, room_type)
    style_key = choices["style_key"]
    treatment = (
        choices["wall_treatment"]
        if wall_finish == "auto by style"
        else wall_finish
    )
    poly = Polygon([(p[0], p[1]) for p in room_m])
    if not poly.is_valid:
        poly = poly.buffer(0)
    # The room slab now carries a tiled PBR floor. Extra procedural lines
    # duplicated its real boards/grout and made the finish look synthetic.
    meshes = []

    # One focal wall, chosen from the longest uninterrupted wall span.
    spans = _free_wall_spans(edges)
    service_room = any(
        word in str(room_type).lower()
        for word in ("kitchen", "bath", "laundry", "utility")
    )
    if spans and spans[0][0] >= 1.25 and not service_room:
        length, a, b = spans[0]
        direction = (b - a) / max(length, 1e-9)
        normal = np.array([-direction[1], direction[0]])
        middle = (a + b) / 2
        inward = normal if poly.contains(Point(*(middle + normal * 0.3))) else -normal
        max_width = 3.4 if profile == "layered" else 2.8
        width = min(length - 0.16, max_width)
        fa = middle - direction * width / 2
        fb = middle + direction * width / 2
        feature_weight = 0.52 if treatment == "accent color" else 0.16
        if treatment == "wood slats":
            feature_color = _mix_color(P["wall"], P["wood"], 0.52)
        elif treatment == "panel moulding":
            feature_color = _mix_color(P["wall"], WHITE_SOFT, 0.12)
        else:
            feature_color = _mix_color(P["wall"], P["accent"], feature_weight)
        feature_treatments = {"wood slats", "panel moulding", "accent color"}
        if treatment in feature_treatments:
            panel = _wall_strip(
                fa, fb, 0.11, WALL_H - 0.10, 0.028, feature_color,
                inward, WALL_THICKNESS / 2 + 0.018,
            )
            if panel:
                apply_archviz_material(
                    panel,
                    "plaster",
                    tint=feature_color,
                    tint_strength=0.52,
                    repeat_m=1.5,
                    detail_maps=False,
                )
                meshes.append(panel)

        if treatment == "wood slats":
            slat_color = _shade(P["wood_dark"], 0.92)
            spacing = 0.23 if profile == "airy" else 0.17
            offset = -width / 2 + spacing / 2
            while offset < width / 2:
                center = middle + direction * offset
                sa = center - direction * 0.018
                sb = center + direction * 0.018
                slat = _wall_strip(
                    sa, sb, 0.16, WALL_H - 0.16, 0.04, slat_color,
                    inward, WALL_THICKNESS / 2 + 0.052,
                )
                if slat:
                    meshes.append(slat)
                offset += spacing
        elif treatment == "panel moulding":
            # Calm full-height proportions match the classic furniture family
            # without mixing in the former industrial sconces or timber slats.
            moulding = _shade(P["wall"], 0.78)
            for z in (0.88, 2.10):
                rail = _wall_strip(
                    fa, fb, z, z + 0.035, 0.04, moulding,
                    inward, WALL_THICKNESS / 2 + 0.052,
                )
                if rail:
                    meshes.append(rail)
            for offset in (-width * 0.28, width * 0.28):
                center = middle + direction * offset
                va = center - direction * 0.018
                vb = center + direction * 0.018
                rail = _wall_strip(
                    va, vb, 0.28, 2.38, 0.04, moulding,
                    inward, WALL_THICKNESS / 2 + 0.052,
                )
                if rail:
                    meshes.append(rail)

        # Sconces are now part of a compatible classic/industrial lighting
        # family instead of being mixed into every room style.
        use_sconces = (
            choices["decor_set"] != "minimal"
            and style_key in ("classic", "industrial")
            and profile != "airy"
            and width >= 2.0
            and not service_room
        )
        if use_sconces:
            for offset in (-width * 0.34, width * 0.34):
                center = middle + direction * offset
                sconce = build_wall_sconce(P)
                sconce_pos = (
                    center
                    + inward * (
                        WALL_THICKNESS / 2 + sconce[2] / 2 + 0.025
                    )
                )
                meshes.extend(place_meshes(
                    list(sconce[0]),
                    sconce_pos,
                    yaw_facing(inward),
                ))

    # A shallow inset ceiling creates a deliberate cove instead of a bare lid.
    inner = poly.buffer(-0.28)
    ceiling_color = P.get("ceiling", CEILING_COLOR)
    for part in _polygon_parts(inner):
        coords = list(part.exterior.coords)[:-1]
        if len(coords) < 3:
            continue
        tray = _orient_horizontal_surface(
            floor_mesh(
                coords,
                _mix_color(CEILING_COLOR, ceiling_color, 0.35),
            ),
            upward=False,
        )
        tray.translate((0, 0, WALL_H - 0.045))
        meshes.append(tray)

        # Warm cove strip around the inset ceiling.
        for a, b in zip(coords, coords[1:] + coords[:1]):
            a, b = np.array(a), np.array(b)
            d = b - a
            length = np.linalg.norm(d)
            if length < 0.10:
                continue
            normal = np.array([-d[1], d[0]]) / length
            cove = _wall_strip(
                a, b, WALL_H - 0.11, WALL_H - 0.075, 0.025,
                _shade(P["shade"], 1.08), normal, 0.0,
            )
            if cove:
                meshes.append(cove)

    # Recessed downlights distributed inside the room, with denser lighting
    # for layered schemes and larger rooms.
    lighting_area = poly.buffer(-0.55)
    if not lighting_area.is_empty:
        minx, miny, maxx, maxy = lighting_area.bounds
        spacing = 1.45 if profile == "layered" else 1.75
        xs = np.arange(minx + spacing / 2, maxx + 0.01, spacing)
        ys = np.arange(miny + spacing / 2, maxy + 0.01, spacing)
        for x in xs:
            for y in ys:
                if not lighting_area.contains(Point(float(x), float(y))):
                    continue
                meshes.append(_cyl(
                    0.085, 0.018, _shade(P["metal"], 0.72),
                    cx=float(x), cy=float(y), z=WALL_H - 0.025,
                ))
                meshes.append(_cyl(
                    0.060, 0.020, _shade(P["shade"], 1.10),
                    cx=float(x), cy=float(y), z=WALL_H - 0.047,
                ))

    return meshes


# ================= FURNISHING ENGINE =================
EDITABLE_FURNITURE_ASSETS = {
    "sofa",
    "armchair",
    "coffee_table",
    "tv_unit",
    "bed",
    "nightstand",
    "wardrobe",
    "kitchen_island",
    "fridge",
    "dining_table",
    "dining_chair",
    "sideboard",
    "desk",
    "office_chair",
    "bookshelf",
    "vanity",
    "toilet",
    "shower",
    "bathtub",
}


def _tripo_material_color(asset_key, palette):
    """Replace image-projected colors with a real room material."""
    if asset_key in {"sofa", "armchair", "office_chair"}:
        return _mix_color(palette["sofa"], palette["wood_dark"], 0.12)
    if asset_key == "bed":
        return _mix_color(palette["sofa"], palette["cushion"], 0.38)
    if asset_key in {
        "coffee_table", "tv_unit", "nightstand", "dining_table",
        "dining_chair", "sideboard", "desk", "bookshelf",
    }:
        return palette["wood"]
    if asset_key in {"wardrobe", "kitchen_island", "vanity"}:
        return palette["cabinet"]
    if asset_key == "fridge":
        return _mix_color([0.70, 0.72, 0.74], palette["metal"], 0.18)
    if asset_key == "toilet":
        return [0.94, 0.94, 0.92]
    if asset_key in {"shower", "bathtub"}:
        return _mix_color([0.70, 0.82, 0.87], palette["metal"], 0.12)
    return palette["table"]


def _furniture_accessories(asset_key, procedural_meshes):
    """Keep functional styling that is separate from the catalog furniture."""
    if asset_key == "coffee_table":
        return procedural_meshes[-1:]
    if asset_key == "tv_unit":
        return procedural_meshes[2:]
    if asset_key == "nightstand":
        return procedural_meshes[2:]
    if asset_key == "dining_table":
        return procedural_meshes[5:]
    if asset_key == "sideboard":
        return procedural_meshes[2:]
    if asset_key == "kitchen_island":
        return procedural_meshes[2:]
    if asset_key == "desk":
        return procedural_meshes[3:]
    if asset_key == "bookshelf":
        return [
            mesh for index, mesh in enumerate(procedural_meshes)
            if index > 0 and (index - 1) % 5 != 0
        ]
    if asset_key == "vanity":
        # Keep the faucet, but replace the flat procedural mirror with the
        # professional modeled wall-mirror asset in the bathroom recipe.
        return procedural_meshes[3:4]
    return []


def _apply_smooth_material(mesh, color):
    """Shade a reconstructed surface smoothly without any image projection."""
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    if len(normals) != len(mesh.vertices):
        mesh.paint_uniform_color(color)
        return mesh
    shade = (
        0.56
        + 0.28 * np.clip(normals @ _KEY_DIR, 0, None)
        + 0.08 * np.clip(normals @ _FILL_DIR, 0, None)
        + 0.08 * np.clip(normals[:, 2], 0, None)
    )
    colors = np.clip(np.asarray(color)[None, :] * shade[:, None], 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


class RoomFurnisher:
    """Style-aware layout engine for a coherent, walkable 3D interior."""

    def __init__(self, room_m, edges, palette, config=None):
        self.room_m = room_m
        self.edges = edges
        self.P = palette
        self.config = config or {}
        self.design_profile = self.config.get("design_profile", "Curated")
        self.brief = str(self.config.get("design_notes", "")).lower()
        if any(word in self.brief for word in ("minimal", "no clutter", "simple")):
            self.design_profile = "Airy"
        elif any(word in self.brief for word in ("luxury", "luxurious", "rich decor")):
            self.design_profile = "Layered"
        resolved_config = dict(self.config)
        resolved_config["design_profile"] = self.design_profile
        self.design_choices = room_design_choices(
            resolved_config, self.config.get("room_type", "")
        )
        signature = "|".join([
            str(self.config.get("name", "room")),
            str(self.config.get("room_type", "")),
            str(self.config.get("style", "")),
            str(self.config.get("design_seed", 0)),
        ])
        self.rng = random.Random(int(hashlib.sha1(signature.encode("utf-8"))
                                     .hexdigest()[:12], 16))
        self.poly = Polygon([(p[0], p[1]) for p in room_m])
        if not self.poly.is_valid:
            self.poly = self.poly.buffer(0)
        self.inset = self.poly.buffer(-0.05)
        self.centroid = np.array([self.poly.centroid.x, self.poly.centroid.y])
        self.meshes = []
        self.placed = []          # blocking footprints (shapely)
        self.editable_objects = []
        self._editable_mesh_assets = {}
        self.door_zones = []      # keep-clear zones in front of doors
        for e in edges:
            p1, p2 = np.array(e["p1"]), np.array(e["p2"])
            for typ, t0, t1 in e.get("openings", []):
                if typ not in ("door", "door_hole"):
                    continue
                c = p1 + (p2 - p1) * ((t0 + t1) / 2)
                self.door_zones.append(Point(c[0], c[1]).buffer(0.95))

    @property
    def layered(self):
        return self.design_profile.lower() == "layered"

    @property
    def airy(self):
        return self.design_profile.lower() == "airy"

    @property
    def wants_plants(self):
        if "no plant" in self.brief:
            return False
        if "plant" in self.brief or "greenery" in self.brief:
            return True
        return self.design_choices["decor_set"] in (
            "art & greenery", "layered"
        )

    @property
    def wants_rugs(self):
        return (
            "no rug" not in self.brief
            and self.design_choices["rug_design"] != "none"
        )

    @property
    def rug_design(self):
        return self.design_choices["rug_design"]

    @property
    def wants_wall_decor(self):
        return self.design_choices["decor_set"] != "minimal"

    def place_rug(self, position, yaw, width, depth):
        """Fit one rug fully inside its room and away from doorway clear zones."""
        if not self.wants_rugs:
            return False
        safe_room = self.poly.buffer(-0.14)
        if safe_room.is_empty:
            return False
        position = np.asarray(position, dtype=float)
        scale = 1.0
        while scale >= 0.58:
            fitted_w = width * scale
            fitted_d = depth * scale
            footprint = footprint_poly(position, yaw, fitted_w, fitted_d)
            clears_doors = not any(
                footprint.intersects(zone.buffer(0.08))
                for zone in self.door_zones
            )
            if footprint.within(safe_room) and clears_doors:
                return self.add(
                    build_rug(
                        self.P,
                        w=fitted_w,
                        d=fitted_d,
                        design=self.rug_design,
                    ),
                    position,
                    yaw,
                    block=False,
                    avoid_doors=False,
                    check=False,
                )
            scale -= 0.08
        return False

    def furniture_builder(self, asset_key, procedural_builder):
        """Load a native catalog model, with Tripo kept for compatibility."""
        use_catalog = self.config.get("use_catalog", False)
        use_triposr = self.config.get("use_triposr", False)
        if (
            asset_key not in EDITABLE_FURNITURE_ASSETS
            or (not use_catalog and not use_triposr)
        ):
            return procedural_builder

        def build_with_local_asset(P, **kw):
            procedural = procedural_builder(P, **kw)
            procedural_meshes, width, depth = procedural
            try:
                mesh_depth = min(depth, 0.90) if asset_key == "kitchen_island" else depth
                if use_catalog:
                    from furniture_catalog import load_catalog_asset

                    style_name = self.config.get("style", "Modern")
                    use_designer_geometry = (
                        (
                            asset_key == "bed"
                            and not any(
                                word in str(style_name).lower()
                                for word in ("classic", "traditional")
                            )
                        )
                        or asset_key == "kitchen_island"
                    )
                    generated = (
                        None
                        if use_designer_geometry
                        else load_catalog_asset(
                            asset_key,
                            style_name,
                            width,
                            mesh_depth,
                            palette=P,
                        )
                    )
                else:
                    from local_3d_ai import load_asset_mesh, preference_key

                    generated = load_asset_mesh(
                        asset_key,
                        self.config.get("style", "Modern"),
                        width,
                        mesh_depth,
                        design_key=preference_key(self.config),
                    )
                if generated:
                    if use_triposr:
                        material = _tripo_material_color(asset_key, P)
                        for mesh in generated:
                            _apply_smooth_material(mesh, material)
                    generated.extend(
                        _furniture_accessories(asset_key, procedural_meshes)
                    )
                    if use_catalog:
                        for mesh in generated:
                            # Retain the object reference as well as its id.
                            # Failed placement attempts can otherwise release a
                            # mesh and let Python reuse its id for unrelated
                            # décor, incorrectly making that décor editable.
                            self._editable_mesh_assets[id(mesh)] = (
                                mesh,
                                asset_key,
                            )
                    return generated, width, depth
            except Exception as exc:
                engine = "catalog" if use_catalog else "TripoSR"
                print(f"[WALK] {engine} asset '{asset_key}' unavailable: {exc}")
            if use_catalog:
                # Procedural fallbacks are still genuine transformable 3D
                # components. Keep them editable when a style-specific catalog
                # model is deliberately avoided or cannot be loaded.
                for mesh in procedural_meshes:
                    self._editable_mesh_assets[id(mesh)] = (
                        mesh,
                        asset_key,
                    )
            return procedural

        return build_with_local_asset

    # ---- geometry helpers ----
    def _inward_normal(self, p1, p2):
        d = np.array(p2) - np.array(p1)
        L = np.linalg.norm(d)
        if L < 1e-9:
            return np.array([0.0, 1.0])
        n = np.array([-d[1], d[0]]) / L
        mid = (np.array(p1) + np.array(p2)) / 2
        return n if self.poly.contains(Point(*(mid + n * 0.3))) else -n

    def wall_slots(self):
        """Free wall runs (no door/window), longest first."""
        slots = []
        for e in self.edges:
            L = e["length"]
            if L < 0.7:
                continue
            p1, p2 = np.array(e["p1"]), np.array(e["p2"])
            pad = 0.30 / L
            blocked = sorted((max(0.0, t0 - pad), min(1.0, t1 + pad))
                             for _, t0, t1 in e.get("openings", []))
            endm = 0.18 / L
            cur, free = endm, []
            for b0, b1 in blocked:
                if b0 > cur:
                    free.append((cur, b0))
                cur = max(cur, b1)
            if cur < 1 - endm:
                free.append((cur, 1 - endm))
            dvec = p2 - p1
            dnorm = dvec / np.linalg.norm(dvec)
            n = self._inward_normal(p1, p2)
            for f0, f1 in free:
                flen = (f1 - f0) * L
                if flen < 0.5:
                    continue
                slots.append(dict(edge=e, p1=p1, p2=p2, t0=f0, t1=f1, len=flen,
                                  mid=p1 + dvec * ((f0 + f1) / 2),
                                  dir=dnorm, n=n))
        slots.sort(key=lambda s: -s["len"])
        return slots

    def _centered_wall_pose(self, slot, width, depth):
        """Return the game-style snapped pose for an item on a wall slot."""
        n = slot["n"]
        pos = slot["mid"] + n * (WALL_GAP + depth / 2)
        return pos, yaw_facing(n)

    def _candidate_clear(self, fp):
        """Check a proposed template footprint without changing scene state."""
        return (
            fp.within(self.inset)
            and not any(fp.intersects(zone) for zone in self.door_zones)
        )

    def living_anchor_slots(self, sofa_width=2.35):
        """Rank centered sofa walls by the quality of the complete composition."""
        slots = self.wall_slots()

        def score(slot):
            sofa_pos, sofa_yaw = self._centered_wall_pose(
                slot, sofa_width, 0.95
            )
            sofa_fp = footprint_poly(
                sofa_pos, sofa_yaw, sofa_width, 0.95
            )
            if (
                slot["len"] < sofa_width + 0.12
                or not self._candidate_clear(sofa_fp)
            ):
                return -1e6

            n, side = slot["n"], slot["dir"]
            score_value = min(slot["len"], 4.5)
            if not slot["edge"].get("openings"):
                score_value += 2.5

            # Score the entire conversation zone, not just the sofa. This
            # prevents a sofa from anchoring a corridor wall in an irregular
            # open-plan apartment where the coffee table and chairs cannot fit.
            zone_center = sofa_pos + n * 1.30
            zone_width = min(4.45, sofa_width + 1.95)
            zone = footprint_poly(
                zone_center, sofa_yaw, zone_width, 3.25
            )
            safe_room = self.poly.buffer(-0.10)
            if safe_room.is_empty:
                return -1e6
            coverage = zone.intersection(safe_room).area / max(zone.area, 1e-9)
            score_value += coverage * 7.0
            if coverage < 0.82:
                score_value -= (0.82 - coverage) * 18.0

            table_pos = sofa_pos + n * 1.5
            table_fp = footprint_poly(table_pos, sofa_yaw, 1.1, 0.6)
            if self._candidate_clear(table_fp):
                score_value += 3.0
            else:
                score_value -= 6.0

            chair_clear = 0
            for direction in (-1, 1):
                chair_pos = table_pos + side * direction * 1.6
                facing = table_pos - chair_pos
                facing /= np.linalg.norm(facing) + 1e-9
                chair_fp = footprint_poly(
                    chair_pos, yaw_facing(facing), 0.92, 0.85
                )
                if self._candidate_clear(chair_fp):
                    chair_clear += 1
            score_value += chair_clear * 1.6

            has_opposing_media_wall = any(
                other["len"] >= 1.35
                and np.dot(other["n"], n) < -0.6
                for other in slots
            )
            if has_opposing_media_wall:
                score_value += 3.0
            return score_value

        return sorted(slots, key=score, reverse=True)

    def find_open_pose(self, width, depth, preferred=None, yaws=None):
        """Find the best empty interior zone for a complete furniture group."""
        safe_room = self.poly.buffer(-0.12)
        if safe_room.is_empty:
            return None
        if yaws is None:
            yaws = [
                yaw_facing(slot["n"]) for slot in self.wall_slots()[:6]
            ] or [0.0, math.pi / 2]
        # Avoid evaluating the same axis repeatedly on rectilinear plans.
        unique_yaws = []
        for yaw in yaws:
            axis_yaw = float(yaw) % math.pi
            if not any(
                abs(math.sin(axis_yaw - known)) < 0.08
                for known in unique_yaws
            ):
                unique_yaws.append(axis_yaw)

        minx, miny, maxx, maxy = safe_room.bounds
        spacing = 0.42
        margin = min(0.38, min(width, depth) * 0.18)
        xs = np.arange(minx + margin, maxx - margin + 0.01, spacing)
        ys = np.arange(miny + margin, maxy - margin + 0.01, spacing)
        points = [(float(x), float(y)) for x in xs for y in ys]
        rep = safe_room.representative_point()
        points.append((rep.x, rep.y))
        if preferred is not None:
            points.append((float(preferred[0]), float(preferred[1])))

        best = None
        for x, y in points:
            position = np.array([x, y], dtype=float)
            for yaw in unique_yaws:
                fp = footprint_poly(position, yaw, width, depth)
                if not fp.within(safe_room):
                    continue
                if any(fp.intersects(zone) for zone in self.door_zones):
                    continue
                if any(fp.intersects(placed) for placed in self.placed):
                    continue
                boundary_clearance = fp.distance(self.poly.boundary)
                placed_clearance = min(
                    (fp.distance(placed) for placed in self.placed),
                    default=0.0,
                )
                score = boundary_clearance * 1.2 + placed_clearance * 2.0
                if self.placed:
                    # In open-plan rooms, intentionally separate the dining
                    # zone from the conversation group.
                    score += min(2.0, min(
                        np.linalg.norm(
                            position
                            - np.array(
                                [placed.centroid.x, placed.centroid.y]
                            )
                        )
                        for placed in self.placed
                    )) * 0.55
                if preferred is not None:
                    score -= np.linalg.norm(
                        position - np.asarray(preferred, dtype=float)
                    ) * 0.16
                else:
                    score -= np.linalg.norm(position - self.centroid) * 0.05
                if best is None or score > best[0]:
                    best = (score, position, yaw, fp)
        if best is None:
            return None
        return dict(pos=best[1], yaw=best[2], footprint=best[3])

    def place_dining_zone(self, position=None, yaw=None, compact=False):
        """Place one coordinated dining composition as a guaranteed group."""
        table_builder = self.furniture_builder(
            "dining_table", build_dining_table
        )
        chair_builder = self.furniture_builder(
            "dining_chair", build_chair
        )
        table_width = 1.50 if compact else 1.78
        table_depth = 0.84 if compact else 0.96
        zone_width = 2.60 if compact else 3.05
        zone_depth = 2.10 if compact else 2.46
        pose = None
        if position is not None and yaw is not None:
            fp = footprint_poly(position, yaw, zone_width, zone_depth)
            if self._ok(fp):
                pose = dict(
                    pos=np.asarray(position, dtype=float),
                    yaw=float(yaw),
                    footprint=fp,
                )
        if pose is None:
            pose = self.find_open_pose(
                zone_width,
                zone_depth,
                preferred=position,
                yaws=None if yaw is None else [yaw, yaw + math.pi / 2],
            )
        if pose is None:
            return None

        center = pose["pos"]
        yaw = pose["yaw"]
        local_x = np.array([math.cos(yaw), math.sin(yaw)])
        local_y = np.array([-math.sin(yaw), math.cos(yaw)])
        self.place_rug(
            center,
            yaw,
            zone_width - 0.08,
            zone_depth - 0.08,
        )
        self.add(
            table_builder(
                self.P,
                w=table_width,
                d=table_depth,
            ),
            center,
            yaw,
            block=False,
            avoid_doors=False,
            check=False,
        )
        side_offset = table_depth / 2 + 0.39
        longitudinal = table_width * (0.25 if compact else 0.27)
        for side in (-1, 1):
            for offset in (-longitudinal, longitudinal):
                chair_pos = (
                    center
                    + local_y * side * side_offset
                    + local_x * offset
                )
                self.add(
                    chair_builder(self.P),
                    chair_pos,
                    yaw_facing(-local_y * side),
                    block=False,
                    avoid_doors=False,
                    check=False,
                )
        if not compact:
            end_offset = table_width / 2 + 0.38
            for side in (-1, 1):
                chair_pos = center + local_x * side * end_offset
                self.add(
                    chair_builder(self.P),
                    chair_pos,
                    yaw_facing(-local_x * side),
                    block=False,
                    avoid_doors=False,
                    check=False,
                )
        self.placed.append(pose["footprint"].buffer(0.06))
        self.pendant(center)
        return dict(pos=center, yaw=yaw, footprint=pose["footprint"])

    def bedroom_anchor_slots(self):
        """Rank centered bed walls as a balanced bed/nightstand composition."""
        slots = self.wall_slots()

        def score(slot):
            bed_pos, bed_yaw = self._centered_wall_pose(slot, 1.8, 2.15)
            bed_fp = footprint_poly(bed_pos, bed_yaw, 1.8, 2.15)
            if slot["len"] < 1.9 or not self._candidate_clear(bed_fp):
                return -1e6

            n, side = slot["n"], slot["dir"]
            score_value = min(slot["len"], 4.2)
            if not slot["edge"].get("openings"):
                score_value += 3.0

            wall_anchor = bed_pos - n * (2.15 / 2 + WALL_GAP)
            nightstand_fps = []
            for direction in (-1, 1):
                ns_pos = (
                    wall_anchor + side * direction * (1.8 / 2 + 0.30)
                    + n * (WALL_GAP + 0.42 / 2)
                )
                ns_fp = footprint_poly(ns_pos, bed_yaw, 0.48, 0.42)
                if self._candidate_clear(ns_fp):
                    score_value += 2.0
                    nightstand_fps.append(ns_fp)

            for other in slots:
                if other["edge"] is slot["edge"] or other["len"] < 1.45:
                    continue
                wardrobe_w = 1.8 if other["len"] >= 2.0 else 1.35
                wardrobe_pos, wardrobe_yaw = self._centered_wall_pose(
                    other, wardrobe_w, 0.62
                )
                wardrobe_fp = footprint_poly(
                    wardrobe_pos, wardrobe_yaw, wardrobe_w, 0.62
                )
                if (
                    self._candidate_clear(wardrobe_fp)
                    and not wardrobe_fp.intersects(bed_fp)
                    and not any(
                        wardrobe_fp.intersects(fp) for fp in nightstand_fps
                    )
                ):
                    score_value += 2.5
                    break
            return score_value

        return sorted(slots, key=score, reverse=True)

    def _ok(self, fp, block=True, avoid_doors=True):
        if not fp.within(self.inset):
            return False
        if avoid_doors and any(fp.intersects(z) for z in self.door_zones):
            return False
        if block and any(fp.intersects(p) for p in self.placed):
            return False
        return True

    def add(self, built, pos, yaw, block=True, avoid_doors=True, check=True):
        """Try to place a built (meshes, w, d) at pos/yaw. Returns True if placed."""
        meshes, w, d = built
        fp = footprint_poly(pos, yaw, w, d)
        if check and not self._ok(fp, block=block, avoid_doors=avoid_doors):
            return False
        self.meshes.extend(place_meshes(meshes, pos, yaw))
        asset_key = next(
            (
                self._editable_mesh_assets[id(mesh)][1]
                for mesh in meshes
                if (
                    id(mesh) in self._editable_mesh_assets
                    and self._editable_mesh_assets[id(mesh)][0] is mesh
                )
            ),
            None,
        )
        if asset_key:
            self.editable_objects.append(
                dict(
                    asset_key=asset_key,
                    meshes=list(meshes),
                    position=np.array([pos[0], pos[1]], dtype=float),
                    yaw=float(yaw),
                    width=float(w),
                    depth=float(d),
                )
            )
        if block:
            self.placed.append(fp.buffer(0.04))
        return True

    def against_wall(self, builder, slots=None, min_side=0.0, prefer=None,
                     block=True, avoid_doors=True, **kw):
        """Snap an item squarely to the center of the best free wall run."""
        slots = slots if slots is not None else self.wall_slots()
        if prefer:
            slots = sorted(slots, key=prefer)
        for s in slots:
            built = builder(self.P, **kw)
            _, w, d = built
            if s["len"] < max(w + 0.1, min_side):
                continue
            n = s["n"]
            pos = s["mid"] + n * (WALL_GAP + d / 2)
            yaw = yaw_facing(n)
            if self.add(built, pos, yaw, block=block, avoid_doors=avoid_doors):
                return dict(
                    slot=s, pos=pos, yaw=yaw, n=n, s=s["dir"], w=w, d=d
                )
        return None

    def in_corner(self, builder, **kw):
        corners = list(self.room_m)
        self.rng.shuffle(corners)
        for v in corners:
            v = np.array(v)
            to_c = self.centroid - v
            L = np.linalg.norm(to_c)
            if L < 0.8:
                continue
            pos = v + to_c / L * 0.55
            built = builder(self.P, **kw)
            if self.add(built, pos, 0.0):
                return True
        return False

    def pendant(self, pos=None):
        p = self.centroid if pos is None else pos
        self.meshes.extend(place_meshes(list(build_pendant(self.P)[0]), p, 0.0))

    def art_on(self, slot_info, w=1.25):
        """Place a real modeled wall object behind an anchored furnishing."""
        if not self.wants_wall_decor:
            return
        s = slot_info["slot"]
        pos = np.array(slot_info["pos"]) - slot_info["n"] * (slot_info["d"] / 2 + 0.01)
        room_type = str(self.config.get("room_type", "")).lower()
        if "office" in room_type:
            built = build_wall_clock(self.P, diameter=min(w, 0.62))
        else:
            built = build_art(self.P, w=w)
        self.add(built, pos, slot_info["yaw"], block=False, check=False)

    def mirror_on(self, slot_info, diameter=0.82):
        """Place a contrasting modeled mirror instead of repeating wall art."""
        if not self.wants_wall_decor:
            return
        pos = (
            np.array(slot_info["pos"])
            - slot_info["n"] * (slot_info["d"] / 2 + 0.01)
        )
        self.add(
            build_round_mirror(
                self.P,
                diameter=diameter,
                ornate=self.design_choices["style_key"] == "classic",
            ),
            pos,
            slot_info["yaw"],
            block=False,
            check=False,
        )

    def feature_on(self, slot_info, w=None):
        """Add a designed wall treatment behind the room's visual anchor."""
        if self.config.get("whole_room_design", False):
            self.art_on(slot_info, w=min(w or slot_info["w"], 1.35))
            return
        s = slot_info["slot"]
        width = min(w or slot_info["w"], max(0.8, s["len"] - 0.16))
        pos = (np.array(slot_info["pos"])
               - slot_info["n"] * (slot_info["d"] / 2 + 0.015))
        self.add(build_slat_feature(self.P, w=width), pos, slot_info["yaw"],
                 block=False, avoid_doors=False, check=False)

    # ---- room recipes ----
    def furnish_living(self):
        sofa_builder = self.furniture_builder("sofa", build_sofa)
        chair_builder = self.furniture_builder("armchair", build_armchair)
        table_builder = self.furniture_builder("coffee_table", build_coffee_table)
        tv_builder = self.furniture_builder("tv_unit", build_tv_unit)
        sofa_width = (
            2.60
            if self.poly.area >= 28.0
            else 2.35 if self.poly.area >= 18.0 else 2.10
        )
        anchor_slots = self.living_anchor_slots(sofa_width)
        sofa = (
            self.against_wall(
                sofa_builder,
                slots=anchor_slots,
                w=sofa_width,
                d=0.98,
            )
            or self.against_wall(
                sofa_builder, slots=anchor_slots, w=1.7, d=0.9
            )
        )
        living_pendant_position = self.centroid
        if sofa:
            n, s = sofa["n"], sofa["s"]
            rug_pos = np.array(sofa["pos"]) + n * 1.55
            living_pendant_position = rug_pos
            rug_width = min(3.25, sofa["w"] + 0.70)
            self.place_rug(rug_pos, sofa["yaw"], rug_width, 2.05)
            table_pos = np.array(sofa["pos"]) + n * 1.52
            self.add(table_builder(self.P, w=1.20, d=0.64),
                     table_pos, sofa["yaw"])
            # A professional conversation group uses paired chairs whenever
            # the room supports them, both angled toward the center.
            chair_sides = [-1, 1]
            if not self.airy:
                for side in chair_sides:
                    ch_pos = table_pos + s * side * 1.52 + n * 0.10
                    f = table_pos - ch_pos
                    f = f / (np.linalg.norm(f) + 1e-9)
                    placed = self.add(
                        chair_builder(self.P), ch_pos, yaw_facing(f)
                    )
                    if not placed:
                        # Concave open-plan rooms can clip the perfectly
                        # symmetrical position by only a few centimetres.
                        # Slide that chair toward the open side while keeping
                        # it aimed at the coffee table.
                        for lateral, forward in (
                            (1.36, 0.42),
                            (1.22, 0.58),
                            (1.48, 0.62),
                        ):
                            ch_pos = (
                                table_pos
                                + s * side * lateral
                                + n * forward
                            )
                            f = table_pos - ch_pos
                            f = f / (np.linalg.norm(f) + 1e-9)
                            if self.add(
                                chair_builder(self.P),
                                ch_pos,
                                yaw_facing(f),
                            ):
                                break

            # Styled side tables frame the sofa and visually connect the
            # seating family. Keep one side open in small/airy rooms.
            table_sides = (-1, 1) if self.layered or self.poly.area >= 22 else (1,)
            placed_side_tables = []
            for side in table_sides:
                side_pos = (
                    np.array(sofa["pos"])
                    + s * side * (sofa["w"] / 2 + 0.31)
                    + n * 0.08
                )
                if self.add(build_side_table(self.P), side_pos, sofa["yaw"]):
                    placed_side_tables.append(side)

            # An arched reading light completes the composition without
            # repeating the lamp on every side.
            lamp_side = (
                -placed_side_tables[0] if placed_side_tables else -1
            )
            lamp_pos = (
                np.array(sofa["pos"])
                + s * lamp_side * (sofa["w"] / 2 + 0.50)
                + n * 0.34
            )
            self.add(build_floor_lamp(self.P), lamp_pos, sofa["yaw"])
            if self.airy:
                self.art_on(sofa)
            else:
                self.feature_on(sofa, w=min(2.4, sofa["w"] + 0.2))
            # TV on the wall facing the sofa
            tv_slots = [sl for sl in self.wall_slots()
                        if np.dot(sl["n"], n) < -0.6]
            if "no tv" not in self.brief and "without tv" not in self.brief:
                tv = self.against_wall(tv_builder, slots=tv_slots)
                if not tv:
                    self.against_wall(tv_builder, slots=tv_slots, w=1.25)
            if self.layered:
                ottoman_pos = np.array(sofa["pos"]) + n * 1.48 - s * 1.35
                self.add(build_ottoman(self.P), ottoman_pos, sofa["yaw"])

        # Large living polygons in real plans are usually open living/dining
        # rooms. Treating them as one sofa vignette left most of the apartment
        # empty, so automatically zone the largest remaining clear area.
        dining_zone = None
        if self.poly.area >= 30.0 and not any(
            word in self.brief
            for word in ("no dining", "without dining", "living only")
        ):
            dining_zone = (
                self.place_dining_zone(compact=False)
                or self.place_dining_zone(compact=True)
            )
            if dining_zone:
                sideboard_builder = self.furniture_builder(
                    "sideboard", build_sideboard
                )
                sideboard = self.against_wall(sideboard_builder)
                if sideboard:
                    self.art_on(sideboard, w=1.05)
        if self.wants_plants:
            self.in_corner(build_plant)
        if not self.airy:
            # Use the same production furniture family as the media and
            # dining storage instead of a primitive hallway table.
            console = self.against_wall(
                self.furniture_builder("sideboard", build_sideboard),
                w=1.35,
                d=0.42,
            )
            if console and self.layered:
                self.mirror_on(console, diameter=0.78)
        # The dining zone has its own pendant. Keep a second light centered on
        # the conversation area instead of one arbitrary fixture in the room.
        self.pendant(living_pendant_position)

    def furnish_bedroom(self):
        bed_builder = self.furniture_builder("bed", build_bed)
        nightstand_builder = self.furniture_builder("nightstand", build_nightstand)
        wardrobe_builder = self.furniture_builder("wardrobe", build_wardrobe)
        anchor_slots = self.bedroom_anchor_slots()
        bed = (
            self.against_wall(bed_builder, slots=anchor_slots)
            or self.against_wall(
                bed_builder, slots=anchor_slots, w=1.5, d=2.0
            )
        )
        if bed:
            n, s, yaw = bed["n"], bed["s"], bed["yaw"]
            anchor = np.array(bed["pos"]) - n * (bed["d"] / 2 + WALL_GAP)
            for side in (-1, 1):
                ns = nightstand_builder(self.P)
                ns_pos = (anchor + s * side * (bed["w"] / 2 + 0.30)
                          + n * (WALL_GAP + ns[2] / 2))
                self.add(ns, ns_pos, yaw)
            rug_pos = np.array(bed["pos"]) + n * (bed["d"] / 2 - 0.4)
            self.place_rug(rug_pos, yaw, bed["w"] + 1.2, 1.6)
            if self.airy:
                self.art_on(bed, w=1.1)
            else:
                self.feature_on(bed, w=min(2.2, bed["w"] + 0.35))
            if not self.airy:
                bench_pos = (
                    np.array(bed["pos"])
                    + n * (bed["d"] / 2 + 0.38)
                )
                self.add(build_bench(self.P), bench_pos, yaw)
        wardrobe = self.against_wall(wardrobe_builder, min_side=2.0)
        if not wardrobe:
            self.against_wall(
                wardrobe_builder, min_side=1.45, w=1.35
            )
        if self.wants_plants:
            self.in_corner(build_plant, tall=False)
        if self.layered:
            self.against_wall(
                build_round_mirror,
                block=False,
                ornate=self.design_choices["style_key"] == "classic",
            )
        if self.poly.area >= 18.0 and not self.airy:
            reading = self.against_wall(
                self.furniture_builder("armchair", build_armchair),
                min_side=1.05,
                w=0.86,
                d=0.82,
            )
            if reading:
                lamp_pos = (
                    np.asarray(reading["pos"])
                    + reading["s"] * (reading["w"] / 2 + 0.30)
                    + reading["n"] * 0.10
                )
                self.add(build_floor_lamp(self.P), lamp_pos, reading["yaw"])
        self.pendant()

    def furnish_kitchen(self):
        fridge_builder = self.furniture_builder("fridge", build_fridge)
        island_builder = self.furniture_builder("kitchen_island", build_island)
        slots = self.wall_slots()
        run = None
        for s in slots:
            L = min(s["len"] - 0.1, 3.4)
            if L < 1.4:
                continue
            built = build_kitchen_run(self.P, w=L)
            pos = s["mid"] + s["n"] * (WALL_GAP - 0.06 + built[2] / 2)
            if self.add(built, pos, yaw_facing(s["n"]), avoid_doors=True):
                run = dict(slot=s, pos=pos, n=s["n"], s=s["dir"], L=L)
                break
        if run:
            # fridge at either end of the counter run, else any free wall
            fr = fridge_builder(self.P)
            placed_fr = False
            for sgn in (1, -1):
                fr_pos = (run["slot"]["mid"]
                          + run["s"] * sgn * (run["L"] / 2 + fr[1] / 2 + 0.12)
                          + run["n"] * (WALL_GAP - 0.06 + fr[2] / 2))
                if self.add(fr, fr_pos, yaw_facing(run["n"])):
                    placed_fr = True
                    break
            if not placed_fr:
                # Search both ends of every other free wall. Center-only
                # placement often collided with the long cabinet run in a
                # compact galley kitchen and silently removed the fridge.
                for slot in slots:
                    if slot["edge"] is run["slot"]["edge"]:
                        continue
                    for fraction in (
                        slot["t0"] + min(0.15, 0.32 / slot["edge"]["length"]),
                        slot["t1"] - min(0.15, 0.32 / slot["edge"]["length"]),
                    ):
                        anchor = (
                            slot["p1"]
                            + (slot["p2"] - slot["p1"]) * fraction
                        )
                        position = (
                            anchor
                            + slot["n"] * (
                                WALL_GAP - 0.04 + fr[2] / 2
                            )
                        )
                        if self.add(
                            fridge_builder(self.P),
                            position,
                            yaw_facing(slot["n"]),
                        ):
                            placed_fr = True
                            break
                    if placed_fr:
                        break
            if not placed_fr:
                pose = self.find_open_pose(0.72, 0.70)
                if pose:
                    placed_fr = self.add(
                        fridge_builder(self.P),
                        pose["pos"],
                        pose["yaw"],
                    )
            # In a generously sized kitchen, add a shorter perpendicular run
            # to form an authored L-shaped work triangle.
            if self.poly.area >= 15.0:
                perpendicular = [
                    slot for slot in slots
                    if (
                        slot["edge"] is not run["slot"]["edge"]
                        and abs(np.dot(slot["dir"], run["s"])) < 0.30
                        and slot["len"] >= 1.55
                    )
                ]
                for slot in perpendicular:
                    length = min(2.15, slot["len"] - 0.12)
                    built = build_kitchen_run(self.P, w=length)
                    position = (
                        slot["mid"]
                        + slot["n"] * (
                            WALL_GAP - 0.06 + built[2] / 2
                        )
                    )
                    if self.add(
                        built,
                        position,
                        yaw_facing(slot["n"]),
                        avoid_doors=True,
                    ):
                        break
            # island facing the run if the room is deep enough
            isl_pos = None
            for cand in (run["slot"]["mid"] + run["n"] * 2.35, self.centroid):
                if self.add(island_builder(self.P), cand, yaw_facing(-run["n"])):
                    isl_pos = cand
                    break
            self.pendant(isl_pos)
        else:
            self.against_wall(fridge_builder)
            self.pendant()
        if self.wants_plants:
            self.in_corner(build_plant, tall=False)

    def furnish_dining(self):
        sideboard_builder = self.furniture_builder("sideboard", build_sideboard)
        slots = self.wall_slots()
        yaw = yaw_facing(slots[0]["n"]) if slots else 0.0
        self.place_dining_zone(
            position=self.centroid,
            yaw=yaw,
            compact=self.poly.area < 12.0,
        )
        sb = self.against_wall(sideboard_builder)
        if sb:
            self.art_on(sb, w=1.0)
        if self.wants_plants:
            self.in_corner(build_plant)

    def furnish_office(self):
        desk_builder = self.furniture_builder("desk", build_desk)
        chair_builder = self.furniture_builder("office_chair", build_office_chair)
        shelf_builder = self.furniture_builder("bookshelf", build_bookshelf)
        desk = self.against_wall(desk_builder)
        if desk:
            ch_pos = np.array(desk["pos"]) + desk["n"] * 0.72
            self.add(chair_builder(self.P), ch_pos, yaw_facing(-desk["n"]))
            self.art_on(desk, w=0.9)
        self.against_wall(shelf_builder, min_side=1.8)
        if self.wants_plants:
            self.in_corner(build_plant)
        self.pendant()

    def furnish_bathroom(self):
        vanity_builder = self.furniture_builder("vanity", build_vanity)
        toilet_builder = self.furniture_builder("toilet", build_toilet)
        shower_builder = self.furniture_builder("shower", build_shower)
        bathtub_builder = self.furniture_builder("bathtub", build_bathtub)

        # Reserve the wet zone first. Previously it was attempted after the
        # vanity and toilet, so an ordinary collision silently removed it.
        wet_fixture = None
        for size in (0.95, 0.78):
            built = shower_builder(self.P, w=size, d=size)
            _meshes, width, depth = built
            corner_offset = math.hypot(width / 2, depth / 2) + 0.08
            corners = sorted(
                (np.asarray(v, dtype=float) for v in self.room_m),
                key=lambda v: -np.linalg.norm(self.centroid - v),
            )
            for corner in corners:
                inward = self.centroid - corner
                distance = np.linalg.norm(inward)
                if distance < 1e-6:
                    continue
                inward /= distance
                pos = corner + inward * corner_offset
                for yaw in (
                    0.0,
                    math.pi / 2,
                    yaw_facing(inward),
                ):
                    if self.add(built, pos, yaw):
                        wet_fixture = "shower"
                        break
                if wet_fixture:
                    break
            if wet_fixture:
                break

        # Long narrow bathrooms often fit a bath more naturally than a square
        # shower. Try both full and compact baths before using the final safe
        # compact-shower guarantee.
        if not wet_fixture:
            for width, depth in ((1.65, 0.78), (1.35, 0.70)):
                placed = self.against_wall(
                    bathtub_builder, w=width, d=depth
                )
                if placed:
                    wet_fixture = "bathtub"
                    break

        if not wet_fixture:
            compact = shower_builder(self.P, w=0.72, d=0.72)
            point = self.poly.representative_point()
            pos = np.array([point.x, point.y])
            if not self.add(
                compact, pos, 0.0, avoid_doors=False
            ):
                # A malformed or impossibly tiny bathroom still receives the
                # required fixture visibly instead of failing silently.
                self.add(
                    compact, pos, 0.0, avoid_doors=False, check=False
                )
            wet_fixture = "compact shower"
            print("[WALK] Bathroom used the compact shower fallback.")

        v = (
            self.against_wall(vanity_builder)
            or self.against_wall(vanity_builder, w=0.76, d=0.46)
        )
        if v:
            mirror_pos = (
                np.asarray(v["pos"])
                - v["n"] * (v["d"] / 2 + 0.01)
            )
            self.add(
                build_round_mirror(self.P, diameter=min(0.82, v["w"] * 0.82),
                                   z=1.16),
                mirror_pos,
                v["yaw"],
                block=False,
                check=False,
            )
        slots = self.wall_slots()
        if v:
            slots = [s for s in slots
                     if not np.allclose(s["mid"], v["slot"]["mid"])] or slots
        self.against_wall(toilet_builder, slots=slots) or self.against_wall(
            toilet_builder, slots=slots, w=0.38, d=0.58
        )
        self.against_wall(build_towel_rail, slots=slots, block=False)
        self.against_wall(build_bathroom_shelf, slots=slots, block=False)
        mat_position = (
            np.asarray(v["pos"]) + v["n"] * (v["d"] / 2 + 0.42)
            if v
            else self.centroid
        )
        self.place_rug(mat_position, v["yaw"] if v else 0.0, 0.85, 0.55)
        if self.wants_plants:
            self.in_corner(build_plant, tall=False)
        self.pendant()

    def furnish_generic(self):
        if self.wants_plants:
            self.in_corner(build_plant)
        self.against_wall(build_console_table)
        if self.wants_wall_decor:
            self.against_wall(build_round_mirror, block=False)
        self.pendant()

    def furnish(self, room_type):
        rt = (room_type or "").lower()
        try:
            if "living" in rt or "studio" in rt:
                self.furnish_living()
            elif "bed" in rt or "guest" in rt or "kids" in rt:
                self.furnish_bedroom()
            elif "kitchen" in rt:
                self.furnish_kitchen()
            elif "dining" in rt:
                self.furnish_dining()
            elif "office" in rt:
                self.furnish_office()
            elif "bath" in rt:
                self.furnish_bathroom()
            else:
                self.furnish_generic()
        except Exception as exc:            # never let one room kill the build
            print(f"[WALK] Furnishing failed for '{room_type}': {exc}")
        return self.meshes, self.placed


# ================= SCENE BUILD =================
def _room_light_specs(poly, config):
    """Return warm architectural light positions distributed within a room."""
    inner = poly.buffer(-0.62)
    if inner.is_empty:
        inner = poly
    profile = str((config or {}).get("design_profile", "Curated")).lower()
    area = float(poly.area)
    points = []
    minx, miny, maxx, maxy = inner.bounds
    horizontal = (maxx - minx) >= (maxy - miny)
    if area >= 30.0:
        fractions = (0.32, 0.68)
        for fraction in fractions:
            x = minx + (maxx - minx) * (
                fraction if horizontal else 0.50
            )
            y = miny + (maxy - miny) * (
                0.50 if horizontal else fraction
            )
            point = Point(float(x), float(y))
            if inner.contains(point):
                points.append(np.array([x, y], dtype=float))
    if not points:
        point = inner.representative_point()
        points.append(np.array([point.x, point.y], dtype=float))

    room_type = str((config or {}).get("room_type", "")).lower()
    neutral = any(
        word in room_type for word in ("kitchen", "bath", "office")
    )
    color = (
        np.array([1.0, 0.90, 0.78], dtype=np.float32)
        if neutral
        else np.array([1.0, 0.82, 0.66], dtype=np.float32)
    )
    intensity = 115000.0 if profile == "layered" else 90000.0
    return [
        dict(
            position=np.array([point[0], point[1], WALL_H - 0.28],
                              dtype=np.float32),
            color=color,
            intensity=intensity,
        )
        for point in points
    ]


def _configure_pbr_lighting(open3d_scene, lights):
    """Apply the same daylight-and-warm-fixture rig to live and QA scenes."""
    open3d_scene.enable_indirect_light(True)
    open3d_scene.set_indirect_light_intensity(25000.0)
    open3d_scene.set_sun_light(
        np.array([-0.35, 0.45, -0.82], dtype=np.float32),
        np.array([1.0, 0.94, 0.86], dtype=np.float32),
        42000.0,
    )
    open3d_scene.enable_sun_light(True)
    for index, light in enumerate(lights or []):
        try:
            open3d_scene.add_spot_light(
                f"interior_light_{index:03d}",
                light["color"],
                light["position"],
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                float(light["intensity"]),
                8.0,
                0.52,
                1.18,
                False,
            )
        except Exception as exc:
            print(f"[WALK] Room light {index + 1} unavailable: {exc}")


def build_scene(rooms_px, doors_px, windows_px, px_per_m=None, room_configs=None,
                furnished=True):
    """Full 3D scene from plan pixels.

    px_per_m=None auto-calibrates the scale from the plan itself (door
    lengths, room sizes) so the model comes out at real-world dimensions.

    Returns dict with: meshes, allowed (walkable shapely area), spawn (x, y),
    spawn_yaw, bounds.
    """
    if px_per_m is None:
        px_per_m = estimate_px_per_m(rooms_px, doors_px)
    # SCALE_BOOST enlarges the building shell (rooms/doors/windows) while
    # furniture and the walker stay real-size, so rooms feel more spacious.
    px_per_m = px_per_m / SCALE_BOOST
    rooms_m = [[px_to_m_real(p, px_per_m) for p in poly] for poly in rooms_px]

    # Normalize winding: interior on the RIGHT of directed edges, so wall
    # boxes (built on the left) go OUTWARD. This fills the band between
    # adjacent rooms (no see-through slits) and puts window glass/sky on
    # the correct side regardless of how the polygon was drawn/detected.
    oriented = []
    for room in rooms_m:
        poly = Polygon([(p[0], p[1]) for p in room])
        pts = list(room)
        if poly.exterior.is_ccw:
            pts = pts[::-1]
        oriented.append(pts)
    rooms_m = oriented
    doors_m = [(px_to_m_real(a, px_per_m), px_to_m_real(b, px_per_m))
               for a, b in doors_px]
    windows_m = [(px_to_m_real(w[0], px_per_m), px_to_m_real(w[1], px_per_m))
                 for w in windows_px]

    if room_configs is None:
        room_configs = [{} for _ in rooms_m]

    all_edges = [build_room_edges(r) for r in rooms_m]
    door_infos = assign_openings(all_edges, doors_m, windows_m)

    meshes = []
    furniture_fps = []
    furniture_objects = []
    room_polys = []
    room_lights = []

    for i, room in enumerate(rooms_m):
        cfg = room_configs[i] if i < len(room_configs) else {}
        style = cfg.get("style", "Modern")
        rtype = cfg.get("room_type", "Living Room")
        P = get_palette(style, cfg)

        poly = Polygon([(p[0], p[1]) for p in room])
        if not poly.is_valid:
            poly = poly.buffer(0)
        room_polys.append(poly)
        room_lights.extend(_room_light_specs(poly, cfg))

        # floor + ceiling in the room's style
        floor = _orient_horizontal_surface(
            floor_mesh(room, P["floor"]),
            upward=True,
        )
        selected_floor_material = floor_material(cfg, rtype, style)
        minx, miny, maxx, maxy = poly.bounds
        apply_archviz_material(
            floor,
            selected_floor_material,
            tint=P["floor"],
            tint_strength=(
                0.30
                if selected_floor_material == "bathroom_tile"
                else 0.16
            ),
            # Keep UVs inside one authored texture tile in the PBR renderer;
            # the source maps already contain a complete board/tile layout.
            repeat_m=max(maxx - minx, maxy - miny, 1.0),
            # The tile normal map can expose the room's two triangulation
            # faces under strong indirect light. Its albedo and calibrated
            # roughness retain the realistic surface without those facets.
            detail_maps=selected_floor_material != "bathroom_tile",
        )
        meshes.append(floor)
        ceil = _orient_horizontal_surface(
            floor_mesh(room, P.get("ceiling", CEILING_COLOR)),
            upward=False,
        )
        ceil.translate((0, 0, WALL_H))
        meshes.append(ceil)

        selected_wall_material = wall_material(cfg, rtype, style)
        meshes.extend(build_walls(
            all_edges[i],
            P["wall"],
            trim_color=_mix_color(P["wood_dark"], P["wall"], 0.52),
        ))
        meshes.extend(build_wall_finish_skins(
            room,
            all_edges[i],
            P["wall"],
            selected_wall_material,
        ))
        meshes.extend(build_room_trim(room, all_edges[i], P, cfg))
        if cfg.get("whole_room_design", True):
            meshes.extend(build_room_design_surfaces(
                room, all_edges[i], P, cfg
            ))

        if furnished:
            cfg = dict(cfg)
            cfg.setdefault("room_type", rtype)
            cfg.setdefault("style", style)
            furnisher = RoomFurnisher(room, all_edges[i], P, cfg)
            fm, fps = furnisher.furnish(rtype)
            meshes.extend(fm)
            furniture_fps.extend(fps)
            furniture_objects.extend(furnisher.editable_objects)

    # ---- building cap: hull ceiling + under-floor slab so hairline seams
    # between rooms show as dark shadow lines, never open sky ----
    hull = unary_union(room_polys).convex_hull.buffer(0.35)
    hull_pts = [(x, y) for x, y in list(hull.exterior.coords)[:-1]]
    cap = _orient_horizontal_surface(
        floor_mesh(hull_pts, CEILING_COLOR),
        upward=True,
    )
    cap.translate((0, 0, WALL_H + 0.02))
    meshes.append(cap)
    base = _orient_horizontal_surface(
        floor_mesh(hull_pts, [0.40, 0.35, 0.30]),
        upward=False,
    )
    base.translate((0, 0, -0.01))
    meshes.append(base)

    # ---- threshold floor strips under every door (rooms' floors don't
    # cover the wall band between two detected polygons) ----
    for a, b in door_infos:
        c = (a + b) / 2
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        ang = math.atan2(d[1], d[0])
        strip = _bx(max(L, MIN_DOOR_W) + 0.3, WALL_THICKNESS * 2 + 0.55, 0.012,
                    [0.62, 0.47, 0.32])
        strip.rotate(_rotz(ang), center=(0, 0, 0))
        strip.translate((c[0], c[1], 0))
        meshes.append(strip)

    # ---- walkable area: each room (inset off its walls) joined ONLY through
    # the door openings. Walls are solid — you cannot pass except at doors.
    # Furniture is NOT subtracted, so you can still reach every room, but the
    # door corridor is kept TIGHT (no wider than the opening, just deep enough
    # to bridge the two rooms) so you can't slip sideways through a wall.
    walk_parts = [p.buffer(-BODY_RADIUS) for p in room_polys]
    for a, b in door_infos:
        c = (a + b) / 2
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        d = d / L
        # width: the clear opening only (so it lines up with the door hole and
        # never spans the wall beside the jamb)
        half_w = max(L, MIN_DOOR_W) / 2 - 0.03
        # depth: just enough to span both rooms' insets + the wall band between
        # their polygons — NOT deep enough to walk far into a wall
        depth = BODY_RADIUS + WALL_THICKNESS + 0.22
        corridor = shp_box(-half_w, -depth, half_w, depth)
        ang = math.degrees(math.atan2(d[1], d[0]))
        corridor = affinity.translate(affinity.rotate(corridor, ang, origin=(0, 0)),
                                      c[0], c[1])
        walk_parts.append(corridor)
    allowed = unary_union([p for p in walk_parts if not p.is_empty])

    # ---- spawn point ----
    order = sorted(range(len(room_polys)), key=lambda i: -room_polys[i].area)
    spawn, spawn_yaw = None, 0.0
    allowed_prep = prep(allowed)
    for i in order:
        c = room_polys[i].centroid
        cands = [(c.x, c.y)]
        minx, miny, maxx, maxy = room_polys[i].bounds
        xs = np.linspace(minx, maxx, 8)[1:-1]
        ys = np.linspace(miny, maxy, 8)[1:-1]
        cands += [(x, y) for x in xs for y in ys]
        for x, y in cands:
            if allowed_prep.contains(Point(x, y)):
                spawn = np.array([x, y])
                break
        if spawn is not None:
            look = np.array([c.x, c.y]) - spawn
            if np.linalg.norm(look) > 0.3:
                spawn_yaw = math.atan2(look[1], look[0])
            break
    if spawn is None:
        spawn = np.array([room_polys[0].centroid.x, room_polys[0].centroid.y])

    # Keep authored normals and mapped materials intact. The realtime light rig
    # now shades both architectural finishes and catalog furniture; baking
    # colors here would flatten textures and double-darken the room.
    meshes = [
        mesh
        for mesh in meshes
        if mesh is not None
    ]

    bounds = unary_union(room_polys).bounds
    return dict(
        meshes=meshes,
        allowed=allowed,
        spawn=spawn,
        spawn_yaw=spawn_yaw,
        bounds=bounds,
        furniture_objects=furniture_objects,
        lights=room_lights,
    )


# ================= CAMERA =================
def _extrinsic(pos3, yaw, pitch):
    """World-to-camera 4x4 for an OpenCV-style pinhole camera."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    fwd = np.array([cp * cy, cp * sy, sp])
    right = np.array([sy, -cy, 0.0])
    down = np.array([sp * cy, sp * sy, -cp])
    R = np.stack([right, down, fwd])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ pos3
    return E


def _set_camera(ctr, pos2, yaw, pitch):
    params = ctr.convert_to_pinhole_camera_parameters()
    pos3 = np.array([pos2[0], pos2[1], EYE_HEIGHT])
    params.extrinsic = _extrinsic(pos3, yaw, pitch)
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)


def _apply_fov(ctr):
    """Widen the camera to CAMERA_FOV so interiors feel open, not zoomed-in."""
    try:
        cur = ctr.get_field_of_view()
        step = (CAMERA_FOV - cur) / 5.0     # Open3D changes FOV in 5° steps
        if abs(step) >= 0.1:
            ctr.change_field_of_view(step=step)
    except Exception:
        pass


# ================= WALKTHROUGH VIEWER =================
def _launch_legacy_walkthrough(rooms_px, doors_px, windows_px, px_per_m=None,
                               room_configs=None, furnished=True,
                               window_title=None, wall_pass=True):
    """Compatibility viewer used only if the modern PBR window is unavailable."""
    print("[WALK] Building 3D model from plan...")
    scene = build_scene(rooms_px, doors_px, windows_px, px_per_m=px_per_m,
                        room_configs=room_configs, furnished=furnished)
    print(f"[WALK] Scene ready: {len(scene['meshes'])} meshes.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_title or
                      "3D Interior Walkthrough — WASD move · drag look · G walls · H help · Q exit",
                      1400, 900)
    for m in scene["meshes"]:
        if m is not None:
            vis.add_geometry(m)

    furniture_objects = scene.get("furniture_objects", [])
    selection_box = None
    if furniture_objects:
        first = furniture_objects[0]
        min_bound = np.min(
            [np.asarray(mesh.get_min_bound()) for mesh in first["meshes"]],
            axis=0,
        ) - 0.035
        max_bound = np.max(
            [np.asarray(mesh.get_max_bound()) for mesh in first["meshes"]],
            axis=0,
        ) + 0.035
        selection_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        )
        selection_box.paint_uniform_color([1.0, 0.42, 0.05])
        vis.add_geometry(selection_box)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.70, 0.80, 0.90])   # soft daylight sky
    opt.light_on = True
    # Floors/ceilings must render from either side regardless of the drawing
    # direction of the room polygon.
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    _apply_fov(ctr)
    allowed = prep(scene["allowed"])

    spawn = np.array(scene["spawn"], dtype=float)
    spawn_yaw = float(scene["spawn_yaw"])
    state = dict(pos=spawn.copy(), yaw=spawn_yaw, pitch=0.0,
                 running=True, shot=False, wall_pass=bool(wall_pass),
                 show_help=True, selected_furniture=0)
    held = set()

    GLFW_RIGHT, GLFW_LEFT, GLFW_DOWN, GLFW_UP = 262, 263, 264, 265
    GLFW_LSHIFT, GLFW_RSHIFT = 340, 344
    keys = ([ord(c) for c in "WASD"] +
            [GLFW_RIGHT, GLFW_LEFT, GLFW_DOWN, GLFW_UP, GLFW_LSHIFT, GLFW_RSHIFT])

    def mk_action(code):
        def cb(_vis, action, _mods):
            if action in (1, 2):
                held.add(code)
            else:
                held.discard(code)
            return False
        return cb

    for k in keys:
        vis.register_key_action_callback(k, mk_action(k))

    def quit_cb(_vis):
        state["running"] = False
        return False

    def shot_cb(_vis):
        state["shot"] = True
        return False

    def reset_cb(_vis):
        state["pos"] = spawn.copy()
        state["yaw"] = spawn_yaw
        state["pitch"] = 0.0
        return False

    def ghost_cb(_vis):
        state["wall_pass"] = not state["wall_pass"]
        mode = "ON — walk through walls" if state["wall_pass"] else "OFF — use doorways"
        print(f"[WALK] Ghost mode {mode}")
        return False

    def help_cb(_vis):
        print_controls()
        return False

    def refresh_selection_box():
        if selection_box is None or not furniture_objects:
            return
        selected = furniture_objects[state["selected_furniture"]]
        min_bound = np.min(
            [np.asarray(mesh.get_min_bound()) for mesh in selected["meshes"]],
            axis=0,
        ) - 0.035
        max_bound = np.max(
            [np.asarray(mesh.get_max_bound()) for mesh in selected["meshes"]],
            axis=0,
        ) + 0.035
        updated = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        )
        selection_box.points = updated.points
        selection_box.lines = updated.lines
        selection_box.paint_uniform_color([1.0, 0.42, 0.05])
        vis.update_geometry(selection_box)

    def select_furniture_cb(_vis):
        if not furniture_objects:
            print("[WALK] No editable catalog furniture in this scene.")
            return False
        state["selected_furniture"] = (
            state["selected_furniture"] + 1
        ) % len(furniture_objects)
        selected = furniture_objects[state["selected_furniture"]]
        refresh_selection_box()
        label = selected["asset_key"].replace("_", " ").title()
        print(
            f"[WALK] Selected {label} "
            f"({state['selected_furniture'] + 1}/{len(furniture_objects)})"
        )
        return False

    def rotate_furniture(delta):
        def callback(_vis):
            if not furniture_objects:
                return False
            selected = furniture_objects[state["selected_furniture"]]
            rotate_furniture_object(selected, delta)
            for mesh in selected["meshes"]:
                vis.update_geometry(mesh)
            refresh_selection_box()
            label = selected["asset_key"].replace("_", " ").title()
            print(
                f"[WALK] Rotated {label} to "
                f"{math.degrees(selected['yaw']) % 360:.0f} degrees"
            )
            return False
        return callback

    vis.register_key_callback(ord("Q"), quit_cb)
    vis.register_key_callback(27, quit_cb)          # ESC
    vis.register_key_callback(32, shot_cb)          # SPACE
    vis.register_key_callback(ord("R"), reset_cb)   # reset view
    vis.register_key_callback(ord("G"), ghost_cb)   # pass through walls
    vis.register_key_callback(ord("H"), help_cb)    # repeat help
    vis.register_key_callback(258, select_furniture_cb)  # TAB
    vis.register_key_callback(ord("["), rotate_furniture(math.radians(15)))
    vis.register_key_callback(ord("]"), rotate_furniture(math.radians(-15)))

    # ---- mouse-look: hold left button and drag to turn/look ----
    mouse = dict(down=False, x=0.0, y=0.0)

    def on_mouse_button(_vis, button, action, _mods):
        if button == 0:                             # left button
            mouse["down"] = (action == 1)
        return False

    def on_mouse_move(_vis, x, y):
        if mouse["down"]:
            state["yaw"] -= (x - mouse["x"]) * MOUSE_SENS
            state["pitch"] = max(-1.2, min(1.2,
                                 state["pitch"] - (y - mouse["y"]) * MOUSE_SENS))
        mouse["x"], mouse["y"] = x, y
        return False

    try:                                            # available in Open3D ≥0.16
        vis.register_mouse_button_callback(on_mouse_button)
        vis.register_mouse_move_callback(on_mouse_move)
        mouse_ok = True
    except Exception:
        mouse_ok = False

    minx, miny, maxx, maxy = scene["bounds"]
    ghost_bounds = prep(shp_box(minx - GHOST_MARGIN, miny - GHOST_MARGIN,
                                maxx + GHOST_MARGIN, maxy + GHOST_MARGIN))

    def try_move(nx, ny):
        p = state["pos"]
        if state["wall_pass"]:
            if ghost_bounds.contains(Point(nx, ny)):
                state["pos"] = np.array([nx, ny])
            return
        if allowed.contains(Point(nx, ny)):
            state["pos"] = np.array([nx, ny])
        elif allowed.contains(Point(nx, p[1])):     # slide along walls
            state["pos"] = np.array([nx, p[1]])
        elif allowed.contains(Point(p[0], ny)):
            state["pos"] = np.array([p[0], ny])

    def print_controls():
        print("\n" + "=" * 64)
        print("AI INTERIOR WALKTHROUGH")
        print("  W A S D      move through the design")
        print("  Mouse drag   look naturally" + ("" if mouse_ok else "  (use arrows)"))
        print("  Arrow keys   look around")
        print("  Shift        move faster")
        print("  G            toggle Ghost mode (walk through walls)")
        print("  TAB          select the next real 3D furniture component")
        print("  [ / ]        rotate selected furniture by 15 degrees")
        print("  R            return to the starting view")
        print("  SPACE        save a design snapshot")
        print("  H            show these controls again")
        print("  Q / ESC      return to the floor-plan studio")
        initial = "ON — walls are passable" if state["wall_pass"] else "OFF — doorways only"
        print(f"  Ghost mode   {initial}")
        if furniture_objects:
            selected = furniture_objects[state["selected_furniture"]]
            print(
                "  Selected     "
                + selected["asset_key"].replace("_", " ").title()
            )
        print("=" * 64 + "\n")

    print_controls()

    _set_camera(ctr, state["pos"], state["yaw"], state["pitch"])
    last = time.perf_counter()

    while state["running"]:
        if not vis.poll_events():
            break
        now = time.perf_counter()
        dt = min(now - last, 0.05)
        last = now

        yaw, pitch = state["yaw"], state["pitch"]
        f = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-f[1], f[0]])
        running = (GLFW_LSHIFT in held) or (GLFW_RSHIFT in held)
        step = WALK_SPEED * (RUN_MULT if running else 1.0) * dt
        p = state["pos"]
        if ord("W") in held:
            try_move(p[0] + f[0] * step, p[1] + f[1] * step)
        p = state["pos"]
        if ord("S") in held:
            try_move(p[0] - f[0] * step, p[1] - f[1] * step)
        p = state["pos"]
        if ord("A") in held:
            try_move(p[0] + left[0] * step, p[1] + left[1] * step)
        p = state["pos"]
        if ord("D") in held:
            try_move(p[0] - left[0] * step, p[1] - left[1] * step)

        if GLFW_LEFT in held:
            state["yaw"] += TURN_SPEED * dt
        if GLFW_RIGHT in held:
            state["yaw"] -= TURN_SPEED * dt
        if GLFW_UP in held:
            state["pitch"] = min(1.15, state["pitch"] + TURN_SPEED * 0.7 * dt)
        if GLFW_DOWN in held:
            state["pitch"] = max(-1.15, state["pitch"] - TURN_SPEED * 0.7 * dt)

        _set_camera(ctr, state["pos"], state["yaw"], state["pitch"])
        vis.update_renderer()

        if state["shot"]:
            state["shot"] = False
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            fname = os.path.join(
                CAPTURE_DIR, time.strftime("walk_%Y%m%d_%H%M%S.png"))
            buf = vis.capture_screen_float_buffer(do_render=True)
            img = (np.asarray(buf) * 255).astype(np.uint8)
            try:
                from PIL import Image
                Image.fromarray(img).save(fname)
                print(f"[WALK] Screenshot saved: {fname}")
            except Exception as exc:
                print(f"[WALK] Screenshot failed: {exc}")

    vis.destroy_window()


def _run_pbr_walkthrough(scene, window_title=None, wall_pass=True):
    """Run a first-person, physically based walkthrough with visible controls."""
    from open3d.visualization import gui, rendering

    app = gui.Application.instance
    try:
        app.initialize()
    except RuntimeError as exc:
        if "initialized" not in str(exc).lower():
            raise

    title = (
        window_title
        or "AI 3D Interior Walkthrough - PBR materials and editable furniture"
    )
    window = app.create_window(title, 1400, 900)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background(
        np.array([0.72, 0.80, 0.88, 1.0], dtype=np.float32)
    )
    scene_widget.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        np.array([0.35, -0.55, -0.76], dtype=np.float32),
    )
    _configure_pbr_lighting(
        scene_widget.scene.scene,
        scene.get("lights", []),
    )
    scene_widget.scene.show_skybox(False)

    geometry_names = {}
    geometry_materials = {}
    for index, mesh in enumerate(scene["meshes"]):
        name = f"room_mesh_{index:04d}"
        material = material_record_for_mesh(mesh)
        scene_widget.scene.add_geometry(name, mesh, material, False)
        geometry_names[id(mesh)] = name
        geometry_materials[id(mesh)] = material

    furniture_objects = scene.get("furniture_objects", [])
    selection_name = "selected_furniture_outline"
    selection_box = {"mesh": None}

    def selected_label_text():
        if not furniture_objects:
            return "Furniture: no editable catalog object"
        selected = furniture_objects[state["selected_furniture"]]
        label = selected["asset_key"].replace("_", " ").title()
        return (
            f"Furniture: {label} "
            f"({state['selected_furniture'] + 1}/{len(furniture_objects)})"
        )

    def make_selection_box():
        if not furniture_objects:
            return None
        selected = furniture_objects[state["selected_furniture"]]
        min_bound = np.min(
            [np.asarray(mesh.get_min_bound()) for mesh in selected["meshes"]],
            axis=0,
        ) - 0.035
        max_bound = np.max(
            [np.asarray(mesh.get_max_bound()) for mesh in selected["meshes"]],
            axis=0,
        ) + 0.035
        line = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        )
        line.paint_uniform_color([1.0, 0.42, 0.05])
        return line

    spawn = np.array(scene["spawn"], dtype=float)
    spawn_yaw = float(scene["spawn_yaw"])
    state = dict(
        pos=spawn.copy(),
        yaw=spawn_yaw,
        pitch=0.0,
        wall_pass=bool(wall_pass),
        selected_furniture=0,
    )
    allowed = prep(scene["allowed"])
    minx, miny, maxx, maxy = scene["bounds"]
    ghost_bounds = prep(shp_box(
        minx - GHOST_MARGIN,
        miny - GHOST_MARGIN,
        maxx + GHOST_MARGIN,
        maxy + GHOST_MARGIN,
    ))

    held = set()
    mouse = dict(down=False, x=0.0, y=0.0)
    last_tick = {"time": time.perf_counter()}

    em = window.theme.font_size
    panel = gui.Vert(
        0.45 * em,
        gui.Margins(0.75 * em, 0.65 * em, 0.75 * em, 0.75 * em),
    )
    panel.background_color = gui.Color(0.035, 0.045, 0.065, 0.91)
    heading = gui.Label("AI 3D Interior")
    instructions = gui.Label(
        "WASD move  |  Shift run\n"
        "Drag / arrows look  |  G pass walls\n"
        "Tab select  |  [ ] rotate  |  Space photo"
    )
    status_label = gui.Label("")
    selected_label = gui.Label("")
    panel.add_child(heading)
    panel.add_child(instructions)
    panel.add_child(status_label)
    panel.add_child(selected_label)

    ghost_toggle = gui.ToggleSwitch("Walk through walls")
    ghost_toggle.is_on = state["wall_pass"]
    panel.add_child(ghost_toggle)

    furniture_row = gui.Horiz(0.35 * em)
    next_button = gui.Button("Next furniture")
    left_button = gui.Button("Rotate left")
    right_button = gui.Button("Rotate right")
    furniture_row.add_child(next_button)
    furniture_row.add_child(left_button)
    furniture_row.add_child(right_button)
    panel.add_child(furniture_row)

    view_row = gui.Horiz(0.35 * em)
    reset_button = gui.Button("Reset view")
    snapshot_button = gui.Button("Save photo")
    close_button = gui.Button("Close")
    view_row.add_child(reset_button)
    view_row.add_child(snapshot_button)
    view_row.add_child(close_button)
    panel.add_child(view_row)

    def update_labels(message=None):
        mode = "ON" if state["wall_pass"] else "OFF"
        status_label.text = message or f"Walk through walls: {mode}"
        selected_label.text = selected_label_text()

    def apply_camera():
        cp = math.cos(state["pitch"])
        forward = np.array([
            cp * math.cos(state["yaw"]),
            cp * math.sin(state["yaw"]),
            math.sin(state["pitch"]),
        ])
        eye = np.array([
            state["pos"][0],
            state["pos"][1],
            EYE_HEIGHT,
        ], dtype=np.float32)
        center = (eye + forward).astype(np.float32)
        scene_widget.scene.camera.look_at(
            center, eye, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

    def try_move(nx, ny):
        position = state["pos"]
        if state["wall_pass"]:
            if ghost_bounds.contains(Point(nx, ny)):
                state["pos"] = np.array([nx, ny])
            return
        if allowed.contains(Point(nx, ny)):
            state["pos"] = np.array([nx, ny])
        elif allowed.contains(Point(nx, position[1])):
            state["pos"] = np.array([nx, position[1]])
        elif allowed.contains(Point(position[0], ny)):
            state["pos"] = np.array([position[0], ny])

    line_material = rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.base_color = [1.0, 0.42, 0.05, 1.0]
    line_material.line_width = 3.0

    def refresh_selection_box():
        if scene_widget.scene.has_geometry(selection_name):
            scene_widget.scene.remove_geometry(selection_name)
        selection_box["mesh"] = make_selection_box()
        if selection_box["mesh"] is not None:
            scene_widget.scene.add_geometry(
                selection_name, selection_box["mesh"], line_material, False
            )
        update_labels()

    def select_next():
        if furniture_objects:
            state["selected_furniture"] = (
                state["selected_furniture"] + 1
            ) % len(furniture_objects)
            refresh_selection_box()

    def rotate_selected(delta):
        if not furniture_objects:
            return
        selected = furniture_objects[state["selected_furniture"]]
        rotate_furniture_object(selected, delta)
        for mesh in selected["meshes"]:
            name = geometry_names[id(mesh)]
            scene_widget.scene.remove_geometry(name)
            material = geometry_materials[id(mesh)]
            scene_widget.scene.add_geometry(name, mesh, material, False)
        refresh_selection_box()
        label = selected["asset_key"].replace("_", " ").title()
        update_labels(
            f"{label}: {math.degrees(selected['yaw']) % 360:.0f} degrees"
        )

    def reset_view():
        state["pos"] = spawn.copy()
        state["yaw"] = spawn_yaw
        state["pitch"] = 0.0
        apply_camera()
        update_labels("View reset")

    def save_snapshot():
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        filename = os.path.join(
            CAPTURE_DIR, time.strftime("walk_pbr_%Y%m%d_%H%M%S.png")
        )
        update_labels("Saving high-quality photo...")

        def write_image(image):
            o3d.io.write_image(filename, image, 9)
            update_labels(f"Saved: {os.path.basename(filename)}")
            print(f"[WALK] PBR screenshot saved: {filename}")

        scene_widget.scene.scene.render_to_image(write_image)

    def set_ghost(enabled):
        state["wall_pass"] = bool(enabled)
        update_labels()

    def on_key(event):
        key = event.key
        if event.type == gui.KeyEvent.Type.UP:
            held.discard(key)
            return gui.Widget.EventCallbackResult.HANDLED

        held.add(key)
        if key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            window.close()
        elif key == gui.KeyName.G:
            ghost_toggle.is_on = not state["wall_pass"]
            set_ghost(ghost_toggle.is_on)
        elif key == gui.KeyName.TAB:
            select_next()
        elif key == gui.KeyName.LEFT_BRACKET:
            rotate_selected(math.radians(15))
        elif key == gui.KeyName.RIGHT_BRACKET:
            rotate_selected(math.radians(-15))
        elif key == gui.KeyName.R:
            reset_view()
        elif key == gui.KeyName.SPACE:
            save_snapshot()
        elif key == gui.KeyName.H:
            panel.visible = not panel.visible
        return gui.Widget.EventCallbackResult.HANDLED

    def on_mouse(event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            mouse["down"] = event.is_button_down(gui.MouseButton.LEFT)
            mouse["x"], mouse["y"] = event.x, event.y
            if mouse["down"]:
                return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.DRAG and mouse["down"]:
            state["yaw"] -= (event.x - mouse["x"]) * MOUSE_SENS
            state["pitch"] = max(
                -1.2,
                min(
                    1.2,
                    state["pitch"] - (event.y - mouse["y"]) * MOUSE_SENS,
                ),
            )
            mouse["x"], mouse["y"] = event.x, event.y
            apply_camera()
            return gui.Widget.EventCallbackResult.CONSUMED
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            mouse["down"] = False
        return gui.Widget.EventCallbackResult.IGNORED

    movement_keys = {
        gui.KeyName.W,
        gui.KeyName.A,
        gui.KeyName.S,
        gui.KeyName.D,
        gui.KeyName.LEFT,
        gui.KeyName.RIGHT,
        gui.KeyName.UP,
        gui.KeyName.DOWN,
    }

    def on_tick():
        now = time.perf_counter()
        dt = min(now - last_tick["time"], 0.05)
        last_tick["time"] = now
        active = bool(held.intersection(movement_keys))
        yaw = state["yaw"]
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        running = (
            gui.KeyName.LEFT_SHIFT in held
            or gui.KeyName.RIGHT_SHIFT in held
        )
        step = WALK_SPEED * (RUN_MULT if running else 1.0) * dt
        position = state["pos"]
        if gui.KeyName.W in held:
            try_move(
                position[0] + forward[0] * step,
                position[1] + forward[1] * step,
            )
        position = state["pos"]
        if gui.KeyName.S in held:
            try_move(
                position[0] - forward[0] * step,
                position[1] - forward[1] * step,
            )
        position = state["pos"]
        if gui.KeyName.A in held:
            try_move(
                position[0] + left[0] * step,
                position[1] + left[1] * step,
            )
        position = state["pos"]
        if gui.KeyName.D in held:
            try_move(
                position[0] - left[0] * step,
                position[1] - left[1] * step,
            )
        if gui.KeyName.LEFT in held:
            state["yaw"] += TURN_SPEED * dt
        if gui.KeyName.RIGHT in held:
            state["yaw"] -= TURN_SPEED * dt
        if gui.KeyName.UP in held:
            state["pitch"] = min(
                1.15, state["pitch"] + TURN_SPEED * 0.7 * dt
            )
        if gui.KeyName.DOWN in held:
            state["pitch"] = max(
                -1.15, state["pitch"] - TURN_SPEED * 0.7 * dt
            )
        if active:
            apply_camera()
        return active

    def on_layout(_context):
        rect = window.content_rect
        scene_widget.frame = rect
        panel.frame = gui.Rect(
            rect.x + int(0.7 * em),
            rect.y + int(0.7 * em),
            int(35 * em),
            int(12.5 * em),
        )

    ghost_toggle.set_on_clicked(set_ghost)
    next_button.set_on_clicked(select_next)
    left_button.set_on_clicked(lambda: rotate_selected(math.radians(15)))
    right_button.set_on_clicked(lambda: rotate_selected(math.radians(-15)))
    reset_button.set_on_clicked(reset_view)
    snapshot_button.set_on_clicked(save_snapshot)
    close_button.set_on_clicked(window.close)
    scene_widget.set_on_key(on_key)
    scene_widget.set_on_mouse(on_mouse)
    window.set_on_tick_event(on_tick)
    window.set_on_layout(on_layout)

    window.add_child(scene_widget)
    window.add_child(panel)
    scene_widget.set_view_controls(gui.SceneWidget.Controls.FLY)
    bounds = o3d.geometry.AxisAlignedBoundingBox()
    for mesh in scene["meshes"]:
        bounds += mesh.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(
        CAMERA_FOV,
        bounds,
        np.array([state["pos"][0], state["pos"][1], EYE_HEIGHT]),
    )
    apply_camera()
    refresh_selection_box()
    update_labels()
    print("[WALK] PBR walkthrough ready. Controls are shown inside the window.")
    app.run()


def launch_walkthrough(rooms_px, doors_px, windows_px, px_per_m=None,
                       room_configs=None, furnished=True, window_title=None,
                       wall_pass=True):
    """Build the furnished design and open the high-quality PBR walkthrough."""
    print("[WALK] Building textured 3D model from plan...")
    scene = build_scene(
        rooms_px,
        doors_px,
        windows_px,
        px_per_m=px_per_m,
        room_configs=room_configs,
        furnished=furnished,
    )
    print(f"[WALK] PBR scene ready: {len(scene['meshes'])} meshes.")
    try:
        return _run_pbr_walkthrough(
            scene,
            window_title=window_title,
            wall_pass=wall_pass,
        )
    except Exception as exc:
        print(f"[WALK] PBR viewer unavailable ({exc}); opening compatibility view.")
        return _launch_legacy_walkthrough(
            rooms_px,
            doors_px,
            windows_px,
            px_per_m=px_per_m,
            room_configs=room_configs,
            furnished=furnished,
            window_title=window_title,
            wall_pass=wall_pass,
        )


# ================= STANDALONE DEMO / VERIFICATION =================
def _demo_plan():
    """Synthetic 3-room plan in plan pixels (100 px = 1 m).

    Living room 5.2 x 4.2 m, bedroom 3.8 x 4.2 m to its right sharing a wall
    with a door, kitchen 5.2 x 3.0 m below the living room with a door.
    Windows on the outer walls.
    """
    living = [(100, 100), (620, 100), (620, 520), (100, 520)]
    bedroom = [(620, 100), (1000, 100), (1000, 520), (620, 520)]
    kitchen = [(100, 520), (620, 520), (620, 820), (100, 820)]
    doors = [((620, 280), (620, 380)),      # living <-> bedroom
             ((300, 520), (400, 520))]      # living <-> kitchen
    windows = [((250, 100), (420, 100), "normal"),      # living, top wall
               ((760, 100), (900, 100), "normal"),      # bedroom, top wall
               ((1000, 250), (1000, 400), "normal"),    # bedroom, right wall
               ((250, 820), (430, 820), "normal")]      # kitchen, bottom wall
    configs = [
        dict(
            room_type="Living Room", style="Modern", use_catalog=True,
            whole_room_design=True, design_profile="Layered",
        ),
        dict(
            room_type="Bedroom", style="Scandinavian", use_catalog=True,
            whole_room_design=True, design_profile="Curated",
        ),
        dict(
            room_type="Kitchen", style="Japandi", use_catalog=True,
            whole_room_design=True, design_profile="Curated",
        ),
    ]
    return living, bedroom, kitchen, doors, windows, configs


def _capture_verification(out_dir):
    """Render screenshots of the demo scene (no interaction) for verification."""
    from PIL import Image

    living, bedroom, kitchen, doors, windows, configs = _demo_plan()
    scene = build_scene([living, bedroom, kitchen], doors, windows,
                        px_per_m=100, room_configs=configs)
    os.makedirs(out_dir, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window("capture", 1280, 800, visible=True)
    for m in scene["meshes"]:
        if m is not None:
            vis.add_geometry(m)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.62, 0.72, 0.82])
    opt.light_on = True
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()

    shots = [
        ("living_from_door", (3.6, -4.4), math.radians(160), -0.05),
        ("living_sofa_view", (4.7, -2.0), math.radians(205), -0.10),
        ("bedroom", (9.1, -2.0), math.radians(-90), -0.08),
        ("kitchen", (2.0, -7.9), math.radians(15), -0.05),
        ("doorway_living_to_bedroom", (5.2, -3.3), math.radians(10), 0.0),
        ("overview", (3.5, -12.5), math.radians(75), 0.55),
    ]
    for name, pos, yaw, pitch in shots:
        if name == "overview":
            params = ctr.convert_to_pinhole_camera_parameters()
            params.extrinsic = _extrinsic(
                np.array([pos[0], pos[1], 11.0]), yaw, -0.9)
            ctr.convert_from_pinhole_camera_parameters(params,
                                                       allow_arbitrary=True)
        else:
            _set_camera(ctr, np.array(pos), yaw, pitch)
        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(buf) * 255).astype(np.uint8)
        path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(img).save(path)
        print(f"[CAPTURE] {path}")
    vis.destroy_window()

    # sanity numbers
    print(f"[CAPTURE] meshes={len(scene['meshes'])}")
    print(f"[CAPTURE] spawn={scene['spawn']} yaw={scene['spawn_yaw']:.2f}")
    print(f"[CAPTURE] walkable_area={scene['allowed'].area:.2f} m^2")


def _capture_pbr_scene(scene, out_dir, shots, title="PBR walkthrough verification"):
    """Render named camera shots through the production PBR scene path."""
    from open3d.visualization import gui, rendering

    os.makedirs(out_dir, exist_ok=True)

    app = gui.Application.instance
    try:
        app.initialize()
    except RuntimeError as exc:
        if "initialized" not in str(exc).lower():
            raise
    window = app.create_window(title, 1280, 800)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    widget.scene.set_background(
        np.array([0.72, 0.80, 0.88, 1.0], dtype=np.float32)
    )
    widget.scene.set_lighting(
        rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
        np.array([0.35, -0.55, -0.76], dtype=np.float32),
    )
    _configure_pbr_lighting(
        widget.scene.scene,
        scene.get("lights", []),
    )
    for index, mesh in enumerate(scene["meshes"]):
        widget.scene.add_geometry(
            f"room_mesh_{index:04d}",
            mesh,
            material_record_for_mesh(mesh),
            False,
        )

    bounds = o3d.geometry.AxisAlignedBoundingBox()
    for mesh in scene["meshes"]:
        bounds += mesh.get_axis_aligned_bounding_box()
    widget.setup_camera(CAMERA_FOV, bounds, bounds.get_center())

    def set_shot_camera(index):
        _name, eye_values, yaw, pitch = shots[index]
        eye = np.asarray(eye_values, dtype=np.float32)
        cp = math.cos(pitch)
        forward = np.asarray(
            [
                cp * math.cos(yaw),
                cp * math.sin(yaw),
                math.sin(pitch),
            ],
            dtype=np.float32,
        )
        widget.scene.camera.look_at(
            eye + forward,
            eye,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    state = {"index": 0, "pending": False, "warmup": 0}

    def finish_capture(image):
        name = shots[state["index"]][0]
        path = os.path.join(out_dir, f"{name}.png")
        o3d.io.write_image(path, image, 9)
        print(f"[PBR CAPTURE] {path}")

        def advance():
            state["index"] += 1
            state["pending"] = False
            state["warmup"] = 0
            if state["index"] >= len(shots):
                window.close()
            else:
                set_shot_camera(state["index"])
                window.post_redraw()

        app.post_to_main_thread(window, advance)

    def on_tick():
        if state["pending"]:
            return False
        state["warmup"] += 1
        if state["warmup"] >= 3:
            state["pending"] = True
            widget.scene.scene.render_to_image(finish_capture)
        return True

    def on_layout(_context):
        widget.frame = window.content_rect

    window.add_child(widget)
    window.set_on_layout(on_layout)
    window.set_on_tick_event(on_tick)
    set_shot_camera(0)
    app.run()

    print(f"[PBR CAPTURE] meshes={len(scene['meshes'])}")
    print(f"[PBR CAPTURE] walkable_area={scene['allowed'].area:.2f} m^2")


def _capture_pbr_verification(out_dir):
    """Render the demo apartment through the same PBR path as the walkthrough."""
    living, bedroom, kitchen, doors, windows, configs = _demo_plan()
    scene = build_scene(
        [living, bedroom, kitchen],
        doors,
        windows,
        px_per_m=100,
        room_configs=configs,
    )
    shots = [
        ("living_from_door", (3.6, -4.4, EYE_HEIGHT), math.radians(160), -0.05),
        ("living_sofa_view", (4.7, -2.0, EYE_HEIGHT), math.radians(205), -0.10),
        ("bedroom", (9.1, -2.0, EYE_HEIGHT), math.radians(-90), -0.08),
        ("kitchen", (2.0, -7.9, EYE_HEIGHT), math.radians(15), -0.05),
        (
            "doorway_living_to_bedroom",
            (5.2, -3.3, EYE_HEIGHT),
            math.radians(10),
            0.0,
        ),
        ("overview", (3.5, -12.5, 11.0), math.radians(75), -0.90),
    ]
    _capture_pbr_scene(scene, out_dir, shots)


if __name__ == "__main__":
    import sys
    if "--capture" in sys.argv:
        i = sys.argv.index("--capture")
        out = sys.argv[i + 1] if len(sys.argv) > i + 1 else "walk_captures"
        _capture_pbr_verification(out)
    else:
        living, bedroom, kitchen, doors, windows, configs = _demo_plan()
        launch_walkthrough([living, bedroom, kitchen], doors, windows,
                           px_per_m=100, room_configs=configs)
