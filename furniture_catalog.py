"""Clean local 3D furniture catalog for the walkthrough.

Unlike image-to-3D reconstruction, these assets were authored as native game
geometry.  Every model has a stable local axis, pivot and measurable footprint,
so it can be positioned and rotated without carrying an image projection.

Source: Kenney Furniture Kit 2.0 (CC0)
https://kenney.nl/assets/furniture-kit
"""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
CATALOG_ROOT = ROOT / "assets" / "furniture_catalog"

MODEL_HEIGHTS = {
    "sofa": 0.90,
    "armchair": 0.86,
    "coffee_table": 0.44,
    "tv_unit": 0.58,
    "bed": 1.15,
    "nightstand": 0.58,
    "wardrobe": 2.20,
    "kitchen_island": 0.94,
    "fridge": 1.92,
    "dining_table": 0.77,
    "dining_chair": 0.92,
    "sideboard": 0.86,
    "desk": 0.78,
    "office_chair": 1.08,
    "bookshelf": 2.00,
    "vanity": 0.92,
    "toilet": 0.78,
    "shower": 2.05,
}

DEFAULT_MODELS = {
    "sofa": "loungeSofa.glb",
    "armchair": "loungeChair.glb",
    "coffee_table": "tableCoffee.glb",
    "tv_unit": "cabinetTelevisionDoors.glb",
    "bed": "bedDouble.glb",
    "nightstand": "cabinetBedDrawerTable.glb",
    "wardrobe": "bookcaseClosedWide.glb",
    "kitchen_island": "kitchenBar.glb",
    "fridge": "kitchenFridgeLarge.glb",
    "dining_table": "table.glb",
    "dining_chair": "chairCushion.glb",
    "sideboard": "cabinetTelevisionDoors.glb",
    "desk": "desk.glb",
    "office_chair": "chairDesk.glb",
    "bookshelf": "bookcaseOpen.glb",
    "vanity": "bathroomSinkSquare.glb",
    "toilet": "toilet.glb",
    "shower": "shower.glb",
}

MODERN_MODELS = {
    "sofa": "loungeDesignSofa.glb",
    "armchair": "loungeDesignChair.glb",
    "coffee_table": "tableCoffeeGlass.glb",
    "dining_chair": "chairModernFrameCushion.glb",
}

CLASSIC_MODELS = {
    "sofa": "loungeSofaLong.glb",
    "armchair": "loungeChairRelax.glb",
    "coffee_table": "tableCoffee.glb",
    "dining_chair": "chairCushion.glb",
}

MODERN_STYLE_WORDS = {
    "modern", "contemporary", "minimalist", "scandinavian", "japandi",
    "industrial", "mid-century",
}
CLASSIC_STYLE_WORDS = {"classic", "traditional", "bohemian", "boho"}

_MESH_CACHE = {}


def _load_trimesh_scene(path):
    """Import GLB components with the project's local pure-Python Trimesh."""
    try:
        trimesh = importlib.import_module("trimesh")
    except ModuleNotFoundError:
        local_packages = (
            ROOT / ".triposr_venv" / "Lib" / "site-packages"
        )
        if local_packages.is_dir() and str(local_packages) not in sys.path:
            sys.path.insert(0, str(local_packages))
        trimesh = importlib.import_module("trimesh")
    return trimesh.load(str(path), force="scene", process=False)


def _shade_materials(mesh):
    """Apply soft fixed lighting while keeping every authored material region."""
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors)
    if len(normals) == 0 or len(colors) != len(normals):
        return mesh

    key = np.array([0.35, 0.55, 0.75])
    key /= np.linalg.norm(key)
    fill = np.array([-0.60, -0.35, 0.45])
    fill /= np.linalg.norm(fill)
    shade = (
        0.68
        + 0.28 * np.clip(normals @ key, 0, None)
        + 0.12 * np.clip(normals @ fill, 0, None)
        + 0.08 * np.clip(normals[:, 2], 0, None)
    )
    vertex_colors = np.clip(colors * shade[:, None], 0, 1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors
    )
    return mesh


def _model_name(asset_key: str, style: str) -> str | None:
    style_key = (style or "").lower()
    if any(word in style_key for word in CLASSIC_STYLE_WORDS):
        return CLASSIC_MODELS.get(asset_key, DEFAULT_MODELS.get(asset_key))
    if any(word in style_key for word in MODERN_STYLE_WORDS):
        return MODERN_MODELS.get(asset_key, DEFAULT_MODELS.get(asset_key))
    return DEFAULT_MODELS.get(asset_key)


def _palette_material(asset_key, palette):
    if not palette:
        return None
    if asset_key in {"sofa", "armchair", "office_chair", "bed"}:
        return np.asarray(palette["sofa"], dtype=float)
    if asset_key in {
        "coffee_table", "tv_unit", "nightstand", "dining_table",
        "dining_chair", "sideboard", "desk", "bookshelf",
    }:
        return np.asarray(palette["wood"], dtype=float)
    if asset_key in {"wardrobe", "kitchen_island", "fridge", "vanity"}:
        return np.asarray(palette["cabinet"], dtype=float)
    if asset_key in {"toilet", "shower"}:
        return np.array([0.92, 0.94, 0.94])
    return None


def _coordinate_material(source_color, target_color):
    """Tint a component to the room palette while retaining material contrast."""
    if target_color is None:
        return source_color
    brightness = float(np.mean(source_color))
    if brightness > 0.88:
        # Keep linens, porcelain and glass recognizably light.
        weight = 0.18
    else:
        weight = 0.78
    tonal_target = np.clip(target_color * (0.72 + 0.48 * brightness), 0, 1)
    return np.clip(
        np.asarray(source_color) * (1.0 - weight) + tonal_target * weight,
        0,
        1,
    )


def catalog_status() -> tuple[bool, str]:
    """Return whether every required native model is installed."""
    required = set(DEFAULT_MODELS.values())
    required.update(MODERN_MODELS.values())
    required.update(CLASSIC_MODELS.values())
    missing = sorted(name for name in required if not (CATALOG_ROOT / name).is_file())
    if missing:
        return False, "Local 3D catalog is missing: " + ", ".join(missing)
    return True, "Local editable 3D furniture catalog is ready."


def load_catalog_asset(
    asset_key: str,
    style: str,
    width: float,
    depth: float,
    height: float | None = None,
    palette=None,
):
    """Load a clean GLB and normalize it to the layout footprint in meters.

    Kenney GLBs use the standard Y-up convention.  The walkthrough is Z-up and
    treats local +Y as the furniture front, so the fixed X-axis rotation also
    gives every model the same predictable facing direction.
    """
    model_name = _model_name(asset_key, style)
    if model_name is None:
        return None
    path = CATALOG_ROOT / model_name
    if not path.is_file():
        return None

    height = float(height or MODEL_HEIGHTS.get(asset_key, 1.0))
    material_target = _palette_material(asset_key, palette)
    material_key = (
        tuple(np.round(material_target, 3))
        if material_target is not None else ()
    )
    cache_key = (
        str(path), round(float(width), 3), round(float(depth), 3),
        round(height, 3), material_key,
    )
    if cache_key in _MESH_CACHE:
        return [copy.deepcopy(mesh) for mesh in _MESH_CACHE[cache_key]]

    scene = _load_trimesh_scene(path)
    source_components = list(scene.dump(concatenate=False))
    if not source_components:
        return None

    component_data = []
    for source in source_components:
        source_vertices = np.asarray(source.vertices, dtype=float)
        if len(source_vertices) == 0 or len(source.faces) == 0:
            continue
        # Native GLB: X width, Y up, Z front/back. Walkthrough: X width,
        # Z up, +Y front. The sign change makes the authored front face +Y.
        vertices = np.column_stack(
            (-source_vertices[:, 0], source_vertices[:, 2], source_vertices[:, 1])
        )
        color_visual = source.visual.to_color()
        main_color = np.asarray(
            getattr(color_visual, "vertex_colors", [184, 184, 184, 255]),
            dtype=float,
        ).reshape(-1)[:3]
        if main_color.max(initial=0.0) > 1.0:
            main_color /= 255.0
        main_color = _coordinate_material(main_color, material_target)
        component_data.append(
            (vertices, np.asarray(source.faces, dtype=np.int32), main_color)
        )
    if not component_data:
        return None

    all_vertices = np.vstack([vertices for vertices, _faces, _color in component_data])
    source_min = all_vertices.min(axis=0)
    source_max = all_vertices.max(axis=0)
    source_extents = source_max - source_min
    if np.any(source_extents < 1e-6):
        return None

    target_extents = np.array([float(width), float(depth), height])
    scale = target_extents / np.maximum(source_extents, 1e-6)
    center_xy = target_extents[:2] / 2
    meshes = []
    for vertices, faces, color in component_data:
        normalized = (vertices - source_min) * scale
        normalized[:, 0] -= center_xy[0]
        normalized[:, 1] -= center_xy[1]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(normalized)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.paint_uniform_color(np.clip(color, 0, 1))
        mesh.compute_vertex_normals()
        _shade_materials(mesh)
        meshes.append(mesh)

    _MESH_CACHE[cache_key] = [copy.deepcopy(mesh) for mesh in meshes]
    return meshes
