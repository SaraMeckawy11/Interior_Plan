"""Local professional 3D furniture and decoration catalog.

The preferred assets are textured, production-authored Poly Haven models.  A
small Kenney set remains as a compact fallback for kitchen and bathroom
fixtures that Poly Haven does not currently provide.  Every model is native 3D
geometry with a stable pivot and measurable footprint; no furniture image
planes or screenshot projections are used.

Sources:
https://polyhaven.com/ (CC0)
https://kenney.nl/assets/furniture-kit (CC0)
https://github.com/KhronosGroup/glTF-Sample-Assets (CC BY 4.0 asset noted below)
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
PRO_ROOT = CATALOG_ROOT / "pro"

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
    "bathtub": 0.62,
    "plant": 1.35,
    "wall_art": 0.85,
    "wall_mirror": 0.92,
    "wall_clock": 0.55,
    "wall_sconce": 0.46,
    "ceiling_light": 0.42,
    "decor_vase": 0.32,
    "throw_pillows": 0.32,
}

DEFAULT_MODELS = {
    "sofa": "pro/glam_velvet_sofa/GlamVelvetSofa.glb",
    "armchair": "pro/modern_arm_chair_01/modern_arm_chair_01_1k.gltf",
    "coffee_table": "pro/modern_coffee_table_01/modern_coffee_table_01_1k.gltf",
    "tv_unit": "pro/modern_wooden_cabinet/modern_wooden_cabinet_1k.gltf",
    "bed": "pro/GothicBed_01/GothicBed_01_1k.gltf",
    "nightstand": (
        "pro/painted_wooden_nightstand/"
        "painted_wooden_nightstand_1k.gltf"
    ),
    "wardrobe": "pro/drawer_cabinet/drawer_cabinet_1k.gltf",
    "kitchen_island": "kitchenBar.glb",
    "fridge": "kitchenFridgeLarge.glb",
    # The catalog dining table includes a permanently modeled plaid cloth
    # that cannot follow the user's palette. The walkthrough's clean PBR
    # timber table is true editable geometry and is used instead.
    "dining_table": None,
    "dining_chair": "pro/sheen_chair/SheenChair.glb",
    "sideboard": "pro/modern_wooden_cabinet/modern_wooden_cabinet_1k.gltf",
    "desk": "pro/metal_office_desk/metal_office_desk_1k.gltf",
    "office_chair": (
        "pro/modern_arm_chair_01/modern_arm_chair_01_1k.gltf"
    ),
    "bookshelf": (
        "pro/wooden_display_shelves_01/"
        "wooden_display_shelves_01_1k.gltf"
    ),
    "vanity": "bathroomSinkSquare.glb",
    "toilet": None,
    "shower": "shower.glb",
    "bathtub": "bathtub.glb",
    "plant": "pro/potted_plant_01/potted_plant_01_1k.gltf",
    "wall_art": (
        "pro/hanging_picture_frame_01/"
        "hanging_picture_frame_01_1k.gltf"
    ),
    "wall_mirror": "pro/ornate_mirror_01/ornate_mirror_01_1k.gltf",
    "wall_clock": "pro/wall_clock/wall_clock_1k.gltf",
    "wall_sconce": (
        "pro/industrial_wall_sconce/industrial_wall_sconce_1k.gltf"
    ),
    "ceiling_light": (
        "pro/modern_ceiling_lamp_01/modern_ceiling_lamp_01_1k.gltf"
    ),
    "decor_vase": "pro/ceramic_vase_03/ceramic_vase_03_1k.gltf",
    "throw_pillows": "pro/throw_pillows_01/throw_pillows_01_1k.gltf",
}

MODERN_MODELS = {
    "sofa": "pro/glam_velvet_sofa/GlamVelvetSofa.glb",
    "armchair": "pro/sheen_chair/SheenChair.glb",
    "coffee_table": "pro/modern_coffee_table_01/modern_coffee_table_01_1k.gltf",
    "tv_unit": "pro/modern_wooden_cabinet/modern_wooden_cabinet_1k.gltf",
    "sideboard": "pro/modern_wooden_cabinet/modern_wooden_cabinet_1k.gltf",
    "dining_table": None,
    "dining_chair": "pro/sheen_chair/SheenChair.glb",
}

BOHO_MODELS = {
    # Relaxed contemporary silhouettes coordinate with layered natural
    # textiles much better than the former Gothic/ornate classic family.
    "sofa": "pro/glam_velvet_sofa/GlamVelvetSofa.glb",
    "armchair": "pro/modern_arm_chair_01/modern_arm_chair_01_1k.gltf",
    "coffee_table": "pro/modern_coffee_table_01/modern_coffee_table_01_1k.gltf",
    "tv_unit": "pro/modern_wooden_cabinet/modern_wooden_cabinet_1k.gltf",
    "sideboard": "pro/modern_wooden_cabinet/modern_wooden_cabinet_1k.gltf",
    "nightstand": (
        "pro/painted_wooden_nightstand/"
        "painted_wooden_nightstand_1k.gltf"
    ),
    "wardrobe": "pro/drawer_cabinet/drawer_cabinet_1k.gltf",
    "bed": None,
    "dining_table": None,
    "dining_chair": "pro/sheen_chair/SheenChair.glb",
}

CLASSIC_MODELS = {
    "sofa": (
        "pro/sheen_wood_leather_sofa/SheenWoodLeatherSofa.glb"
    ),
    "armchair": "pro/ArmChair_01/ArmChair_01_1k.gltf",
    "coffee_table": "pro/CoffeeTable_01/CoffeeTable_01_1k.gltf",
    "tv_unit": "pro/ClassicConsole_01/ClassicConsole_01_1k.gltf",
    "sideboard": "pro/ClassicConsole_01/ClassicConsole_01_1k.gltf",
    "nightstand": (
        "pro/ClassicNightstand_01/ClassicNightstand_01_1k.gltf"
    ),
    "wardrobe": "pro/GothicCabinet_01/GothicCabinet_01_1k.gltf",
    "dining_table": None,
    "dining_chair": "pro/dining_chair_02/dining_chair_02_1k.gltf",
}

MODERN_STYLE_WORDS = {
    "modern", "contemporary", "minimalist", "scandinavian", "japandi",
    "industrial", "mid-century",
}
CLASSIC_STYLE_WORDS = {"classic", "traditional"}
BOHO_STYLE_WORDS = {"bohemian", "boho"}

_MESH_CACHE = {}
_PBR_CACHE = {}
_PBR_MESH_MATERIALS = {}


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
    if mesh.has_triangle_uvs() and len(mesh.textures):
        # Textured catalog models are lit by the renderer. Baking light into
        # their vertex colors would multiply the authored albedo twice and
        # make fabric, wood and metal look muddy.
        return mesh
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
    if any(word in style_key for word in BOHO_STYLE_WORDS):
        return BOHO_MODELS.get(asset_key, DEFAULT_MODELS.get(asset_key))
    if any(word in style_key for word in MODERN_STYLE_WORDS):
        return MODERN_MODELS.get(asset_key, DEFAULT_MODELS.get(asset_key))
    return DEFAULT_MODELS.get(asset_key)


def _palette_material(asset_key, palette):
    if not palette:
        return None
    if asset_key in {
        "sofa", "armchair", "office_chair", "bed", "throw_pillows",
    }:
        return np.asarray(palette["sofa"], dtype=float)
    if asset_key in {
        "coffee_table", "tv_unit", "nightstand", "dining_table",
        "dining_chair", "sideboard", "desk", "bookshelf",
    }:
        return np.asarray(palette["wood"], dtype=float)
    if asset_key in {"wardrobe", "kitchen_island", "fridge", "vanity"}:
        return np.asarray(palette["cabinet"], dtype=float)
    if asset_key in {"toilet", "shower", "bathtub"}:
        return np.array([0.92, 0.94, 0.94])
    if asset_key in {"wall_art", "decor_vase"}:
        return np.asarray(palette["accent"], dtype=float)
    if asset_key in {"wall_mirror", "wall_clock", "wall_sconce", "ceiling_light"}:
        return np.asarray(palette["metal"], dtype=float)
    if asset_key == "plant":
        return np.asarray(palette.get("wood", [0.42, 0.32, 0.22]), dtype=float)
    return None


def _coordinate_material(
    source_color,
    target_color,
    professional=False,
    coordination_strength=None,
):
    """Coordinate authored materials without erasing their texture detail."""
    if target_color is None:
        return source_color
    source_color = np.asarray(source_color, dtype=float)
    brightness = np.mean(source_color, axis=-1, keepdims=True)
    # Professional models already carry carefully authored color variation.
    # A restrained tint keeps the user's palette visible without flattening
    # fabric weave, wood grain, patina, or painted details.
    strength = (
        float(coordination_strength)
        if coordination_strength is not None
        else (0.28 if professional else 0.72)
    )
    weight = np.full_like(brightness, strength)
    white_weight = min(0.24, max(0.10, strength * 0.28))
    weight = np.where(brightness > 0.88, white_weight, weight)
    tonal_floor = 0.84 if strength >= 0.85 else 0.72
    tonal_range = 0.36 if strength >= 0.85 else 0.48
    tonal_target = np.clip(
        np.asarray(target_color) * (tonal_floor + tonal_range * brightness),
        0,
        1,
    )
    return np.clip(
        source_color * (1.0 - weight) + tonal_target * weight,
        0,
        1,
    )


def _image_pixels(image):
    if image is None:
        return None
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return None
    return np.ascontiguousarray(pixels[:, :, :3])


def _authored_texture(
    source,
    target_color,
    professional=False,
    coordination_strength=None,
):
    """Return the model's mapped albedo and UVs, retaining real material detail."""
    visual = getattr(source, "visual", None)
    material = getattr(visual, "material", None)
    uv = getattr(visual, "uv", None)
    if material is None or uv is None:
        return None, None, None

    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        return None, None, None

    pixels = _image_pixels(image)
    if pixels is None:
        return None, None, None
    coordinated = _coordinate_material(
        pixels[:, :, :3].astype(np.float32) / 255.0,
        target_color,
        professional=professional,
        coordination_strength=coordination_strength,
    )
    texture = np.ascontiguousarray(
        np.clip(np.rint(coordinated * 255.0), 0, 255).astype(np.uint8)
    )
    pbr = {
        "albedo": texture,
        "normal": _image_pixels(getattr(material, "normalTexture", None)),
        "arm": None,
        "roughness": float(
            getattr(material, "roughnessFactor", None) or 0.52
        ),
        "metallic": float(
            getattr(material, "metallicFactor", None) or 0.0
        ),
    }
    metallic_roughness = _image_pixels(
        getattr(material, "metallicRoughnessTexture", None)
    )
    if metallic_roughness is not None:
        arm = metallic_roughness.copy()
        occlusion = _image_pixels(getattr(material, "occlusionTexture", None))
        arm[:, :, 0] = (
            occlusion[:, :, 0]
            if occlusion is not None and occlusion.shape[:2] == arm.shape[:2]
            else 255
        )
        pbr["arm"] = arm
    return np.asarray(uv, dtype=float), texture, pbr


def catalog_material_record_for_mesh(mesh):
    """Return the catalog model's complete glTF PBR material, when available."""
    registered = _PBR_MESH_MATERIALS.get(id(mesh))
    if registered is None or registered[0] is not mesh:
        return None
    spec = registered[1]
    from open3d.visualization import rendering

    record = rendering.MaterialRecord()
    record.shader = "defaultLit"
    record.base_color = [1.0, 1.0, 1.0, 1.0]
    record.base_roughness = spec["roughness"]
    record.base_metallic = spec["metallic"]
    record.base_reflectance = 0.42
    record.albedo_img = o3d.geometry.Image(spec["albedo"].copy())
    if spec["normal"] is not None:
        record.normal_img = o3d.geometry.Image(spec["normal"].copy())
    if spec["arm"] is not None:
        record.ao_rough_metal_img = o3d.geometry.Image(spec["arm"].copy())
    return record


def catalog_status() -> tuple[bool, str]:
    """Return whether every required native model is installed."""
    required = {
        name
        for mapping in (
            DEFAULT_MODELS,
            MODERN_MODELS,
            BOHO_MODELS,
            CLASSIC_MODELS,
        )
        for name in mapping.values()
        if name is not None
    }
    missing = sorted(name for name in required if not (CATALOG_ROOT / name).is_file())
    if missing:
        return False, "Local 3D catalog is missing: " + ", ".join(missing)
    return True, "Professional local editable 3D catalog is ready."


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
    coordination_strength = {
        # The contemporary sofa ships with a nearly black default material.
        # A stronger textile-safe recolor lets it follow the selected room
        # palette without removing its velvet shading or authored details.
        "sofa": 0.92,
        "armchair": 0.66,
        "office_chair": 0.58,
        "bed": 0.52,
        "throw_pillows": 0.58,
    }.get(asset_key, 0.30)
    material_key = (
        tuple(np.round(material_target, 3))
        if material_target is not None else ()
    )
    cache_key = (
        str(path), round(float(width), 3), round(float(depth), 3),
        round(height, 3), material_key, round(coordination_strength, 2),
    )
    if cache_key in _MESH_CACHE:
        meshes = [copy.deepcopy(mesh) for mesh in _MESH_CACHE[cache_key]]
        for mesh, spec in zip(meshes, _PBR_CACHE.get(cache_key, [])):
            if spec is not None:
                _PBR_MESH_MATERIALS[id(mesh)] = (mesh, spec)
        return meshes

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
        professional = PRO_ROOT in path.parents
        uv, texture, pbr = _authored_texture(
            source,
            material_target,
            professional=professional,
            coordination_strength=coordination_strength,
        )
        if uv is not None and len(uv) == len(source_vertices):
            # White vertex colors let the mapped albedo reach both the legacy
            # walkthrough and the modern PBR renderer without a second tint.
            vertex_colors = np.ones((len(source_vertices), 3), dtype=float)
        else:
            uv, texture, pbr = None, None, None
            color_visual = source.visual.to_color()
            vertex_colors = np.asarray(
                getattr(color_visual, "vertex_colors", [184, 184, 184, 255]),
                dtype=float,
            )
            if vertex_colors.ndim == 1:
                vertex_colors = vertex_colors.reshape(1, -1)
            vertex_colors = vertex_colors[:, :3]
            if vertex_colors.max(initial=0.0) > 1.0:
                vertex_colors /= 255.0
            if len(vertex_colors) != len(source_vertices):
                average = np.mean(vertex_colors, axis=0, keepdims=True)
                vertex_colors = np.repeat(
                    average, len(source_vertices), axis=0
                )
            vertex_colors = _coordinate_material(
                vertex_colors,
                material_target,
                professional=professional,
                coordination_strength=coordination_strength,
            )
        component_data.append(
            (
                vertices,
                np.asarray(source.faces, dtype=np.int32),
                vertex_colors,
                uv,
                texture,
                pbr,
            )
        )
    if not component_data:
        return None

    all_vertices = np.vstack([
        vertices
        for vertices, _faces, _color, _uv, _texture, _pbr in component_data
    ])
    source_min = all_vertices.min(axis=0)
    source_max = all_vertices.max(axis=0)
    source_extents = source_max - source_min
    if np.any(source_extents < 1e-6):
        return None

    target_extents = np.array([float(width), float(depth), height])
    scale = target_extents / np.maximum(source_extents, 1e-6)
    center_xy = target_extents[:2] / 2
    meshes = []
    pbr_specs = []
    for vertices, faces, colors, uv, texture, pbr in component_data:
        normalized = (vertices - source_min) * scale
        normalized[:, 0] -= center_xy[0]
        normalized[:, 1] -= center_xy[1]
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(normalized)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(colors, 0, 1)
        )
        if uv is not None and texture is not None:
            triangle_uvs = uv[faces].reshape((-1, 2))
            mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
            mesh.triangle_material_ids = o3d.utility.IntVector(
                np.zeros(len(faces), dtype=np.int32)
            )
            mesh.textures = [o3d.geometry.Image(texture)]
        mesh.compute_vertex_normals()
        _shade_materials(mesh)
        if pbr is not None:
            _PBR_MESH_MATERIALS[id(mesh)] = (mesh, pbr)
        pbr_specs.append(pbr)
        meshes.append(mesh)

    _MESH_CACHE[cache_key] = [copy.deepcopy(mesh) for mesh in meshes]
    _PBR_CACHE[cache_key] = pbr_specs
    return meshes
