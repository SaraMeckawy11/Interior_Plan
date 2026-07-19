"""Local CC0 architectural PBR material library.

The walkthrough uses these assets for visible mapped albedo in its interactive
renderer and keeps the normal, roughness, displacement and AO maps beside it
for the modern PBR renderer. Assets are stored locally, so room generation
does not make a network request.

Source: https://ambientcg.com/ (Creative Commons CC0)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
MATERIAL_ROOT = ROOT / "assets" / "archviz_materials" / "ambientcg"

MATERIALS = {
    "warm_oak": ("WoodFloor007", 2.2, 0.42),
    "dark_wood": ("WoodFloor041", 2.4, 0.56),
    "plaster": ("Plaster001", 1.5, 0.88),
    # Wallpaper002C has the soft mineral movement needed for limewash when
    # strongly tinted, while retaining its authored wall-scale surface.
    "limewash": ("Wallpaper002C", 1.3, 0.90),
    "wallpaper": ("Wallpaper002C", 1.3, 0.82),
    "bathroom_tile": ("Tiles133A", 1.15, 0.34),
    "marble": ("Marble019", 1.8, 0.24),
    "carpet": ("Carpet001", 1.0, 0.92),
    "curtain_fabric": ("Fabric062", 0.42, 0.88),
    "concrete": ("Concrete034", 1.1, 0.72),
    "terrazzo": ("Terrazzo019M", 1.3, 0.38),
}
_MESH_MATERIALS = {}


def _map_path(material_name: str, map_name: str) -> Path | None:
    asset_id = MATERIALS[material_name][0]
    folder = MATERIAL_ROOT / asset_id
    matches = sorted(folder.glob(f"{asset_id}_1K-JPG_{map_name}.*"))
    return matches[0] if matches else None


@lru_cache(maxsize=64)
def _texture_pixels(
    material_name: str,
    tint_key: tuple[float, float, float] | tuple,
    tint_strength: float,
) -> np.ndarray:
    path = _map_path(material_name, "Color")
    if path is None:
        raise FileNotFoundError(f"Missing local material texture: {material_name}")
    pixels = np.asarray(o3d.io.read_image(str(path)), dtype=np.uint8)
    if not tint_key or tint_strength <= 0:
        return np.ascontiguousarray(pixels[:, :, :3])

    source = pixels[:, :, :3].astype(np.float32) / 255.0
    target = np.asarray(tint_key, dtype=np.float32)
    luminance = np.mean(source, axis=2, keepdims=True)
    tonal_target = np.clip(target * (0.68 + 0.55 * luminance), 0, 1)
    tinted = source * (1.0 - tint_strength) + tonal_target * tint_strength
    return np.ascontiguousarray(
        np.clip(np.rint(tinted * 255.0), 0, 255).astype(np.uint8)
    )


def _projected_triangle_uvs(mesh, repeat_m: float):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        return np.empty((0, 2), dtype=float)

    tri_vertices = vertices[triangles]
    normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0],
    )
    axis = np.argmax(np.abs(normals), axis=1)
    uvs = np.empty((len(triangles), 3, 2), dtype=float)
    minimum = vertices.min(axis=0)
    repeat_m = max(float(repeat_m), 0.05)
    for index in range(len(triangles)):
        points = tri_vertices[index]
        if axis[index] == 2:       # floor / ceiling
            uv = (points[:, [0, 1]] - minimum[[0, 1]]) / repeat_m
        elif axis[index] == 0:     # wall whose normal is mainly X
            uv = (points[:, [1, 2]] - minimum[[1, 2]]) / repeat_m
        else:                      # wall whose normal is mainly Y
            uv = (points[:, [0, 2]] - minimum[[0, 2]]) / repeat_m
        uvs[index] = uv
    return uvs.reshape((-1, 2))


def apply_archviz_material(
    mesh,
    material_name: str,
    *,
    tint=None,
    tint_strength: float = 0.0,
    repeat_m: float | None = None,
    triangle_uvs=None,
    detail_maps: bool = True,
):
    """Map a local material onto a real mesh without changing its geometry."""
    if material_name not in MATERIALS:
        return mesh
    tint_key = tuple(np.round(np.asarray(tint, dtype=float), 3)) if tint is not None else ()
    pixels = _texture_pixels(material_name, tint_key, round(float(tint_strength), 3))
    if triangle_uvs is None:
        scale = repeat_m if repeat_m is not None else MATERIALS[material_name][1]
        triangle_uvs = _projected_triangle_uvs(mesh, scale)
    if len(triangle_uvs) != len(mesh.triangles) * 3:
        return mesh

    mesh.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(triangle_uvs, dtype=float))
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(len(mesh.triangles), dtype=np.int32)
    )
    mesh.textures = [o3d.geometry.Image(pixels.copy())]
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.ones((len(mesh.vertices), 3), dtype=float)
    )
    mesh.compute_vertex_normals()
    _MESH_MATERIALS[id(mesh)] = (
        material_name,
        tint_key,
        round(float(tint_strength), 3),
        bool(detail_maps),
    )
    return mesh


def pbr_material(
    material_name: str,
    *,
    tint=None,
    tint_strength=0.0,
    detail_maps=True,
):
    """Build an Open3D physically based material using every available map."""
    from open3d.visualization import rendering

    tint_key = tuple(np.round(np.asarray(tint, dtype=float), 3)) if tint is not None else ()
    record = rendering.MaterialRecord()
    record.shader = "defaultLit"
    record.base_color = [1.0, 1.0, 1.0, 1.0]
    record.base_roughness = MATERIALS[material_name][2]
    record.base_metallic = 0.0
    record.albedo_img = o3d.geometry.Image(
        _texture_pixels(
            material_name, tint_key, round(float(tint_strength), 3)
        ).copy()
    )
    maps = []
    if detail_maps:
        maps.extend([
            ("Roughness", "roughness_img"),
            ("AmbientOcclusion", "ao_img"),
        ])
    # Box-projected plank UVs do not have a single continuous tangent basis;
    # applying their normal map to cabinet corners produces black facets.
    # The authored albedo and roughness still provide realistic wood response.
    if detail_maps and material_name not in {"warm_oak", "dark_wood"}:
        maps.insert(0, ("NormalDX", "normal_img"))
    for map_name, attribute in maps:
        path = _map_path(material_name, map_name)
        if path is not None:
            setattr(record, attribute, o3d.io.read_image(str(path)))
    return record


def material_record_for_mesh(mesh):
    """Return a soft-lit PBR material for architecture or catalog geometry."""
    from open3d.visualization import rendering

    if hasattr(mesh, "has_vertex_normals") and not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    try:
        from furniture_catalog import catalog_material_record_for_mesh

        catalog_material = catalog_material_record_for_mesh(mesh)
        if catalog_material is not None:
            return catalog_material
    except (ImportError, RuntimeError):
        pass

    registered = _MESH_MATERIALS.get(id(mesh))
    if registered is not None:
        material_name, tint_key, tint_strength, detail_maps = registered
        return pbr_material(
            material_name,
            tint=tint_key or None,
            tint_strength=tint_strength,
            detail_maps=detail_maps,
        )

    record = rendering.MaterialRecord()
    record.shader = "defaultLit"
    record.base_roughness = 0.64
    record.base_metallic = 0.0
    record.base_reflectance = 0.38
    if mesh.has_triangle_uvs() and len(mesh.textures):
        record.albedo_img = o3d.geometry.Image(
            np.ascontiguousarray(np.asarray(mesh.textures[0]).copy())
        )
        record.base_color = [1.0, 1.0, 1.0, 1.0]
        record.base_roughness = 0.52
    else:
        colors = np.asarray(mesh.vertex_colors)
        color = (
            np.clip(np.mean(colors, axis=0), 0, 1)
            if len(colors)
            else np.array([0.74, 0.74, 0.74])
        )
        record.base_color = [float(color[0]), float(color[1]), float(color[2]), 1.0]
    return record


def floor_material(config, room_type: str, style: str) -> str:
    finish = str((config or {}).get("floor_finish", "Auto by style")).lower()
    room_key = (room_type or "").lower()
    style_key = (style or "").lower()
    explicit = {
        "light oak": "warm_oak",
        "warm oak": "warm_oak",
        "dark walnut": "dark_wood",
        "natural stone": "marble",
        "polished concrete": "concrete",
        "terrazzo": "terrazzo",
        "large tile": "bathroom_tile",
    }
    if finish != "auto by style":
        return explicit.get(finish, "warm_oak")
    if "bath" in room_key:
        return "bathroom_tile"
    if "kitchen" in room_key:
        return "bathroom_tile"
    if "industrial" in style_key:
        return "concrete"
    if any(word in style_key for word in ("classic", "traditional", "rustic")):
        return "dark_wood"
    return "warm_oak"


def wall_material(config, room_type: str, style: str) -> str:
    finish = str((config or {}).get("wall_finish", "Auto by style")).lower()
    room_key = (room_type or "").lower()
    style_key = (style or "").lower()
    # Explicit user choices always win. Previously the bathroom and
    # Industrial defaults were checked first, so the wall selector appeared
    # broken even though the selected value reached the renderer.
    if finish != "auto by style":
        return {
            "concrete": "concrete",
            "wallpaper": "wallpaper",
            "limewash": "limewash",
            "warm paint": "plaster",
            "cool paint": "plaster",
            "wood slats": "plaster",
            "panel moulding": "plaster",
            "accent color": "plaster",
        }.get(finish, "plaster")
    if "bath" in room_key:
        return "bathroom_tile"
    if "industrial" in style_key:
        return "concrete"
    if any(word in style_key for word in ("bohemian", "boho")):
        return "limewash"
    return "plaster"
