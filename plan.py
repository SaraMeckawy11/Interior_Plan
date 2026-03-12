"""
Clean version — REALISTIC WALLS (ASCII only)
- Correct real-world scale
- Solid wall volumes
- Proper wall height
- No furniture
"""

import os
import cv2
import numpy as np
import trimesh
import open3d as o3d

# ------------------------------
# USER CONFIG
# ------------------------------
FLOORPLAN_PATH = r"C:\Users\Lenovo\Desktop\Interior_plan\plan\1.jpg"
OUTPUT_PATH = r"C:\Users\Lenovo\Desktop\Interior_plan\plan\output\floorplan_3d.obj"

PX_PER_METER = 50.0     # pixels per meter (adjust if needed)
WALL_HEIGHT = 3.0       # meters
FLOOR_THICKNESS = 0.15  # meters

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ------------------------------
# LOAD FLOORPLAN
# ------------------------------
img = cv2.imread(FLOORPLAN_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Floorplan image not found")

H, W = img.shape

# ------------------------------
# WALL MASK
# ------------------------------
_, wall_mask = cv2.threshold(
    img, 140, 255, cv2.THRESH_BINARY_INV
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
wall_mask = cv2.dilate(wall_mask, kernel, iterations=2)

# ------------------------------
# BUILD WALLS AS SOLID VOLUMES
# ------------------------------
wall_pixels = np.column_stack(np.where(wall_mask > 0))
wall_meshes = []

voxel_size = 1.0 / PX_PER_METER

for y, x in wall_pixels:
    xm = x / PX_PER_METER
    ym = (H - y) / PX_PER_METER

    box = trimesh.creation.box(
        extents=(voxel_size, voxel_size, WALL_HEIGHT)
    )
    box.apply_translation([xm, ym, WALL_HEIGHT / 2])
    wall_meshes.append(box)

walls = trimesh.util.concatenate(wall_meshes)

# ------------------------------
# FLOOR
# ------------------------------
floor = trimesh.creation.box(
    extents=(
        W / PX_PER_METER,
        H / PX_PER_METER,
        FLOOR_THICKNESS
    )
)
floor.apply_translation([
    (W / PX_PER_METER) / 2,
    (H / PX_PER_METER) / 2,
    -FLOOR_THICKNESS / 2
])

# ------------------------------
# EXPORT OBJ
# ------------------------------
scene = trimesh.util.concatenate([floor, walls])
scene.export(OUTPUT_PATH)

print("Realistic walls generated")
print("Wall height:", WALL_HEIGHT, "meters")
print("Saved to:", OUTPUT_PATH)

# ------------------------------
# VISUALIZE
# ------------------------------
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(scene.vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(scene.faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.85, 0.85, 0.85])

o3d.visualization.draw_geometries(
    [o3d_mesh],
    window_name="3D Floorplan"
)
