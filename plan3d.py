"""
3D visualization for floor plans with realistic doors and windows
"""

import open3d as o3d
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import math

# ================= CONFIG =================
WALL_HEIGHT = 3.6
WALL_THICKNESS = 0.15

DOOR_HEIGHT = 2.4
MIN_DOOR_WIDTH = 1.5
DOOR_FRAME_WIDTH = 0.08
DOOR_FRAME_DEPTH = 0.06
DOOR_THICKNESS = 0.05

WINDOW_SILL = 0.9
WINDOW_HEIGHT = 1.5
WINDOW_WIDTH = 1.0
WINDOW_FRAME_WIDTH = 0.06
WINDOW_FRAME_DEPTH = 0.05
WINDOW_GLASS_THICKNESS = 0.01

EYE_HEIGHT = 1.65
MOVE_SPEED = 0.08

ANGLE_TOL_DEG = 3.0
MERGE_DIST_TOL = 0.05

ROOM_SCALE = 5.0

# Scaled dimensions
DOOR_WIDTH_SCALED = MIN_DOOR_WIDTH * ROOM_SCALE
WINDOW_WIDTH_SCALED = WINDOW_WIDTH * ROOM_SCALE

# Colors
WALL_COLOR = [0.78, 0.74, 0.68]        # Warm medium beige wall (darker for contrast)
FLOOR_COLOR = [0.55, 0.38, 0.22]       # Rich wood floor
CEILING_COLOR = [0.95, 0.95, 0.93]     # Off-white ceiling
DOOR_FRAME_COLOR = [0.30, 0.18, 0.08]  # Dark wood frame (richer contrast)
DOOR_PANEL_COLOR = [0.45, 0.30, 0.18]  # Wood door panel (darker)
DOOR_HANDLE_COLOR = [0.85, 0.75, 0.30] # Polished brass handle
WINDOW_FRAME_COLOR = [0.95, 0.95, 0.95] # Bright white frame (contrast against wall)
WINDOW_GLASS_COLOR = [0.95, 0.97, 1.00] # Near-white daylight glass
WINDOW_SILL_COLOR = [0.96, 0.96, 0.94]  # Bright white sill (stands out)
SKY_COLOR = [0.92, 0.96, 1.00]          # Very bright sky behind glass
# =========================================


# ---------- UTILS ----------
def px_to_m(p, px_per_m):
    return (np.array([p[0], -p[1]], dtype=float) / float(px_per_m)) * ROOM_SCALE


def polygon_centroid_3d(poly):
    c = Polygon([(x, y) for x, y in poly]).centroid
    return np.array([c.x, c.y, EYE_HEIGHT])


def create_box_mesh(width, height, depth, color):
    """Create a box mesh with given dimensions and color"""
    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def floor_mesh(poly, color):
    """Create floor mesh from polygon"""
    verts = [[x, y, 0.0] for x, y in poly]
    tris = [[0, i, i + 1] for i in range(1, len(verts) - 1)]
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts)
    m.triangles = o3d.utility.Vector3iVector(tris)
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


# Virtual light direction for baked wall shading (light from window side)
_LIGHT_DIR = np.array([0.0, 1.0, -0.3])
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)

def _baked_wall_color(p1, p2, base_color):
    """Compute a baked shading color for a wall based on its orientation.
    Walls facing the virtual light are brighter; walls facing away are dimmer.
    This replaces Open3D's realtime lighting to avoid specular artifacts."""
    wall_dir = np.array(p2[:2]) - np.array(p1[:2])
    length = np.linalg.norm(wall_dir)
    if length < 1e-9:
        return base_color
    wall_dir /= length
    # Outward normal (perpendicular to wall, pointing into room)
    normal_2d = np.array([-wall_dir[1], wall_dir[0]])
    normal_3d = np.array([normal_2d[0], normal_2d[1], 0.0])
    # Lambertian factor: how much this wall faces the light
    dot = np.dot(normal_3d, _LIGHT_DIR)
    # Map [-1, 1] → [0.78, 1.12] brightness range (subtle but visible)
    factor = 0.95 + 0.17 * dot
    return [min(1.0, c * factor) for c in base_color]


def wall_segment(p1, p2, z0, z1, color=None):
    """Create a wall segment between two points.
    Uses baked directional shading so walls look separated without realtime light."""
    if color is None:
        color = WALL_COLOR
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    length = np.linalg.norm(p2[:2] - p1[:2])
    if length < 0.05 or z1 <= z0:
        return None

    mesh = o3d.geometry.TriangleMesh.create_box(
        width=length,
        height=WALL_THICKNESS,
        depth=z1 - z0
    )

    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    mesh.rotate(
        o3d.geometry.get_rotation_matrix_from_xyz((0, 0, angle)),
        center=(0, 0, 0)
    )

    mesh.translate((p1[0], p1[1], z0))
    # Bake directional shading into vertex color (no specular artifacts)
    shaded_color = _baked_wall_color(p1, p2, color)
    mesh.paint_uniform_color(shaded_color)
    mesh.compute_vertex_normals()
    return mesh


# ---------- REALISTIC DOOR ----------
def create_door_geometry(p1, p2, wall_angle):
    """
    Create a realistic door with frame, panel, and handle
    Returns list of meshes
    """
    meshes = []
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    door_width = np.linalg.norm(p2 - p1)
    
    # Door direction vectors
    dir_along = (p2 - p1) / door_width if door_width > 0 else np.array([1, 0])
    dir_perp = np.array([-dir_along[1], dir_along[0]])
    
    # === DOOR PANEL (slightly open, rotated 20 degrees) ===
    door_open_angle = math.radians(20)
    panel_width = door_width - 0.04  # Slightly smaller than opening
    
    # Door panel
    door_panel = create_box_mesh(panel_width, DOOR_THICKNESS, DOOR_HEIGHT - 0.02, DOOR_PANEL_COLOR)
    
    # Position at hinge point (left side of opening)
    hinge_pos = p1 + dir_along * 0.02
    
    # Rotate around Z axis for wall alignment, then add open angle
    door_panel.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle + door_open_angle)), center=(0, 0, 0))
    door_panel.translate([hinge_pos[0], hinge_pos[1], 0.01])
    meshes.append(door_panel)
    
    # === DOOR HANDLE ===
    handle_height = DOOR_HEIGHT * 0.45
    
    # Calculate handle position on the opened door
    handle_offset = panel_width * 0.85
    handle_local_x = handle_offset * math.cos(door_open_angle)
    handle_local_y = handle_offset * math.sin(door_open_angle)
    handle_pos = hinge_pos + dir_along * handle_local_x + dir_perp * handle_local_y
    
    # Handle base plate
    base_plate = create_box_mesh(0.04, 0.02, 0.14, DOOR_HANDLE_COLOR)
    base_plate.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle + door_open_angle)), center=(0, 0, 0))
    base_plate.translate([handle_pos[0], handle_pos[1], handle_height - 0.07])
    meshes.append(base_plate)
    
    # Handle lever
    lever = create_box_mesh(0.14, 0.025, 0.03, DOOR_HANDLE_COLOR)
    lever.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle + door_open_angle)), center=(0, 0, 0))
    lever_offset = dir_perp * 0.025
    lever.translate([handle_pos[0] + lever_offset[0], handle_pos[1] + lever_offset[1], handle_height])
    meshes.append(lever)
    
    return meshes


# ---------- REALISTIC WINDOW ----------
def create_window_geometry(p1, p2, wall_angle, sill_height=WINDOW_SILL, window_height=WINDOW_HEIGHT):
    """
    Create a realistic window with frame, glass panes, and sill.
    The glass pane is placed exactly within the wall opening.
    Returns list of meshes
    """
    meshes = []
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    window_width = np.linalg.norm(p2 - p1)
    if window_width < 0.1:
        return meshes

    dir_along = (p2 - p1) / window_width
    dir_perp = np.array([-dir_along[1], dir_along[0]])
    frame_outer_offset = dir_perp * (WALL_THICKNESS - WINDOW_FRAME_DEPTH)
    glass_plane_offset = dir_perp * max(WALL_THICKNESS - 0.012, 0.0)
    reveal_depth = max(WALL_THICKNESS - WINDOW_FRAME_DEPTH, 0.01)
    reveal_color = [0.90, 0.88, 0.84]

    # --- Window Sill ---
    sill_depth = WALL_THICKNESS + 0.08
    sill_mesh = create_box_mesh(window_width, sill_depth, 0.04, WINDOW_SILL_COLOR)
    sill_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    sill_start = p1 + dir_perp * (WALL_THICKNESS - sill_depth)
    sill_mesh.translate([sill_start[0], sill_start[1], sill_height - 0.04])
    meshes.append(sill_mesh)

    # --- Glass Pane (vertex-colored: stronger centered daylight glow) ---
    glass_width = window_width - WINDOW_FRAME_WIDTH * 2
    glass_height = window_height - WINDOW_FRAME_WIDTH * 2
    if glass_width > 0 and glass_height > 0:
        # Build a quad with per-vertex colors: bright at centre, dimmer at edges
        gx0 = WINDOW_FRAME_WIDTH
        gx1 = WINDOW_FRAME_WIDTH + glass_width
        gz0 = sill_height + WINDOW_FRAME_WIDTH
        gz1 = gz0 + glass_height
        gmx = (gx0 + gx1) / 2
        gmz = (gz0 + gz1) / 2

        # 9-point grid: corners, edge midpoints, centre
        pts_local = [
            (gx0, gz0), (gmx, gz0), (gx1, gz0),    # bottom row
            (gx0, gmz), (gmx, gmz), (gx1, gmz),    # middle row
            (gx0, gz1), (gmx, gz1), (gx1, gz1),     # top row
        ]
        edge_col = np.array([0.84, 0.90, 0.97])     # slightly softer cool edges
        mid_col  = np.array([0.94, 0.97, 1.00])     # slightly reduced daylight mid tone
        ctr_col  = np.array([1.00, 1.00, 1.00])     # pure white centre
        colors_9 = [edge_col, mid_col, edge_col,
                     mid_col, ctr_col, mid_col,
                     edge_col, mid_col, edge_col]

        # Convert local coords to world
        verts_3d = []
        for (lx, lz) in pts_local:
            # lx is distance along wall from p1, lz is world Z
            world_xy = p1 + dir_along * lx + glass_plane_offset
            verts_3d.append([world_xy[0], world_xy[1], lz])

        # 8 triangles covering the 3x3 grid
        tris = [
            [0,1,4],[0,4,3],  # bottom-left quad
            [1,2,5],[1,5,4],  # bottom-right quad
            [3,4,7],[3,7,6],  # top-left quad
            [4,5,8],[4,8,7],  # top-right quad
        ]

        glass_mesh = o3d.geometry.TriangleMesh()
        glass_mesh.vertices = o3d.utility.Vector3dVector(verts_3d)
        glass_mesh.triangles = o3d.utility.Vector3iVector(tris)
        glass_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_9)
        # No normals — colour is baked, light-independent
        meshes.append(glass_mesh)

    # --- Sky box behind the window ---
    sky_extra_w = window_width * 0.45
    sky_extra_h = window_height * 0.45
    sky_w = window_width + sky_extra_w
    sky_h = window_height + sky_extra_h
    sky_depth = 0.7
    sky_panel = create_box_mesh(sky_w, sky_depth, sky_h, [0.95, 0.98, 1.00])
    sky_panel.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    sky_start = p1 - dir_along * (sky_extra_w / 2) + dir_perp * (WALL_THICKNESS + 0.03)
    sky_panel.translate([sky_start[0], sky_start[1], sill_height - sky_extra_h / 2])
    meshes.append(sky_panel)

    # --- Light spill on the floor below the window ---
    # Wider and brighter trapezoid to read more clearly as daylight from the window
    spill_inward = 3.2   # how far the light reaches into the room
    near_expand = 0.10   # slight expansion at the window edge
    far_expand = 0.85    # wider spread deeper into the room
    floor_verts = [
        [p1[0] - dir_along[0] * near_expand, p1[1] - dir_along[1] * near_expand, 0.005],
        [p2[0] + dir_along[0] * near_expand, p2[1] + dir_along[1] * near_expand, 0.005],
        [p2[0] + dir_along[0] * far_expand - dir_perp[0] * spill_inward, p2[1] + dir_along[1] * far_expand - dir_perp[1] * spill_inward, 0.005],
        [p1[0] - dir_along[0] * far_expand - dir_perp[0] * spill_inward, p1[1] - dir_along[1] * far_expand - dir_perp[1] * spill_inward, 0.005],
    ]
    spill_mesh = o3d.geometry.TriangleMesh()
    spill_mesh.vertices = o3d.utility.Vector3dVector(floor_verts)
    spill_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    # Gradient: brighter near window, dimmer further away
    bright = np.array([0.96, 0.95, 0.85])
    dim = np.array([0.76, 0.68, 0.50])
    spill_mesh.vertex_colors = o3d.utility.Vector3dVector([bright, bright, dim, dim])
    spill_mesh.compute_vertex_normals()
    meshes.append(spill_mesh)

    # --- Opening reveals (show the full wall thickness inside the window opening) ---
    bottom_reveal = create_box_mesh(window_width, reveal_depth, WINDOW_FRAME_WIDTH, reveal_color)
    bottom_reveal.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    bottom_reveal.translate([p1[0], p1[1], sill_height])
    meshes.append(bottom_reveal)

    top_reveal = create_box_mesh(window_width, reveal_depth, WINDOW_FRAME_WIDTH, reveal_color)
    top_reveal.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    top_reveal.translate([p1[0], p1[1], sill_height + window_height - WINDOW_FRAME_WIDTH])
    meshes.append(top_reveal)

    left_reveal = create_box_mesh(WINDOW_FRAME_WIDTH, reveal_depth, window_height, reveal_color)
    left_reveal.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    left_reveal.translate([p1[0], p1[1], sill_height])
    meshes.append(left_reveal)

    right_reveal = create_box_mesh(WINDOW_FRAME_WIDTH, reveal_depth, window_height, reveal_color)
    right_reveal.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    right_reveal.translate([p2[0] - dir_along[0] * WINDOW_FRAME_WIDTH, p2[1] - dir_along[1] * WINDOW_FRAME_WIDTH, sill_height])
    meshes.append(right_reveal)

    # --- White Frame (4 bars around the glass) ---
    frame_w = WINDOW_FRAME_WIDTH
    # Bottom frame bar
    bot_frame = create_box_mesh(window_width, WINDOW_FRAME_DEPTH, frame_w, WINDOW_FRAME_COLOR)
    bot_frame.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    bot_frame.translate([p1[0] + frame_outer_offset[0], p1[1] + frame_outer_offset[1], sill_height])
    meshes.append(bot_frame)
    # Top frame bar
    top_frame = create_box_mesh(window_width, WINDOW_FRAME_DEPTH, frame_w, WINDOW_FRAME_COLOR)
    top_frame.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    top_frame.translate([p1[0] + frame_outer_offset[0], p1[1] + frame_outer_offset[1], sill_height + window_height - frame_w])
    meshes.append(top_frame)
    # Left frame bar
    left_frame = create_box_mesh(frame_w, WINDOW_FRAME_DEPTH, window_height, WINDOW_FRAME_COLOR)
    left_frame.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    left_frame.translate([p1[0] + frame_outer_offset[0], p1[1] + frame_outer_offset[1], sill_height])
    meshes.append(left_frame)
    # Right frame bar
    right_frame = create_box_mesh(frame_w, WINDOW_FRAME_DEPTH, window_height, WINDOW_FRAME_COLOR)
    right_frame.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, wall_angle)), center=(0, 0, 0))
    right_start = p2 - dir_along * frame_w
    right_frame.translate([right_start[0] + frame_outer_offset[0], right_start[1] + frame_outer_offset[1], sill_height])
    meshes.append(right_frame)

    return meshes


# ---------- EDGE MERGING ----------
def are_collinear(e1, e2):
    v1 = np.array(e1["p2"]) - np.array(e1["p1"])
    v2 = np.array(e2["p2"]) - np.array(e2["p1"])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return False
    v1 /= n1
    v2 /= n2

    if abs(np.dot(v1, v2)) < math.cos(math.radians(ANGLE_TOL_DEG)):
        return False

    return LineString([e1["p1"], e1["p2"]]).distance(
        LineString([e2["p1"], e2["p2"]])
    ) < MERGE_DIST_TOL


def merge_edges(raw_edges):
    merged, used = [], set()

    for i, e1 in enumerate(raw_edges):
        if i in used:
            continue

        pts = [e1["p1"], e1["p2"]]

        for j, e2 in enumerate(raw_edges):
            if j <= i or j in used:
                continue
            if are_collinear(e1, e2):
                pts.extend([e2["p1"], e2["p2"]])
                used.add(j)

        dir_vec = np.array(e1["p2"]) - np.array(e1["p1"])
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-9:
            continue
        dir_vec /= norm
        p0 = np.array(pts[0])

        proj = [(np.dot(np.array(p) - p0, dir_vec), p) for p in pts]
        proj.sort(key=lambda x: x[0])

        merged.append({
            "p1": proj[0][1],
            "p2": proj[-1][1],
            "line": LineString([proj[0][1], proj[-1][1]]),
            "length": abs(proj[-1][0] - proj[0][0]),
            "openings": []
        })

        used.add(i)

    return merged


# ---------- WALL BUILD WITH REALISTIC OPENINGS ----------
def build_walls_from_edges(edges):
    """Build wall meshes from edges, including realistic doors and windows"""
    meshes = []

    for e in edges:
        p1, p2 = np.array(e["p1"]), np.array(e["p2"])
        length = e["length"]
        wall_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        
        if length < 0.1:
            continue
        
        last = 0.0
        openings = sorted(e.get("openings", []), key=lambda x: x[1])

        for typ, t0, t1 in openings:
            t0 = max(0, min(1, t0))
            t1 = max(0, min(1, t1))
            
            # Wall segment before opening
            if t0 > last + 0.01:
                w = wall_segment(
                    p1 + (p2 - p1) * last,
                    p1 + (p2 - p1) * t0,
                    0, WALL_HEIGHT, WALL_COLOR
                )
                if w:
                    meshes.append(w)

            # Opening positions
            open_p1 = p1 + (p2 - p1) * t0
            open_p2 = p1 + (p2 - p1) * t1

            if typ == "door":
                # Wall above door
                w_above = wall_segment(open_p1, open_p2, DOOR_HEIGHT, WALL_HEIGHT, WALL_COLOR)
                if w_above:
                    meshes.append(w_above)
                
                # Add realistic door geometry
                door_meshes = create_door_geometry(open_p1, open_p2, wall_angle)
                meshes.extend(door_meshes)

            elif typ == "window":
                # Wall below window (under sill)
                w_below = wall_segment(open_p1, open_p2, 0, WINDOW_SILL, WALL_COLOR)
                if w_below:
                    meshes.append(w_below)
                
                # Wall above window
                w_above = wall_segment(open_p1, open_p2, WINDOW_SILL + WINDOW_HEIGHT, WALL_HEIGHT, WALL_COLOR)
                if w_above:
                    meshes.append(w_above)
                
                # Add realistic window geometry
                window_meshes = create_window_geometry(open_p1, open_p2, wall_angle)
                meshes.extend(window_meshes)

            last = t1

        # Wall segment after last opening
        if last < 0.99:
            w = wall_segment(
                p1 + (p2 - p1) * last,
                p2,
                0, WALL_HEIGHT, WALL_COLOR
            )
            if w:
                meshes.append(w)

    return meshes


# ---------- MAIN ----------
def build_3d_apartment_and_walk(
    outer_px,
    room_polygons_px,
    doors_px,
    windows_px,
    px_per_m=100,
    room_only=False
):
    """Build and display 3D apartment with realistic doors and windows"""
    outer = [px_to_m(p, px_per_m) for p in outer_px]
    rooms = [[px_to_m(p, px_per_m) for p in poly] for poly in room_polygons_px]

    loops = rooms if room_only else [outer] + rooms

    # Build edges per loop and keep them separate
    all_edges = []
    loop_edges = []
    loop_types = []
    
    for loop_idx, loop in enumerate(loops):
        loop_type = "inner" if room_only or loop_idx > 0 else "outer"
        raw_edges = []
        for i in range(len(loop)):
            p1, p2 = loop[i], loop[(i + 1) % len(loop)]
            if np.linalg.norm(p2 - p1) < 0.1:
                continue
            raw_edges.append({
                "p1": p1,
                "p2": p2,
                "line": LineString([p1, p2]),
                "length": LineString([p1, p2]).length,
                "openings": [],
                "loop_type": loop_type
            })
        
        merged = merge_edges(raw_edges)
        loop_edges.append(merged)
        loop_types.append(loop_type)
        all_edges.extend(merged)

    # Assign doors to appropriate loops
    for d1, d2 in doors_px:
        a, b = px_to_m(d1, px_per_m), px_to_m(d2, px_per_m)
        door_line = LineString([a, b])
        
        closest_edge = None
        min_dist = float('inf')
        
        for loop_idx in range(len(loop_edges)):
            if loop_types[loop_idx] == "inner":
                for edge in loop_edges[loop_idx]:
                    dist = edge["line"].distance(door_line)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = edge
        
        if closest_edge is None and not room_only and len(loop_edges) > 0:
            for edge in loop_edges[0]:
                dist = edge["line"].distance(door_line)
                if dist < min_dist:
                    min_dist = dist
                    closest_edge = edge
        
        if closest_edge and min_dist < 0.5:
            mid = (a + b) / 2
            dir_vec = (np.array(closest_edge["p2"]) - np.array(closest_edge["p1"]))
            norm = np.linalg.norm(dir_vec)
            if norm > 0:
                dir_vec /= norm
                half = MIN_DOOR_WIDTH / 2
                p0, p1 = mid - dir_vec * half, mid + dir_vec * half
                t0 = closest_edge["line"].project(Point(p0)) / closest_edge["length"]
                t1 = closest_edge["line"].project(Point(p1)) / closest_edge["length"]
                closest_edge["openings"].append(("door", min(t0, t1), max(t0, t1)))

    # Assign windows to appropriate loops
    for w1, w2, _ in windows_px:
        a, b = px_to_m(w1, px_per_m), px_to_m(w2, px_per_m)
        window_line = LineString([a, b])
        
        closest_edge = None
        min_dist = float('inf')
        
        for loop_idx in range(len(loop_edges)):
            if loop_types[loop_idx] == "inner":
                for edge in loop_edges[loop_idx]:
                    dist = edge["line"].distance(window_line)
                    if dist < min_dist:
                        min_dist = dist
                        closest_edge = edge
        
        if closest_edge is None and not room_only and len(loop_edges) > 0:
            for edge in loop_edges[0]:
                dist = edge["line"].distance(window_line)
                if dist < min_dist:
                    min_dist = dist
                    closest_edge = edge
        
        if closest_edge and min_dist < 0.5:
            t0 = closest_edge["line"].project(Point(a)) / closest_edge["length"]
            t1 = closest_edge["line"].project(Point(b)) / closest_edge["length"]
            closest_edge["openings"].append(("window", min(t0, t1), max(t0, t1)))

    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Apartment View - WASD to move, Q to quit", 1400, 900)

    # Add floor with wood color
    if room_only:
        floor = floor_mesh(rooms[0], FLOOR_COLOR)
        vis.add_geometry(floor)
        # Add ceiling
        ceiling = floor_mesh(rooms[0], CEILING_COLOR)
        ceiling.translate((0, 0, WALL_HEIGHT))
        vis.add_geometry(ceiling)
    else:
        # Main floor
        floor = floor_mesh(outer, FLOOR_COLOR)
        vis.add_geometry(floor)
        # Add ceiling
        ceiling = floor_mesh(outer, CEILING_COLOR)
        ceiling.translate((0, 0, WALL_HEIGHT))
        vis.add_geometry(ceiling)

    # Add walls with realistic doors and windows
    for w in build_walls_from_edges(all_edges):
        if w:
            vis.add_geometry(w)

    # Setup camera
    ctr = vis.get_view_control()
    cam = polygon_centroid_3d(rooms[0] if room_only else outer)
    ctr.set_lookat(cam)
    ctr.set_front([0, 1, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.5)

    # Render options — light OFF to eliminate specular reflection on walls.
    # Wall separation is baked into vertex colors via _baked_wall_color().
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.15, 0.15, 0.18])  # Dark background
    render_opt.light_on = False

    # Movement controls
    def move(dx, dy):
        cam[0] += dx
        cam[1] += dy
        ctr.set_lookat(cam)

    vis.register_key_callback(ord("W"), lambda v: move(0, MOVE_SPEED))
    vis.register_key_callback(ord("w"), lambda v: move(0, MOVE_SPEED))
    vis.register_key_callback(ord("S"), lambda v: move(0, -MOVE_SPEED))
    vis.register_key_callback(ord("s"), lambda v: move(0, -MOVE_SPEED))
    vis.register_key_callback(ord("A"), lambda v: move(-MOVE_SPEED, 0))
    vis.register_key_callback(ord("a"), lambda v: move(-MOVE_SPEED, 0))
    vis.register_key_callback(ord("D"), lambda v: move(MOVE_SPEED, 0))
    vis.register_key_callback(ord("d"), lambda v: move(MOVE_SPEED, 0))
    
    print("\n" + "=" * 50)
    print("3D VIEW CONTROLS:")
    print("  W/S - Move forward/backward")
    print("  A/D - Move left/right")
    print("  Mouse - Rotate view")
    print("  Close window to exit")
    print("=" * 50 + "\n")

    vis.run()
    vis.destroy_window()

