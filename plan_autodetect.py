"""
Auto-detect rooms and doors from a floor-plan image (classical CV).

Works on the common "black walls on white paper" plan style:
  1. Threshold dark strokes, then morphological OPEN drops thin lines
     (furniture symbols, door swing arcs) and keeps thick wall strokes.
  2. Morphological CLOSE seals door gaps so each room becomes its own region;
     regions touching the image border are the outside and get removed.
  3. Each remaining connected component is a room; its contour is simplified
     to a polygon (original image pixel coordinates, same space the editor
     draws in).
  4. Doors: for every pair of nearby rooms, look at the wall band between
     them; where the band has NO wall pixels there is a passage — fit a
     rotated rect to those gap pixels and take its long axis as the door.

Entry point:
    rooms, doors = detect_rooms_and_doors(gray_image)
    rooms: [ [(x, y), ...] per room ]      doors: [ ((x1, y1), (x2, y2)) ]
"""

import math

import cv2
import numpy as np


def _exterior_mask(dark):
    """White area connected to the image border in the ORIGINAL drawing.

    Window/door symbols block the flood, so this is the true outside of the
    building. Thin symbol lines can have 1-2 px anti-aliasing gaps that
    would let the flood leak inside, so the strokes are bridged (dilated)
    a little before flooding — escalating if a leak is still detected.
    """
    h, w = dark.shape
    for bridge in (3, 5, 9, 13):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge))
        walls_b = cv2.dilate(dark, kernel)
        free = cv2.bitwise_not(walls_b)
        flood = free.copy()
        ffmask = np.zeros((h + 2, w + 2), np.uint8)
        for x in range(0, w, 8):
            for y in (0, h - 1):
                if flood[y, x]:
                    cv2.floodFill(flood, ffmask, (x, y), 0)
        for y in range(0, h, 8):
            for x in (0, w - 1):
                if flood[y, x]:
                    cv2.floodFill(flood, ffmask, (x, y), 0)
        exterior = cv2.bitwise_and(free, cv2.bitwise_not(flood))
        # a leak floods the inside too: exterior would swallow most white
        if (exterior > 0).sum() < 0.62 * (free > 0).sum():
            return exterior
    return exterior


def _wall_mask(gray):
    """Wall mask: thick strokes + everything dark near the building exterior.

    Walls are the THICKEST strokes in a plan: seed at thick-stroke cores
    (distance transform) and grow back to full stroke width, which drops
    thin strokes (door arcs/leaves, furniture) even when they are CONNECTED
    to the wall network. The outer shell then gets the thin strokes near the
    exterior added back (window symbols, entrance arcs), so the building
    stays sealed while interior doorways stay open.

    Returns (walls, dark, exterior).
    """
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dt = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    max_dt = float(dt.max())
    exterior = _exterior_mask(dark)
    if max_dt <= 1.0:
        return dark, dark, exterior
    thr = max(1.2, 0.45 * max_dt)
    seed = (dt >= thr).astype(np.uint8) * 255
    r = int(math.ceil(max_dt)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    walls = cv2.bitwise_and(cv2.dilate(seed, kernel), dark)
    # seal the outer shell: dark strokes within the outer wall band
    shell_r = int(math.ceil(2 * max_dt)) + 3
    shell_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * shell_r + 1, 2 * shell_r + 1))
    shell = cv2.bitwise_and(dark, cv2.dilate(exterior, shell_kernel))
    walls = cv2.bitwise_or(walls, shell)
    return walls, dark, exterior


def _interior_components(walls, close_k, min_area, exclude=None):
    """Label map of room regions after sealing door gaps and dropping outside."""
    h, w = walls.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    sealed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)
    interior = cv2.bitwise_not(sealed)
    if exclude is not None:
        interior = cv2.bitwise_and(interior, cv2.bitwise_not(exclude))

    # safety: anything still connected to the border is outside the building
    flood = interior.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 8):
        for y in (0, h - 1):
            if flood[y, x]:
                cv2.floodFill(flood, ffmask, (x, y), 0)
    for y in range(0, h, 8):
        for x in (0, w - 1):
            if flood[y, x]:
                cv2.floodFill(flood, ffmask, (x, y), 0)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(flood, 8)
    room_ids = [i for i in range(1, num)
                if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return labels, room_ids, stats


def _orthogonalize(poly):
    """Snap a mostly-rectilinear polygon to clean horizontal/vertical edges.

    Floor plans are rectilinear; contour simplification leaves small
    diagonal cut-corners that would become skewed walls in 3D. Each edge is
    classified h/v, consecutive same-orientation edges merge (length-
    weighted), and vertices are rebuilt from the line intersections.
    Returns the original polygon unchanged if it isn't rectilinear enough.
    """
    pts = [np.array(p, dtype=float) for p in poly]
    n = len(pts)
    per = axis_per = 0.0
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i]
        L = math.hypot(d[0], d[1])
        per += L
        ang = abs(math.degrees(math.atan2(d[1], d[0]))) % 90.0
        if ang < 15.0 or ang > 75.0:
            axis_per += L
    if per <= 0 or axis_per / per < 0.80:
        return poly

    edges = []
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        d = q - p
        L = math.hypot(d[0], d[1])
        if L < 1e-6:
            continue
        if abs(d[0]) >= abs(d[1]):
            edges.append(["h", (p[1] + q[1]) / 2, L])
        else:
            edges.append(["v", (p[0] + q[0]) / 2, L])

    merged = []
    for e in edges:
        if merged and merged[-1][0] == e[0]:
            t, c, L = merged[-1]
            merged[-1] = [t, (c * L + e[1] * e[2]) / (L + e[2]), L + e[2]]
        else:
            merged.append(list(e))
    while len(merged) > 2 and merged[0][0] == merged[-1][0]:
        t, c, L = merged.pop()
        t0, c0, L0 = merged[0]
        merged[0] = [t0, (c0 * L0 + c * L) / (L0 + L), L0 + L]
    if len(merged) < 4 or len(merged) % 2:
        return poly

    out = []
    m = len(merged)
    for i in range(m):
        a, b = merged[i], merged[(i + 1) % m]
        if a[0] == b[0]:
            return poly
        x = b[1] if b[0] == "v" else a[1]
        y = b[1] if b[0] == "h" else a[1]
        out.append((int(round(x)), int(round(y))))
    return out


def _room_polygon(comp_mask, eps_frac=0.012):
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    eps = eps_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    poly = [(int(p[0][0]), int(p[0][1])) for p in approx]
    return poly if len(poly) >= 3 else None


def _detect_doors(walls, labels, room_ids, min_dim):
    """Door segments between adjacent rooms.

    A doorway is a gap in the wall network: sealing the walls with a SMALL
    kernel plugs the gap without swallowing corridors or corners. Each plug
    that (a) is shaped like a doorway (thin across the wall, door-sized
    along it) and (b) touches exactly two different rooms is a door.
    Multiple sealing scales are tried so narrow and wide doors both plug.
    """
    from itertools import combinations

    h, w = walls.shape
    dt = cv2.distanceTransform(walls, cv2.DIST_L2, 5)
    wall_t = max(2.0, 2.0 * float(dt.max()))          # wall stroke thickness
    adj_k = int(max(9, wall_t + 6)) | 1
    adj_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (adj_k, adj_k))
    room_label_set = set(room_ids)

    # rooms dilated a little past the wall thickness (for plug refinement)
    dil = {}
    for i in room_ids:
        comp = (labels == i).astype(np.uint8) * 255
        dil[i] = cv2.dilate(comp, adj_kernel)

    doors = []
    found = []      # (pair, cx, cy, ux, uy, half_len) — for overlap dedupe

    def is_duplicate(pair, cx, cy, ux, uy, half):
        for p, fx, fy, fux, fuy, fhalf in found:
            if p != pair:
                continue
            dx, dy = cx - fx, cy - fy
            along = abs(dx * fux + dy * fuy)
            across = abs(-dx * fuy + dy * fux)
            if across < wall_t * 3 + 8 and along < (half + fhalf) * 0.75:
                return True
        return False

    def gap_span(cx, cy, ux, uy, fallback_half):
        """True doorway half-extents: walk the wall line to the jambs."""
        spans = []
        for sgn in (1, -1):
            d, hit = fallback_half, False
            for t in range(1, int(min_dim * 0.3)):
                x = int(round(cx + sgn * ux * t))
                y = int(round(cy + sgn * uy * t))
                if not (0 <= x < w and 0 <= y < h):
                    break
                if walls[y, x]:
                    d, hit = t, True
                    break
            spans.append(d if hit else fallback_half)
        return spans

    def first_room_along(cx, cy, nx, ny, max_d):
        """Walk from (cx, cy) along (nx, ny); first room label reached, or
        None if a wall blocks the way first."""
        for t in range(1, int(max_d)):
            x, y = int(round(cx + nx * t)), int(round(cy + ny * t))
            if not (0 <= x < w and 0 <= y < h):
                return None
            if walls[y, x]:
                return None
            lab = labels[y, x]
            if lab in room_label_set:
                return lab
        return None

    for frac in (0.035, 0.05, 0.07, 0.09):
        close_s = max(7, int(round(min_dim * frac)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_s, close_s))
        sealed = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)
        plugs = cv2.bitwise_and(sealed, cv2.bitwise_not(walls))
        num, plabels, pstats, _ = cv2.connectedComponentsWithStats(plugs, 8)
        for g in range(1, num):
            if pstats[g, cv2.CC_STAT_AREA] < 15:
                continue
            comp = (plabels == g).astype(np.uint8) * 255
            touching = [i for i in room_ids
                        if cv2.countNonZero(cv2.bitwise_and(comp, dil[i]))]
            if len(touching) < 2:
                continue
            for a, b in combinations(touching, 2):
                # part of the plug that lies BETWEEN rooms a and b
                refined = cv2.bitwise_and(comp,
                                          cv2.bitwise_and(dil[a], dil[b]))
                if cv2.countNonZero(refined) < 12:
                    continue
                pts = cv2.findNonZero(refined)
                (cx, cy), (rw, rh), ang = cv2.minAreaRect(
                    pts.astype(np.float32))
                long_len, short_len = max(rw, rh), min(rw, rh)
                if short_len > max(3.0 * wall_t, 14):
                    continue
                if long_len < max(2 * wall_t, 10) or long_len > min_dim * 0.22:
                    continue
                theta = math.radians(ang if rw >= rh else ang + 90.0)
                ux, uy = math.cos(theta), math.sin(theta)
                nx, ny = -uy, ux                     # across the wall
                # a REAL doorway: walking straight out of the plug on both
                # sides reaches room a and room b without hitting any wall
                reach = close_s + wall_t * 2 + 8
                lab_pos = first_room_along(cx, cy, nx, ny, reach)
                lab_neg = first_room_along(cx, cy, -nx, -ny, reach)
                if {lab_pos, lab_neg} != {a, b}:
                    continue
                # re-measure the actual gap between jambs (the plug itself
                # is clipped by the eroded room labels)
                d_fwd, d_back = gap_span(cx, cy, ux, uy, long_len / 2)
                gap_len = d_back + d_fwd
                if gap_len > min_dim * 0.35:          # runaway walk: keep rect
                    d_back = d_fwd = long_len / 2
                    gap_len = long_len
                ncx = cx + ux * (d_fwd - d_back) / 2
                ncy = cy + uy * (d_fwd - d_back) / 2
                if is_duplicate((a, b), ncx, ncy, ux, uy, gap_len / 2):
                    continue
                found.append(((a, b), ncx, ncy, ux, uy, gap_len / 2))
                doors.append(((int(ncx - ux * gap_len / 2),
                               int(ncy - uy * gap_len / 2)),
                              (int(ncx + ux * gap_len / 2),
                               int(ncy + uy * gap_len / 2))))
    return doors


def detect_rooms_and_doors(gray):
    """Detect room polygons + door segments from a floor-plan image.

    Tries several door-sealing kernel sizes and keeps the segmentation that
    produces the most valid rooms (door gaps sealed = rooms separate; a too
    large kernel destroys narrow rooms, giving fewer).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    walls, dark, exterior = _wall_mask(gray)
    # exclude only the outside: thin furniture lines must NOT split rooms
    exclude = exterior
    min_area = 0.004 * h * w

    best = None
    for frac in (0.035, 0.05, 0.065, 0.08, 0.10):
        close_k = max(7, int(round(min_dim * frac)) | 1)
        labels, room_ids, stats = _interior_components(
            walls, close_k, min_area, exclude)
        if not room_ids:
            continue
        # coverage sanity: rooms should fill a decent part of the building
        area_frac = sum(stats[i, cv2.CC_STAT_AREA] for i in room_ids) / (h * w)
        if area_frac < 0.10:
            continue
        if best is None or len(room_ids) > best[0]:
            best = (len(room_ids), labels, room_ids, close_k)

    if best is None:
        return [], []

    _, labels, room_ids, close_k = best

    # stable room order: top-to-bottom, then left-to-right
    def _key(i):
        ys, xs = np.where(labels == i)
        return (ys.mean() // (h / 4), xs.mean())
    room_ids = sorted(room_ids, key=_key)

    rooms = []
    kept_ids = []
    for i in room_ids:
        comp = (labels == i).astype(np.uint8) * 255
        poly = _room_polygon(comp)
        if poly:
            rooms.append(_orthogonalize(poly))
            kept_ids.append(i)

    doors = _detect_doors(walls, labels, kept_ids, min_dim)
    print(f"[DETECT] rooms={len(rooms)} doors={len(doors)} (close_k={close_k})")
    return rooms, doors


if __name__ == "__main__":
    import os
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "plan", "1.jpg")
    img = cv2.imread(path)
    rooms, doors = detect_rooms_and_doors(img)

    # overlay for visual verification
    vis = img.copy()
    colors = [(0, 170, 0), (200, 60, 0), (0, 80, 220), (160, 0, 160),
              (0, 150, 150), (180, 120, 0), (60, 60, 220), (120, 180, 0)]
    for k, poly in enumerate(rooms):
        cv2.polylines(vis, [np.array(poly, np.int32)], True,
                      colors[k % len(colors)], 3)
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(vis, f"R{k+1}", (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colors[k % len(colors)], 2)
    for p1, p2 in doors:
        cv2.line(vis, p1, p2, (0, 0, 255), 5)
    out = os.path.join(os.path.dirname(path), "_autodetect_overlay.png")
    cv2.imwrite(out, vis)
    print(f"[DETECT] overlay saved: {out}")
