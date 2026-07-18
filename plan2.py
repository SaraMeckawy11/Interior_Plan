"""
Tkinter plan editor (extended, non-destructive additions)

Added:
- Per-room storage (Room 1, Room 2, ...)
- Room viewer window
- Per-room 3D view (same pipeline as Segment & 3D)
- 3D options dialog with configurable rendering
"""

import os, math, json

# ---------------------------------------------------------------------------
# Auto-relaunch under the project venv. The app's dependencies (OpenCV, torch,
# OpenCV, Open3D, and the gallery dependencies live in the project environment,
# not in the system Python.
# If this file is started with any other interpreter (e.g. VS Code's default
# "python"), re-run it once under the venv so the Run button just works.
# ---------------------------------------------------------------------------
import sys as _sys
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_CANDIDATES = [
    os.environ.get("INTERIOR_PLAN_PYTHON", ""),
    os.path.join(_PROJECT_DIR, ".venv", "Scripts", "python.exe"),
    os.path.join(_PROJECT_DIR, "hf_cache", "venv", "Scripts", "python.exe"),
    r"C:\Sara\Interior_design\hf_cache\venv\Scripts\python.exe",
    os.path.join(_PROJECT_DIR, ".triposr_venv", "Scripts", "python.exe"),
    r"C:\SIA\Interior_design\hf_cache\venv\Scripts\python.exe",
]
_VENV_PY = next((path for path in _VENV_CANDIDATES
                 if path and os.path.isfile(path)), "")
if _VENV_PY and os.path.normcase(os.path.realpath(_sys.executable)) \
        != os.path.normcase(os.path.realpath(_VENV_PY)):
    import importlib.util as _ilu
    if any(_ilu.find_spec(name) is None
           for name in ("cv2", "open3d", "shapely")):
        import subprocess as _sp
        print(f"[BOOT] Relaunching under project venv:\n       {_VENV_PY}")
        raise SystemExit(_sp.call([_VENV_PY, os.path.abspath(__file__), *_sys.argv[1:]]))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from io import BytesIO
import threading

from plan3d import build_3d_apartment_and_walk
from room_gallery import (
    DesignGallery, save_screenshot_entry, load_metadata, save_metadata,
    DesignEntry, SCREENSHOTS_DIR, AI_DESIGNS_DIR, COLORS as GALLERY_COLORS,
    update_design_entry
)

# ---------------- CONFIG ----------------
IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan", "1.jpg")
OUT_DIR = os.path.join(os.getcwd(), "seg_output")
os.makedirs(OUT_DIR, exist_ok=True)

CANVAS_MAX_W = 1000
CANVAS_MAX_H = 800
SNAP_RADIUS = 10
POLYGON_CLOSE_DIST = 20
DOOR_WINDOW_THICKNESS = 10

# ---------- STATE ----------
class AppState:
    def __init__(self):
        self.image = None
        self.orig_img_cv = None
        self.orig_img_color = None
        self.scale = 1.0
        self.offset = (0, 0)

        self.outer = []
        self.outer_closed = False
        self.inners = []
        self.current_inner = []

        self.doors = []
        self.windows = []  # (p1, p2, type)

        self.mode = "outer"
        self.window_type = "normal"
        self.undo_stack = []

        self.room_segments = {}  # "Room 1" → polygon

state = AppState()

# ---------- PLAN LAYOUT PERSISTENCE (draw once, keep forever) ----------
LAYOUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "plan_layout.json")


def save_layout(allow_empty=False):
    """Autosave the drawn/detected plan so it never has to be redrawn.

    Guard against data loss: never overwrite a saved plan with an EMPTY one
    unless the user explicitly cleared it (Reset passes allow_empty=True).
    This stops a stray action, or a second app instance that started before
    the layout loaded, from wiping the saved plan.
    """
    if not (state.inners or state.doors or state.windows) and not allow_empty:
        return
    try:
        data = {
            "image_path": os.path.relpath(IMAGE_PATH, _PROJECT_DIR).replace("\\", "/"),
            "rooms": [[list(map(int, p)) for p in poly] for poly in state.inners],
            "doors": [[list(map(int, a)), list(map(int, b))]
                      for a, b in state.doors],
            "windows": [[list(map(int, w[0])), list(map(int, w[1])), w[2]]
                        for w in state.windows],
        }
        with open(LAYOUT_FILE, "w") as f:
            json.dump(data, f, indent=1)
        try:
            lbl_saved.config(text="✓ Autosaved")
            root.after(1600, lambda: lbl_saved.config(text=""))
        except Exception:
            pass   # UI not built yet
    except Exception as e:
        print(f"[LAYOUT] Save failed: {e}")


def load_layout():
    """Restore the last saved plan layout (rooms, doors, windows)."""
    if not os.path.exists(LAYOUT_FILE):
        return False
    try:
        with open(LAYOUT_FILE) as f:
            data = json.load(f)
        saved_image = data.get("image_path", "")
        if not os.path.isabs(saved_image):
            saved_image = os.path.join(_PROJECT_DIR, saved_image)
        if os.path.normcase(os.path.realpath(saved_image)) != \
                os.path.normcase(os.path.realpath(IMAGE_PATH)):
            return False   # layout belongs to a different plan image
        state.inners = [[tuple(p) for p in poly] for poly in data.get("rooms", [])]
        state.room_segments = {f"Room {i + 1}": poly
                               for i, poly in enumerate(state.inners)}
        state.doors = [(tuple(a), tuple(b)) for a, b in data.get("doors", [])]
        state.windows = [(tuple(a), tuple(b), t)
                         for a, b, t in data.get("windows", [])]
        print(f"[LAYOUT] Restored {len(state.inners)} rooms, "
              f"{len(state.doors)} doors, {len(state.windows)} windows")
        return bool(state.inners or state.doors or state.windows)
    except Exception as e:
        print(f"[LAYOUT] Load failed: {e}")
        return False


# ---------- UTIL ----------
def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, CANVAS_MAX_W / w, CANVAS_MAX_H / h)
    state.scale = scale
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    state.image = img
    state.orig_img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    state.orig_img_color = cv2.imread(path, cv2.IMREAD_COLOR)

def img2canvas(p):
    return int(p[0] * state.scale), int(p[1] * state.scale)

def canvas2img(p):
    return int(p[0] / state.scale), int(p[1] / state.scale)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def polygon_to_mask(poly, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    return mask

def extract_room_image(poly):
    h, w = state.orig_img_color.shape[:2]
    mask = polygon_to_mask(poly, (h, w))
    masked = cv2.bitwise_and(state.orig_img_color, state.orig_img_color, mask=mask)
    x, y, bw, bh = cv2.boundingRect(np.array(poly, np.int32))
    crop = masked[y:y+bh, x:x+bw]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# ---------- MODERN UI DESIGN ----------
root = tk.Tk()
root.title("Floor Plan Studio")
root.geometry("1440x920")
root.minsize(1180, 780)

# Design tokens (light theme)
BG_COLOR = "#eef1f5"          # app background
SURFACE = "#ffffff"           # bars / cards
SURFACE_ALT = "#f8fafc"       # secondary bar
BORDER = "#e2e8f0"
TEXT_COLOR = "#1f2937"
TEXT_MUTED = "#64748b"
PRIMARY_COLOR = "#2563eb"; PRIMARY_HOVER = "#1d4ed8"
SECONDARY_COLOR = "#eef1f5"; SECONDARY_HOVER = "#e2e8f0"
SUCCESS_COLOR = "#16a34a"; SUCCESS_HOVER = "#15803d"
DANGER_COLOR = "#dc2626"; DANGER_HOVER = "#b91c1c"
ACCENT_COLOR = "#e11d48"; ACCENT_HOVER = "#be123c"
PURPLE_COLOR = "#7c3aed"; PURPLE_HOVER = "#6d28d9"

# Canvas drawing palette
ROOM_COLORS = ["#16a34a", "#2563eb", "#d97706", "#7c3aed",
               "#0d9488", "#db2777", "#65a30d", "#dc2626"]
DOOR_COLOR = "#f97316"
WINDOW_COLOR = "#0ea5e9"

root.configure(bg=BG_COLOR)


def make_btn(parent, text, bg, hover, fg="white", command=None, font_size=10, **kw):
    """Flat button with hover feedback (consistent across the app)."""
    btn = tk.Button(parent, text=text, font=("Segoe UI", font_size, "bold"),
                    bg=bg, fg=fg, relief=tk.FLAT, bd=0, padx=14, pady=8,
                    cursor="hand2", activebackground=hover, activeforeground=fg,
                    command=command, **kw)
    btn._base_bg = bg
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=btn._base_bg))
    return btn


# ===== Header row 1: identity + primary actions =====
header = tk.Frame(root, bg=SURFACE)
header.pack(side=tk.TOP, fill=tk.X)

title_box = tk.Frame(header, bg=SURFACE)
title_box.pack(side=tk.LEFT, padx=(20, 10), pady=10)
tk.Label(title_box, text="🏠 Floor Plan Studio", font=("Segoe UI", 16, "bold"),
         bg=SURFACE, fg=TEXT_COLOR).pack(anchor="w")
tk.Label(title_box, text="Draw rooms → choose finishes → explore a realistic 3D interior",
         font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_MUTED).pack(anchor="w")

action_frame = tk.Frame(header, bg=SURFACE)
action_frame.pack(side=tk.RIGHT, padx=20, pady=10)

# ===== Header row 2: drawing workflow =====
workflow = tk.Frame(root, bg=SURFACE_ALT)
workflow.pack(side=tk.TOP, fill=tk.X)
tk.Frame(root, bg=BORDER, height=1).pack(side=tk.TOP, fill=tk.X)

mode_frame = tk.Frame(workflow, bg=SURFACE_ALT)
mode_frame.pack(side=tk.LEFT, padx=20, pady=8)

tk.Label(mode_frame, text="DRAW", font=("Segoe UI", 9, "bold"),
         bg=SURFACE_ALT, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(0, 12))

mode_buttons = {}
MODE_LABELS = {"inner": "①  Rooms", "door": "②  Doors", "window": "③  Windows"}
MODE_HINTS = {
    "inner": "Click the corners of a room — click the first point again (or press Enter) to close it",
    "door": "Click the two end points of a door opening on a wall — Esc cancels",
    "window": "Click the two end points of a window on a wall — Esc cancels",
}


def create_mode_btn(mode_val):
    btn = make_btn(mode_frame, MODE_LABELS[mode_val], SECONDARY_COLOR,
                   SECONDARY_HOVER, fg=TEXT_COLOR,
                   command=lambda: set_mode(mode_val))
    btn.config(padx=16, pady=6)
    btn.pack(side=tk.LEFT, padx=3)
    mode_buttons[mode_val] = btn
    return btn


for _m in ("inner", "door", "window"):
    create_mode_btn(_m)

tk.Frame(mode_frame, bg=BORDER, width=1, height=26).pack(side=tk.LEFT, padx=14)

edit_frame = tk.Frame(workflow, bg=SURFACE_ALT)
edit_frame.pack(side=tk.LEFT, pady=8)
# Undo / Reset buttons are created after their functions are defined (see bottom)

hint_lbl = tk.Label(workflow,
                    text="Shortcuts:  1 / 2 / 3 switch tools  •  Ctrl+Z undo  •  Esc cancel  •  Enter close room",
                    font=("Segoe UI", 9), bg=SURFACE_ALT, fg=TEXT_MUTED)
hint_lbl.pack(side=tk.RIGHT, padx=20)

# ===== Status bar =====
status_frame = tk.Frame(root, bg=SURFACE)
status_frame.pack(side=tk.BOTTOM, fill=tk.X)
tk.Frame(root, bg=BORDER, height=1).pack(side=tk.BOTTOM, fill=tk.X)

lbl_mode = tk.Label(status_frame, text="", font=("Segoe UI", 10, "bold"),
                    bg=SURFACE, fg=PRIMARY_COLOR)
lbl_mode.pack(side=tk.LEFT, padx=(20, 8), pady=8)

lbl_info = tk.Label(status_frame, text="", font=("Segoe UI", 10),
                    bg=SURFACE, fg=TEXT_MUTED)
lbl_info.pack(side=tk.LEFT, padx=8, pady=8)

lbl_saved = tk.Label(status_frame, text="", font=("Segoe UI", 9),
                     bg=SURFACE, fg=SUCCESS_COLOR)
lbl_saved.pack(side=tk.RIGHT, padx=20, pady=8)

lbl_counts = tk.Label(status_frame, text="", font=("Segoe UI", 10),
                      bg=SURFACE, fg=TEXT_COLOR)
lbl_counts.pack(side=tk.RIGHT, padx=12, pady=8)

# ===== Canvas =====
canvas = tk.Canvas(root, width=CANVAS_MAX_W, height=700, bg="#d7dde5",
                   highlightthickness=0, cursor="crosshair")
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def set_mode(m):
    state.mode = m
    if m == "window":
        state.window_type = "normal"
    lbl_mode.config(text=MODE_LABELS[m].replace("  ", " "))
    lbl_info.config(text=MODE_HINTS[m], fg=TEXT_MUTED)
    for mode_val, btn in mode_buttons.items():
        if mode_val == m:
            btn._base_bg = PRIMARY_COLOR
            btn.config(bg=PRIMARY_COLOR, fg="white")
        else:
            btn._base_bg = SECONDARY_COLOR
            btn.config(bg=SECONDARY_COLOR, fg=TEXT_COLOR)
    redraw()

# ---------- UNDO / RESET / CANCEL ----------
def undo():
    if not state.undo_stack:
        lbl_info.config(text="Nothing to undo", fg=TEXT_MUTED)
        return

    action, _data = state.undo_stack.pop()

    if action == "inner_point" and state.current_inner:
        state.current_inner.pop()
    elif action == "inner_finish" and state.inners:
        last = state.inners.pop()
        state.current_inner = last
        state.room_segments.pop(f"Room {len(state.inners) + 1}", None)
    elif action == "door" and state.doors:
        state.doors.pop()
    elif action == "window" and state.windows:
        state.windows.pop()

    save_layout()
    redraw()


def cancel_current(event=None):
    """Esc / right-click: abandon the shape currently being drawn."""
    global mouse_start
    changed = False
    if mouse_start is not None:
        mouse_start = None
        changed = True
    if state.current_inner:
        while state.undo_stack and state.undo_stack[-1][0] == "inner_point":
            state.undo_stack.pop()
        state.current_inner = []
        changed = True
    canvas.delete("preview")
    if changed:
        lbl_info.config(text="Cancelled", fg=TEXT_MUTED)
        redraw()


def finish_room(event=None):
    """Close the room polygon currently being drawn (Enter or first-point click)."""
    if len(state.current_inner) < 3:
        return
    poly = state.current_inner[:]
    state.inners.append(poly)
    room_id = len(state.inners)
    state.room_segments[f"Room {room_id}"] = poly
    state.undo_stack.append(("inner_finish", None))
    state.current_inner = []
    lbl_info.config(text=f"✓ Room {room_id} created — draw another, "
                         f"or press 2 to add doors", fg=SUCCESS_COLOR)
    save_layout()
    redraw()


def reset_all():
    """Clear the whole plan (with confirmation) — including the saved layout."""
    global mouse_start
    if not (state.inners or state.doors or state.windows or state.current_inner):
        lbl_info.config(text="Nothing to reset", fg=TEXT_MUTED)
        return
    if not messagebox.askyesno(
            "Reset Plan",
            "Clear ALL rooms, doors and windows?\n\n"
            "The autosaved layout will be cleared too."):
        return
    state.inners = []
    state.room_segments = {}
    state.doors = []
    state.windows = []
    state.current_inner = []
    state.undo_stack = []
    mouse_start = None
    save_layout(allow_empty=True)   # explicit clear may write an empty file
    redraw()
    set_mode("inner")
    lbl_info.config(text="Plan cleared — draw the first room", fg=TEXT_MUTED)

# ---------- DRAW ----------
def redraw():
    canvas.delete("all")
    if not state.image:
        load_image(IMAGE_PATH)

    imtk = ImageTk.PhotoImage(state.image)
    canvas.image = imtk
    canvas.create_image(0, 0, anchor="nw", image=imtk)

    # finished rooms: tinted fill, colored outline, centered label
    for idx, poly in enumerate(state.inners):
        col = ROOM_COLORS[idx % len(ROOM_COLORS)]
        pts = [c for p in poly for c in img2canvas(p)]
        if len(poly) > 2:
            canvas.create_polygon(*pts, fill=col, stipple="gray12",
                                  outline=col, width=2)
            cx = sum(img2canvas(p)[0] for p in poly) / len(poly)
            cy = sum(img2canvas(p)[1] for p in poly) / len(poly)
            canvas.create_text(cx, cy, text=f"Room {idx + 1}",
                               font=("Segoe UI", 11, "bold"), fill=col)

    # room being drawn: polyline + vertex handles (first point emphasized)
    cur = [img2canvas(p) for p in state.current_inner]
    if len(cur) > 1:
        canvas.create_line(*sum(cur, ()), fill=PRIMARY_COLOR, width=2)
    for i, (x, y) in enumerate(cur):
        r = 6 if i == 0 else 4
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="white",
                           outline=PRIMARY_COLOR, width=2)

    for p1, p2 in state.doors:
        canvas.create_line(*img2canvas(p1), *img2canvas(p2), fill=DOOR_COLOR,
                           width=7, capstyle=tk.ROUND)

    for p1, p2, t in state.windows:
        canvas.create_line(*img2canvas(p1), *img2canvas(p2), fill=WINDOW_COLOR,
                           width=7, capstyle=tk.ROUND)

    lbl_counts.config(text=f"Rooms {len(state.inners)}   •   "
                           f"Doors {len(state.doors)}   •   "
                           f"Windows {len(state.windows)}")

# ---------- INPUT ----------
mouse_start = None
def click(e):
    global mouse_start
    pt = canvas2img((e.x, e.y))

    if state.mode == "inner":
        if (len(state.current_inner) >= 3 and
                dist(img2canvas(state.current_inner[0]), (e.x, e.y)) < POLYGON_CLOSE_DIST):
            finish_room()
        else:
            state.current_inner.append(pt)
            state.undo_stack.append(("inner_point", None))
            n = len(state.current_inner)
            if n < 3:
                lbl_info.config(text=f"Room outline: {n} point{'s' if n > 1 else ''}",
                                fg=TEXT_MUTED)
            else:
                lbl_info.config(text=f"Room outline: {n} points — click the first "
                                     f"point (or press Enter) to close",
                                fg=PRIMARY_COLOR)
            redraw()

    elif state.mode in ("door", "window"):
        if mouse_start is None:
            mouse_start = pt
            what = "door" if state.mode == "door" else "window"
            lbl_info.config(text=f"Now click the other end of the {what}",
                            fg=PRIMARY_COLOR)
        else:
            if state.mode == "door":
                state.doors.append((mouse_start, pt))
                state.undo_stack.append(("door", None))
                lbl_info.config(text=f"✓ Door {len(state.doors)} added — add more, "
                                     f"or press 3 for windows", fg=SUCCESS_COLOR)
            else:
                state.windows.append((mouse_start, pt, state.window_type))
                state.undo_stack.append(("window", None))
                lbl_info.config(text=f"✓ Window {len(state.windows)} added — "
                                     f"ready for the 3D Walkthrough", fg=SUCCESS_COLOR)
            mouse_start = None
            save_layout()
            redraw()

canvas.bind("<Button-1>", click)

# Live previews while drawing
def on_motion(e):
    canvas.delete("preview")
    if state.mode == "inner" and state.current_inner:
        x0, y0 = img2canvas(state.current_inner[-1])
        canvas.create_line(x0, y0, e.x, e.y, fill=PRIMARY_COLOR, width=2,
                           dash=(5, 4), tags="preview")
        if len(state.current_inner) >= 3:
            fx, fy = img2canvas(state.current_inner[0])
            if dist((fx, fy), (e.x, e.y)) < POLYGON_CLOSE_DIST:
                canvas.create_oval(fx - 11, fy - 11, fx + 11, fy + 11,
                                   outline=SUCCESS_COLOR, width=3, tags="preview")
    elif state.mode in ("door", "window") and mouse_start is not None:
        color = DOOR_COLOR if state.mode == "door" else WINDOW_COLOR
        canvas.create_line(*img2canvas(mouse_start), e.x, e.y, fill=color,
                           width=7, dash=(6, 4), capstyle=tk.ROUND, tags="preview")

canvas.bind("<Motion>", on_motion)
canvas.bind("<Button-3>", cancel_current)


# ---------- AI DESIGN GENERATOR FUNCTION ----------
def generate_ai_interior_design(entry: DesignEntry):
    """Generate AI interior design from a screenshot, using actual floor plan layout data"""
    try:
        print(f"[INFO] Starting AI generation for {entry.room_name}...")
        print(f"[INFO] Screenshot path: {entry.screenshot_path}")
        print(f"[INFO] Room type: {entry.room_type}, Style: {entry.style}")
        print(f"[INFO] Layout info: has_doors={entry.has_doors}, has_windows={entry.has_windows}")
        print(f"[INFO] Visible openings: {entry.visible_openings}")
        
        # Load the screenshot
        if not os.path.exists(entry.screenshot_path):
            print(f"[ERROR] Screenshot file not found: {entry.screenshot_path}")
            return None
            
        screenshot = cv2.imread(entry.screenshot_path)
        if screenshot is None:
            print(f"[ERROR] Could not read screenshot image")
            return None
        
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        print(f"[INFO] Image loaded: {screenshot_rgb.shape}")
        
        # Create mask (full image)
        h, w = screenshot_rgb.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        print(f"[INFO] Calling AI model with layout hints...")
        
        # Generate furnished room using planAI with actual layout data
        # Pass has_doors/has_windows from the floor plan editor so the AI
        # doesn't try to auto-detect from the 3D render (which gives wrong results)
        # Also pass visible_openings (with 3D corners) + camera_params so the AI
        # can project door/window positions directly onto the Canny conditioning image
        import planAI as ai_furnisher

        results = ai_furnisher.generate_furnished_plan(
            base_rgb=screenshot_rgb,
            room_masks=[mask],
            doors=[],
            windows=[],
            room_types=[entry.room_type],
            px_per_m=100,
            has_doors=entry.has_doors,
            has_windows=entry.has_windows,
            visible_openings=entry.visible_openings,
            camera_params=entry.camera_params,
            styles=[entry.style]
        )
        
        print(f"[INFO] AI generation complete, results: {len(results) if results else 0}")
        
        if results and len(results) > 0:
            furnished_img = results[0]["furnished"]
            
            # Save AI generated image
            ai_filename = f"{entry.design_id}_ai_design.png"
            ai_path = os.path.join(AI_DESIGNS_DIR, ai_filename)
            furnished_img.save(ai_path)
            
            print(f"[OK] AI design saved to: {ai_path}")
            return ai_path
        else:
            print("[ERROR] No results returned from AI generation")
            return None
            
    except Exception as e:
        print(f"[ERROR] AI generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------- DESIGN GALLERY ----------
def open_design_gallery():
    """Open the design gallery window"""
    gallery = DesignGallery(root, ai_generator_callback=generate_ai_interior_design)


# ---------- ROOM VIEWER ----------
def open_room_viewer():
    if not state.room_segments:
        messagebox.showinfo("No rooms", "No rooms defined yet")
        return

    viewer = tk.Toplevel(root)
    viewer.title("Room Viewer")
    viewer.geometry(f"{CANVAS_MAX_W+220}x{CANVAS_MAX_H}")

    sidebar = tk.Frame(viewer, width=220)
    sidebar.pack(side=tk.LEFT, fill=tk.Y)

    view_canvas = tk.Canvas(viewer, width=CANVAS_MAX_W, height=CANVAS_MAX_H, bg="#ddd")
    view_canvas.pack(side=tk.RIGHT)

    def show_room(name):
        view_canvas.delete("all")
        img = extract_room_image(state.room_segments[name])

        w, h = img.size
        scale = min(CANVAS_MAX_W / w, CANVAS_MAX_H / h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        imtk = ImageTk.PhotoImage(img)
        view_canvas.image = imtk
        view_canvas.create_image(0, 0, anchor="nw", image=imtk)

canvas.bind("<Button-1>", click)

# ---------- ROOM PERSPECTIVE VIEW WITH SCREENSHOT ----------
def take_room_perspective_screenshot(room_poly, room_name="Room"):
    """Let user select which wall to remove for perspective view with screenshot capability"""
    import open3d as o3d
    from plan3d import (
        px_to_m, polygon_centroid_3d, floor_mesh, build_walls_from_edges,
        merge_edges, WALL_HEIGHT, EYE_HEIGHT, MOVE_SPEED, MIN_DOOR_WIDTH, WINDOW_WIDTH
    )
    from shapely.geometry import LineString, Point
    
    room_m = [px_to_m(p, 100) for p in room_poly]
    
    # Build edges for this room
    raw_edges = []
    for i in range(len(room_m)):
        p1, p2 = room_m[i], room_m[(i + 1) % len(room_m)]
        if np.linalg.norm(p2 - p1) < 0.1:
            continue
        raw_edges.append({
            "p1": p1,
            "p2": p2,
            "line": LineString([p1, p2]),
            "length": LineString([p1, p2]).length,
            "openings": [],
            "loop_type": "inner",
            "wall_id": i
        })
    
    edges = merge_edges(raw_edges)
    
    # Assign doors to this room's edges
    for d1, d2 in state.doors:
        a, b = px_to_m(d1, 100), px_to_m(d2, 100)
        door_line = LineString([a, b])
        edge = min(edges, key=lambda e: e["line"].distance(door_line))
        
        mid = (a + b) / 2
        dir_vec = np.array(edge["p2"]) - np.array(edge["p1"])
        dir_vec /= np.linalg.norm(dir_vec)
        half = MIN_DOOR_WIDTH / 2
        p0, p1 = mid - dir_vec * half, mid + dir_vec * half
        t0 = edge["line"].project(Point(p0)) / edge["length"]
        t1 = edge["line"].project(Point(p1)) / edge["length"]
        edge["openings"].append(("door", min(t0, t1), max(t0, t1)))
    
    # Assign windows to this room's edges
    for w1, w2, _ in state.windows:
        a, b = px_to_m(w1, 100), px_to_m(w2, 100)
        window_line = LineString([a, b])
        edge = min(edges, key=lambda e: e["line"].distance(window_line))
        
        t0 = edge["line"].project(Point(a)) / edge["length"]
        t1 = edge["line"].project(Point(b)) / edge["length"]
        edge["openings"].append(("window", min(t0, t1), max(t0, t1)))
    
    # Create wall selection dialog with modern styling
    wall_select = tk.Toplevel(root)
    wall_select.title("📸 Capture Room Perspective")
    wall_select.geometry("450x600")
    wall_select.configure(bg=GALLERY_COLORS["bg"])
    wall_select.transient(root)
    
    # Center dialog
    wall_select.geometry(f"+{root.winfo_x() + 200}+{root.winfo_y() + 50}")
    
    # Header
    header_frame = tk.Frame(wall_select, bg=GALLERY_COLORS["surface"], height=80)
    header_frame.pack(fill=tk.X)
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="📸 Capture Perspective", font=("Segoe UI", 18, "bold"),
             bg=GALLERY_COLORS["surface"], fg=GALLERY_COLORS["text"]).pack(pady=20)
    
    # Content
    content_frame = tk.Frame(wall_select, bg=GALLERY_COLORS["bg"])
    content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
    
    tk.Label(content_frame, text="Select which wall to remove\nfor the perspective view:",
             font=("Segoe UI", 11), bg=GALLERY_COLORS["bg"], 
             fg=GALLERY_COLORS["text"], justify=tk.CENTER).pack(pady=(0, 15))
    
    selected_wall = tk.IntVar(value=0)
    
    # Wall selection with modern radio buttons
    walls_frame = tk.Frame(content_frame, bg=GALLERY_COLORS["surface"])
    walls_frame.pack(fill=tk.X, pady=10)
    
    for edge_idx, edge in enumerate(edges):
        distance = np.linalg.norm(np.array(edge["p2"]) - np.array(edge["p1"]))
        
        wall_option = tk.Frame(walls_frame, bg=GALLERY_COLORS["surface"])
        wall_option.pack(fill=tk.X, padx=10, pady=5)
        
        rb = tk.Radiobutton(wall_option, text=f"  Wall {edge_idx + 1}",
                           variable=selected_wall, value=edge_idx,
                           font=("Segoe UI", 11), bg=GALLERY_COLORS["surface"],
                           fg=GALLERY_COLORS["text"], selectcolor=GALLERY_COLORS["primary"],
                           activebackground=GALLERY_COLORS["surface"],
                           activeforeground=GALLERY_COLORS["text"])
        rb.pack(side=tk.LEFT)
        
        tk.Label(wall_option, text=f"({distance:.1f}m)", font=("Segoe UI", 10),
                 bg=GALLERY_COLORS["surface"], fg=GALLERY_COLORS["text_secondary"]).pack(side=tk.RIGHT, padx=10)
    
    # Room type selection
    tk.Label(content_frame, text="Room Type:", font=("Segoe UI", 11, "bold"),
             bg=GALLERY_COLORS["bg"], fg=GALLERY_COLORS["text"]).pack(anchor="w", pady=(20, 5))
    
    room_types = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office", "Dining Room"]
    room_type_var = tk.StringVar(value="Living Room")
    
    room_type_menu = ttk.Combobox(content_frame, textvariable=room_type_var, 
                                   values=room_types, font=("Segoe UI", 10), state="readonly")
    room_type_menu.pack(fill=tk.X, pady=5)
    
    # Options
    auto_generate_var = tk.BooleanVar(value=False)
    tk.Checkbutton(content_frame, text="🎨 Auto-generate AI design after capture",
                   variable=auto_generate_var, font=("Segoe UI", 10),
                   bg=GALLERY_COLORS["bg"], fg=GALLERY_COLORS["text"],
                   selectcolor=GALLERY_COLORS["surface"],
                   activebackground=GALLERY_COLORS["bg"]).pack(anchor="w", pady=15)
    
    # Status label
    status_label = tk.Label(content_frame, text="Press SPACE in 3D view to capture screenshot",
                            font=("Segoe UI", 10), bg=GALLERY_COLORS["bg"], 
                            fg=GALLERY_COLORS["text_secondary"])
    status_label.pack(pady=5)
    
    def show_screenshot_notification(entry, auto_generate):
        """Show notification after screenshot is taken"""
        notif = tk.Toplevel(root)
        notif.title("✓ Screenshot Captured")
        notif.geometry("400x350")
        notif.configure(bg=GALLERY_COLORS["bg"])
        notif.transient(root)
        
        notif.geometry(f"+{root.winfo_x() + 300}+{root.winfo_y() + 150}")
        
        # Success icon and message
        tk.Label(notif, text="✓", font=("Segoe UI", 48),
                 bg=GALLERY_COLORS["bg"], fg=GALLERY_COLORS["success"]).pack(pady=20)
        
        tk.Label(notif, text="Screenshot Captured!", font=("Segoe UI", 16, "bold"),
                 bg=GALLERY_COLORS["bg"], fg=GALLERY_COLORS["text"]).pack()
        
        tk.Label(notif, text=f"Saved to gallery for {entry.room_name}",
                 font=("Segoe UI", 11), bg=GALLERY_COLORS["bg"],
                 fg=GALLERY_COLORS["text_secondary"]).pack(pady=10)
        
        # Preview
        try:
            img = Image.open(entry.screenshot_path)
            img.thumbnail((200, 120))
            photo = ImageTk.PhotoImage(img)
            preview = tk.Label(notif, image=photo, bg=GALLERY_COLORS["surface"])
            preview.image = photo
            preview.pack(pady=10)
        except:
            pass
        
        # Buttons
        btn_frame = tk.Frame(notif, bg=GALLERY_COLORS["bg"])
        btn_frame.pack(pady=15)
        
        def open_gallery_and_close():
            notif.destroy()
            open_design_gallery()
        
        tk.Button(btn_frame, text="📂 Open Gallery", font=("Segoe UI", 10, "bold"),
                  bg=GALLERY_COLORS["primary"], fg=GALLERY_COLORS["text"],
                  relief=tk.FLAT, padx=15, pady=8, cursor="hand2",
                  command=open_gallery_and_close).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Close", font=("Segoe UI", 10),
                  bg=GALLERY_COLORS["surface"], fg=GALLERY_COLORS["text"],
                  relief=tk.FLAT, padx=15, pady=8, cursor="hand2",
                  command=notif.destroy).pack(side=tk.LEFT, padx=5)
        
        # Auto-close after 5 seconds
        notif.after(5000, lambda: notif.destroy() if notif.winfo_exists() else None)
        
        # If auto-generate is enabled
        if auto_generate:
            notif.after(500, lambda: trigger_auto_generate(entry, notif))
    
    def trigger_auto_generate(entry, notif):
        """Trigger AI generation automatically with progress dialog"""
        print(f"[INFO] Auto-generating AI design for {entry.room_name}...")
        
        # Create progress indicator
        progress_win = tk.Toplevel(root)
        progress_win.title("Generating AI Design")
        progress_win.geometry("400x180")
        progress_win.configure(bg=GALLERY_COLORS["bg"])
        progress_win.transient(root)
        
        # Center
        progress_win.update_idletasks()
        px = (progress_win.winfo_screenwidth() - 400) // 2
        py = (progress_win.winfo_screenheight() - 180) // 2
        progress_win.geometry(f"400x180+{px}+{py}")
        
        tk.Label(progress_win, text="Generating AI Design...",
                 font=("Segoe UI", 14, "bold"), bg=GALLERY_COLORS["bg"],
                 fg=GALLERY_COLORS["text"]).pack(pady=25)
        
        tk.Label(progress_win, text="This may take a few minutes...",
                 font=("Segoe UI", 10), bg=GALLERY_COLORS["bg"],
                 fg=GALLERY_COLORS["text_secondary"]).pack(pady=5)
        
        progress_bar = ttk.Progressbar(progress_win, mode='indeterminate', length=300)
        progress_bar.pack(pady=20)
        progress_bar.start(10)
        
        def generate():
            try:
                print(f"[INFO] Calling AI generator...")
                result = generate_ai_interior_design(entry)
                print(f"[INFO] AI result: {result}")
                
                def on_complete():
                    progress_bar.stop()
                    if progress_win.winfo_exists():
                        progress_win.destroy()
                    
                    if result and os.path.exists(result):
                        # Update metadata using the imported function
                        update_design_entry(entry.design_id, {
                            "ai_path": result,
                            "status": "completed"
                        })
                        print(f"[OK] AI design completed and saved: {result}")
                        messagebox.showinfo("Success", "AI design generated successfully!")
                    else:
                        messagebox.showerror("Error", 
                            "Failed to generate AI design.\nCheck the console for details.")
                
                root.after(100, on_complete)
                
            except Exception as e:
                print(f"[ERROR] Auto-generation failed: {e}")
                import traceback
                traceback.print_exc()
                
                def on_error():
                    progress_bar.stop()
                    if progress_win.winfo_exists():
                        progress_win.destroy()
                    messagebox.showerror("Error", f"Generation failed:\n{str(e)}")
                
                root.after(100, on_error)
        
        threading.Thread(target=generate, daemon=True).start()
    
    def launch_perspective_and_capture():
        wall_idx = selected_wall.get()
        selected_edge = edges[wall_idx]
        room_type = room_type_var.get()
        
        status_label.config(text="Opening 3D view...", fg=GALLERY_COLORS["warning"])
        wall_select.update()
        
        # Import colors and constants from plan3d for consistency
        from plan3d import FLOOR_COLOR, CEILING_COLOR, WALL_HEIGHT, DOOR_HEIGHT
        
        # Create visualization
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Room Perspective - Press SPACE to capture, Q to close", 1400, 900)
        
        # Add floor with realistic wood color
        vis.add_geometry(floor_mesh(room_m, FLOOR_COLOR))
        
        # Add ceiling
        ceiling = floor_mesh(room_m, CEILING_COLOR)
        ceiling.translate((0, 0, WALL_HEIGHT))
        vis.add_geometry(ceiling)
        
        # Add walls - skip the selected wall
        for edge_idx, edge in enumerate(edges):
            if edge_idx != wall_idx:
                for w in build_walls_from_edges([edge]):
                    if w:
                        vis.add_geometry(w)
        
        # Camera setup
        room_center = polygon_centroid_3d(room_m)
        
        wall_p1 = np.array(selected_edge["p1"])
        wall_p2 = np.array(selected_edge["p2"])
        wall_center = (wall_p1 + wall_p2) / 2
        
        into_room = room_center[:2] - wall_center
        into_room /= np.linalg.norm(into_room)
        
        cam_dist = 1.0
        cam_pos = np.array([
            wall_center[0] - into_room[0] * cam_dist,
            wall_center[1] - into_room[1] * cam_dist,
            EYE_HEIGHT
        ])
        
        look_target = np.array([room_center[0], room_center[1], EYE_HEIGHT])
        
        ctr = vis.get_view_control()
        ctr.set_lookat(look_target)
        ctr.set_front(cam_pos - look_target)
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.48)
        
        # Set render options for better lighting/contrast
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0.15, 0.15, 0.18])  # Dark background outside
        render_opt.light_on = False
        render_opt.point_show_normal = False
        
        vis.poll_events()
        vis.update_renderer()
        
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        
        running = [True]
        screenshot_taken = [False]
        saved_entry = [None]
        
        # Analyze visible walls for actual door/window openings from floor plan data
        # Also store 3D geometry so planAI can project them to 2D pixel positions
        visible_has_doors = False
        visible_has_windows = False
        visible_openings = []
        for edge_idx, edge in enumerate(edges):
            if edge_idx == wall_idx:
                continue  # Skip removed wall
            ep1 = np.array(edge["p1"])
            ep2 = np.array(edge["p2"])
            edge_vec = ep2 - ep1
            edge_len = np.linalg.norm(edge_vec)
            for opening in edge.get("openings", []):
                opening_type = opening[0]  # "door" or "window"
                t0, t1 = opening[1], opening[2]
                # 3D world positions of the opening (bottom left & bottom right)
                # Door: floor to DOOR_HEIGHT; Window: WINDOW_BOTTOM to WINDOW_TOP
                WINDOW_BOTTOM = 0.8
                WINDOW_TOP    = 2.2
                if opening_type == "door":
                    z_bottom = 0.0
                    z_top    = DOOR_HEIGHT
                else:
                    z_bottom = WINDOW_BOTTOM
                    z_top    = WINDOW_TOP
                # XY positions along the wall
                xy0 = ep1 + edge_vec * t0
                xy1 = ep1 + edge_vec * t1
                visible_openings.append({
                    "type": opening_type,
                    "wall_index": edge_idx,
                    # 3D corners: [bottom-left, bottom-right, top-right, top-left]
                    "corners_3d": [
                        [float(xy0[0]), float(xy0[1]), z_bottom],
                        [float(xy1[0]), float(xy1[1]), z_bottom],
                        [float(xy1[0]), float(xy1[1]), z_top],
                        [float(xy0[0]), float(xy0[1]), z_top],
                    ]
                })
                if opening_type == "door":
                    visible_has_doors = True
                elif opening_type == "window":
                    visible_has_windows = True

        print(f"[INFO] Visible layout: doors={visible_has_doors}, windows={visible_has_windows}, openings={len(visible_openings)}")
        
        def take_screenshot(vis):
            """Capture screenshot"""
            # Capture the current view
            vis.poll_events()
            vis.update_renderer()
            
            # Capture to image
            image = vis.capture_screen_float_buffer(do_render=True)
            image_np = (np.asarray(image) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Capture CURRENT camera parameters at the moment of screenshot
            cur_cam = ctr.convert_to_pinhole_camera_parameters()
            intr = cur_cam.intrinsic
            cam_dict = {
                "width":  intr.width,
                "height": intr.height,
                # 3×3 intrinsic matrix stored as list of rows
                "intrinsic": intr.intrinsic_matrix.tolist(),
                # 4×4 extrinsic matrix (world-to-camera) stored as list of rows
                "extrinsic": cur_cam.extrinsic.tolist(),
            }
            
            # Save screenshot with actual layout info from the floor plan
            entry = save_screenshot_entry(
                room_name=room_name,
                wall_index=wall_idx,
                screenshot_image=pil_image,
                room_type=room_type,
                has_doors=visible_has_doors,
                has_windows=visible_has_windows,
                visible_openings=visible_openings,
                camera_params=cam_dict
            )
            
            saved_entry[0] = entry
            screenshot_taken[0] = True
            
            print(f"[OK] Screenshot saved: {entry.screenshot_path}")
            
            # Show success notification
            root.after(0, lambda: show_screenshot_notification(entry, auto_generate_var.get()))
            
            return False  # Don't close yet
        
        def on_close(vis):
            running[0] = False
            return False
        
        # Register key callbacks
        vis.register_key_callback(32, take_screenshot)  # SPACE key
        vis.register_key_callback(ord("Q"), on_close)
        vis.register_key_callback(ord("q"), on_close)
        vis.register_key_callback(27, on_close)  # ESC
        
        print("\n" + "="*50)
        print("PERSPECTIVE VIEW CONTROLS:")
        print("  SPACE - Take screenshot")
        print("  Q/ESC - Close window")
        print("="*50 + "\n")
        
        import time
        while running[0]:
            if not vis.poll_events():
                break
            ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
            vis.update_renderer()
            time.sleep(0.01)
        
        vis.destroy_window()
        wall_select.destroy()
    
    # Buttons
    btn_frame = tk.Frame(content_frame, bg=GALLERY_COLORS["bg"])
    btn_frame.pack(pady=20)
    
    tk.Button(btn_frame, text="📸 Launch View", font=("Segoe UI", 11, "bold"),
              bg=GALLERY_COLORS["accent"], fg=GALLERY_COLORS["text"],
              relief=tk.FLAT, padx=25, pady=12, cursor="hand2",
              command=launch_perspective_and_capture).pack(side=tk.LEFT, padx=10)
    
    tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 11),
              bg=GALLERY_COLORS["surface"], fg=GALLERY_COLORS["text"],
              relief=tk.FLAT, padx=25, pady=12, cursor="hand2",
              command=wall_select.destroy).pack(side=tk.LEFT, padx=10)


# ---------- ROOM VIEWER WITH 3D OPTIONS ----------
def open_room_viewer():
    if not state.room_segments:
        messagebox.showinfo("No rooms", "No rooms defined yet")
        return

    viewer = tk.Toplevel(root)
    viewer.title("Room Viewer")
    viewer.geometry(f"{CANVAS_MAX_W+300}x{CANVAS_MAX_H}")
    viewer.configure(bg=BG_COLOR)

    # Sidebar with modern styling
    sidebar = tk.Frame(viewer, bg=BG_COLOR, width=300)
    sidebar.pack(side=tk.LEFT, fill=tk.Y)
    sidebar.pack_propagate(False)

    # Sidebar title
    tk.Label(sidebar, text="🏠 Rooms", font=("Segoe UI", 16, "bold"), 
             bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=15, padx=15, anchor="w")
    
    # Gallery button at top
    tk.Button(sidebar, text="🎨 Open Design Gallery", font=("Segoe UI", 10, "bold"),
              bg=ACCENT_COLOR, fg="white", relief=tk.FLAT, padx=15, pady=8,
              cursor="hand2", command=open_design_gallery).pack(pady=(0, 15), padx=15, fill=tk.X)

    view_canvas = tk.Canvas(viewer, width=CANVAS_MAX_W, height=CANVAS_MAX_H, bg="#eeeeee")
    view_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def show_room(name):
        view_canvas.delete("all")
        img = extract_room_image(state.room_segments[name])

        w, h = img.size
        scale = min(CANVAS_MAX_W / w, CANVAS_MAX_H / h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        imtk = ImageTk.PhotoImage(img)
        view_canvas.image = imtk
        view_canvas.create_image(0, 0, anchor="nw", image=imtk)

    def view_room_3d(name):
        build_3d_apartment_and_walk(
            outer_px=[],
            room_polygons_px=[state.room_segments[name]],
            doors_px=state.doors,
            windows_px=state.windows,
            px_per_m=100,
            room_only=True
        )

    def walk_room_3d(name):
        """Furnished real-scale walkthrough of a single room."""
        import plan_walkthrough
        room_type, style = "Living Room", "Modern"
        try:
            for d in load_metadata().get("designs", []):
                if d.get("room_name") == name:
                    room_type = d.get("room_type", room_type)
                    style = d.get("style", style)
        except Exception:
            pass
        plan_walkthrough.launch_walkthrough(
            [state.room_segments[name]], state.doors, state.windows,
            px_per_m=None,
            room_configs=[dict(room_type=room_type, style=style, name=name)])

    # Room list
    for name in state.room_segments.keys():
        room_frame = tk.Frame(sidebar, bg="white", relief=tk.FLAT)
        room_frame.pack(pady=8, padx=10, fill=tk.X)

        # Room name header
        room_header = tk.Frame(room_frame, bg="white")
        room_header.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        tk.Label(room_header, text=name, font=("Segoe UI", 12, "bold"), 
                 bg="white", fg=PRIMARY_COLOR).pack(side=tk.LEFT)

        # Buttons frame - row 1
        btn_frame1 = tk.Frame(room_frame, bg="white")
        btn_frame1.pack(fill=tk.X, padx=10, pady=3)

        tk.Button(btn_frame1, text="📷 View 2D", font=("Segoe UI", 9), width=12,
                  bg=SECONDARY_COLOR, fg=TEXT_COLOR, relief=tk.FLAT, cursor="hand2",
                  command=lambda n=name: show_room(n)).pack(side=tk.LEFT, padx=2)

        tk.Button(btn_frame1, text="🎬 View 3D", font=("Segoe UI", 9), width=12,
                  bg=PRIMARY_COLOR, fg="white", relief=tk.FLAT, cursor="hand2",
                  command=lambda n=name: view_room_3d(n)).pack(side=tk.LEFT, padx=2)

        # Buttons frame - row 2
        btn_frame2 = tk.Frame(room_frame, bg="white")
        btn_frame2.pack(fill=tk.X, padx=10, pady=3)

        tk.Button(btn_frame2, text="🚶 Walkthrough", font=("Segoe UI", 9), width=26,
                  bg="#6f42c1", fg="white", relief=tk.FLAT, cursor="hand2",
                  command=lambda n=name: walk_room_3d(n)).pack(fill=tk.X, padx=2)

        # Buttons frame - row 3
        btn_frame3 = tk.Frame(room_frame, bg="white")
        btn_frame3.pack(fill=tk.X, padx=10, pady=(3, 10))

        tk.Button(btn_frame3, text="� Capture Perspective", font=("Segoe UI", 9), width=26,
                  bg=SUCCESS_COLOR, fg="white", relief=tk.FLAT, cursor="hand2",
                  command=lambda n=name: take_room_perspective_screenshot(state.room_segments[n], n)).pack(fill=tk.X, padx=2)

# ---------- AUTO-DETECT ROOMS & DOORS FROM THE PLAN IMAGE ----------
def auto_detect_plan():
    """Extract rooms and doors from the plan image — no drawing needed."""
    if state.inners and not messagebox.askyesno(
            "Auto-Detect",
            "Replace the current rooms & doors with auto-detected ones?"):
        return
    try:
        import plan_autodetect
        rooms, doors = plan_autodetect.detect_rooms_and_doors(state.orig_img_cv)
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Auto-Detect", f"Detection failed:\n{e}")
        return
    if not rooms:
        messagebox.showerror(
            "Auto-Detect", "No rooms could be detected in this plan image.\n"
                           "You can still draw them manually.")
        return
    state.inners = [[tuple(p) for p in poly] for poly in rooms]
    state.room_segments = {f"Room {i + 1}": poly
                           for i, poly in enumerate(state.inners)}
    state.doors = [(tuple(a), tuple(b)) for a, b in doors]
    state.current_inner = []
    state.undo_stack.clear()
    save_layout()
    redraw()
    lbl_info.config(text=f"✓ Auto-detected {len(rooms)} rooms, "
                         f"{len(doors)} doors", fg="#28a745")
    messagebox.showinfo(
        "Auto-Detect",
        f"Detected {len(rooms)} rooms and {len(doors)} doors.\n\n"
        "Add windows (or adjust) if you like, then click 3D Walkthrough.")


# ---------- 3D WALKTHROUGH (2D plan -> furnished 3D model you can walk) ----------
def open_walkthrough_dialog():
    """Configure and launch the preference-driven, walkable 3D interior."""
    if not state.room_segments:
        messagebox.showinfo("No rooms", "Draw at least one room first")
        return

    import plan_walkthrough

    room_types = ["Living Room", "Bedroom", "Kitchen", "Bathroom",
                  "Office", "Dining Room"]
    styles = ["Modern Minimalist", "Scandinavian", "Industrial", "Bohemian",
              "Mid-Century Modern", "Contemporary", "Traditional", "Japandi",
              "Modern", "Minimalist", "Classic", "Boho"]

    # Default each room to the preferences of its latest gallery design
    saved_prefs = {}
    try:
        for d in load_metadata().get("designs", []):
            saved_prefs[d.get("room_name")] = (
                d.get("room_type", "Living Room"), d.get("style", "Modern"))
    except Exception:
        pass

    dlg = tk.Toplevel(root)
    dlg.title("3D Interior Walkthrough")
    dlg.configure(bg=BG_COLOR)
    dlg.transient(root)
    dlg.grab_set()
    dlg.resizable(False, True)
    dlg.minsize(660, 560)
    dlg.maxsize(760, max(620, root.winfo_screenheight() - 80))
    dlg.geometry(f"+{root.winfo_x() + 260}+{root.winfo_y() + 90}")

    # Header band
    head = tk.Frame(dlg, bg=SURFACE)
    head.pack(fill=tk.X)
    tk.Label(head, text="3D Interior Walkthrough", font=("Segoe UI", 17, "bold"),
             bg=SURFACE, fg=TEXT_COLOR).pack(anchor="w", padx=24, pady=(18, 3))
    tk.Label(head, text="Turn the floor plan into a coordinated interior you can explore.",
             font=("Segoe UI", 10), bg=SURFACE, fg=TEXT_MUTED,
             justify=tk.LEFT).pack(anchor="w", padx=24, pady=(0, 12))
    steps = tk.Frame(head, bg=SURFACE)
    steps.pack(fill=tk.X, padx=24, pady=(0, 16))
    for label, color in (("1  Assign rooms", PRIMARY_COLOR),
                         ("2  Apply finishes", PURPLE_COLOR),
                         ("3  Walk through", SUCCESS_COLOR)):
        tk.Label(steps, text=label, font=("Segoe UI", 9, "bold"),
                 bg="#f1f5f9", fg=color, padx=10, pady=5).pack(
                     side=tk.LEFT, padx=(0, 8))
    tk.Frame(dlg, bg=BORDER, height=1).pack(fill=tk.X)

    body = tk.Frame(dlg, bg=BG_COLOR)
    body.pack(fill=tk.BOTH, expand=True, padx=24, pady=18)

    global_card = tk.Frame(body, bg=SURFACE, highlightbackground=BORDER,
                           highlightthickness=1)
    global_card.pack(fill=tk.X, pady=(0, 14))
    tk.Label(global_card, text="DESIGN DIRECTION", font=("Segoe UI", 8, "bold"),
             bg=SURFACE, fg=TEXT_MUTED).grid(row=0, column=0, columnspan=3,
                                             sticky="w", padx=12, pady=(10, 5))
    profile_var = tk.StringVar(value="Curated")
    ttk.Combobox(global_card, textvariable=profile_var,
                 values=["Airy", "Curated", "Layered"], width=14,
                 state="readonly", font=("Segoe UI", 10)).grid(
                     row=1, column=0, padx=(12, 8), pady=(0, 11), sticky="w")
    tk.Label(global_card, text="Airy: essentials  •  Curated: balanced  •  Layered: richer decor",
             font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_MUTED).grid(
                 row=1, column=1, padx=(0, 12), pady=(0, 11), sticky="w")
    palette_var = tk.StringVar(value="Warm neutral")
    tk.Label(global_card, text="COLOR MOOD", font=("Segoe UI", 8, "bold"),
             bg=SURFACE, fg=TEXT_MUTED).grid(
                 row=2, column=0, sticky="w", padx=12, pady=(0, 4))
    tk.Label(global_card, text="PERSONAL BRIEF (OPTIONAL)", font=("Segoe UI", 8, "bold"),
             bg=SURFACE, fg=TEXT_MUTED).grid(
                 row=2, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
    ttk.Combobox(
        global_card, textvariable=palette_var,
        values=["Warm neutral", "Cool neutral", "Earthy natural",
                "Light and airy", "Monochrome", "Bold accents"],
        width=18, state="readonly", font=("Segoe UI", 9),
    ).grid(row=3, column=0, padx=(12, 8), pady=(0, 11), sticky="w")
    notes_var = tk.StringVar()
    notes_entry = tk.Entry(
        global_card, textvariable=notes_var, font=("Segoe UI", 9),
        relief=tk.FLAT, highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=PRIMARY_COLOR,
    )
    notes_entry.grid(
        row=3, column=1, padx=(0, 12), pady=(0, 11), sticky="ew"
    )
    floor_var = tk.StringVar(value="Auto by style")
    wall_var = tk.StringVar(value="Auto by style")
    tk.Label(global_card, text="FLOOR FINISH", font=("Segoe UI", 8, "bold"),
             bg=SURFACE, fg=TEXT_MUTED).grid(
                 row=4, column=0, sticky="w", padx=12, pady=(0, 4))
    tk.Label(global_card, text="WALL FINISH", font=("Segoe UI", 8, "bold"),
             bg=SURFACE, fg=TEXT_MUTED).grid(
                 row=4, column=1, sticky="w", padx=(0, 12), pady=(0, 4))
    ttk.Combobox(
        global_card, textvariable=floor_var,
        values=["Auto by style", "Light oak", "Warm oak", "Dark walnut",
                "Natural stone", "Polished concrete", "Terrazzo", "Large tile"],
        width=18, state="readonly", font=("Segoe UI", 9),
    ).grid(row=5, column=0, padx=(12, 8), pady=(0, 11), sticky="w")
    ttk.Combobox(
        global_card, textvariable=wall_var,
        values=["Auto by style", "Warm paint", "Cool paint", "Limewash",
                "Wood slats", "Panel moulding", "Concrete", "Accent color"],
        width=28, state="readonly", font=("Segoe UI", 9),
    ).grid(row=5, column=1, padx=(0, 12), pady=(0, 11), sticky="ew")
    global_card.grid_columnconfigure(1, weight=1)

    # Column headers
    hdr = tk.Frame(body, bg=BG_COLOR)
    hdr.pack(fill=tk.X, pady=(0, 6))
    tk.Label(hdr, text="ROOM", width=12, anchor="w", font=("Segoe UI", 8, "bold"),
             bg=BG_COLOR, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=(4, 8))
    tk.Label(hdr, text="TYPE", width=15, anchor="w", font=("Segoe UI", 8, "bold"),
             bg=BG_COLOR, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=4)
    tk.Label(hdr, text="STYLE", width=20, anchor="w", font=("Segoe UI", 8, "bold"),
             bg=BG_COLOR, fg=TEXT_MUTED).pack(side=tk.LEFT, padx=4)

    room_vars = []  # (name, type_var, style_var)
    for idx, name in enumerate(state.room_segments.keys()):
        def_type, def_style = saved_prefs.get(name, ("Living Room", "Modern"))
        row = tk.Frame(body, bg=SURFACE, highlightbackground=BORDER,
                       highlightthickness=1)
        row.pack(fill=tk.X, pady=3)
        swatch = ROOM_COLORS[idx % len(ROOM_COLORS)]
        chip = tk.Frame(row, bg=SURFACE)
        chip.pack(side=tk.LEFT, padx=(10, 0), pady=9)
        tk.Label(chip, text="●", font=("Segoe UI", 11), bg=SURFACE,
                 fg=swatch).pack(side=tk.LEFT)
        tk.Label(chip, text=name, width=9, anchor="w",
                 font=("Segoe UI", 10, "bold"), bg=SURFACE,
                 fg=TEXT_COLOR).pack(side=tk.LEFT, padx=(4, 0))
        tvar = tk.StringVar(value=def_type if def_type in room_types else "Living Room")
        ttk.Combobox(row, textvariable=tvar, values=room_types, width=15,
                     font=("Segoe UI", 10), state="readonly").pack(side=tk.LEFT, padx=6)
        svar = tk.StringVar(value=def_style if def_style in styles else "Modern")
        ttk.Combobox(row, textvariable=svar, values=styles, width=20,
                     font=("Segoe UI", 10), state="readonly").pack(side=tk.LEFT, padx=6)
        room_vars.append((name, tvar, svar))

    apply_row = tk.Frame(body, bg=BG_COLOR)
    apply_row.pack(fill=tk.X, pady=(10, 4))
    apply_style_var = tk.StringVar(value="Modern")
    ttk.Combobox(apply_row, textvariable=apply_style_var, values=styles, width=22,
                 state="readonly", font=("Segoe UI", 9)).pack(side=tk.LEFT)

    def apply_style_to_all():
        for _name, _tvar, svar in room_vars:
            svar.set(apply_style_var.get())

    make_btn(apply_row, "Apply style to all rooms", SECONDARY_COLOR,
             SECONDARY_HOVER, fg=TEXT_COLOR, command=apply_style_to_all,
             font_size=9).pack(side=tk.LEFT, padx=8)

    explore_card = tk.Frame(body, bg=SURFACE, highlightbackground=BORDER,
                            highlightthickness=1)
    explore_card.pack(fill=tk.X, pady=(8, 0))
    wall_pass_var = tk.BooleanVar(value=True)
    tk.Checkbutton(explore_card, text="Free explore — allow walking through walls",
                   variable=wall_pass_var, font=("Segoe UI", 10, "bold"),
                   bg=SURFACE, fg=TEXT_COLOR, selectcolor=SURFACE,
                   activebackground=SURFACE, cursor="hand2").pack(
                       anchor="w", padx=10, pady=(9, 1))
    tk.Label(explore_card,
             text="Press G while walking to switch between pass-through walls and doorways.",
             font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_MUTED).pack(
                 anchor="w", padx=14, pady=(0, 9))

    engine_card = tk.Frame(body, bg=SURFACE, highlightbackground=BORDER,
                           highlightthickness=1)
    engine_card.pack(fill=tk.X, pady=(8, 0))
    engine_var = tk.StringVar(value="Procedural - instant & stable")
    try:
        import local_3d_ai
        tripo_ready, tripo_reason = local_3d_ai.runtime_status()
    except Exception as exc:
        tripo_ready, tripo_reason = False, f"TripoSR is unavailable: {exc}"
    engine_values = ["Procedural - instant & stable"]
    if tripo_ready:
        engine_values.append("TripoSR - local AI (experimental)")
    tk.Label(engine_card, text="FURNITURE GEOMETRY", font=("Segoe UI", 8, "bold"),
             bg=SURFACE, fg=TEXT_MUTED).grid(
                 row=0, column=0, sticky="w", padx=10, pady=(9, 4))
    ttk.Combobox(
        engine_card, textvariable=engine_var, values=engine_values,
        width=34, state="readonly", font=("Segoe UI", 9),
    ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 4))
    tk.Label(
        engine_card,
        text=(tripo_reason + " TripoSR changes furniture meshes only; the complete "
              "room layout, finishes, lighting and decor stay coordinated."),
        font=("Segoe UI", 9), bg=SURFACE,
        fg=SUCCESS_COLOR if tripo_ready else TEXT_MUTED,
        wraplength=600, justify=tk.LEFT,
    ).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 9))
    engine_card.grid_columnconfigure(0, weight=1)

    def launch(only_room=None):
        configs, rooms = [], []
        variation = "preference-render-v3"
        use_triposr = engine_var.get().startswith("TripoSR")
        for name, tvar, svar in room_vars:
            if only_room and name != only_room:
                continue
            rooms.append(state.room_segments[name])
            configs.append(dict(room_type=tvar.get(), style=svar.get(), name=name,
                                design_profile=profile_var.get(),
                                color_mood=palette_var.get(),
                                design_notes=notes_var.get(),
                                floor_finish=floor_var.get(),
                                wall_finish=wall_var.get(),
                                design_seed=variation,
                                whole_room_design=True,
                                use_triposr=use_triposr))
        dlg.withdraw()
        loading = tk.Toplevel(root)
        loading.title("Building 3D interior")
        loading.configure(bg=SURFACE)
        loading.transient(root)
        loading.resizable(False, False)
        loading.geometry(f"480x175+{root.winfo_x() + 480}+{root.winfo_y() + 260}")
        tk.Label(loading, text="Building your designed 3D interior…",
                 font=("Segoe UI", 14, "bold"), bg=SURFACE,
                 fg=TEXT_COLOR).pack(pady=(28, 6))
        loading_status = tk.StringVar(
            value="Applying finishes, lighting, furniture, decor and clear circulation")
        tk.Label(loading, textvariable=loading_status,
                 font=("Segoe UI", 9), bg=SURFACE, fg=TEXT_MUTED,
                 wraplength=430, justify=tk.CENTER).pack()
        loading.update()

        def update_loading(message):
            if loading.winfo_exists():
                loading_status.set(message)
                loading.update()

        try:
            if use_triposr:
                update_loading("Preparing local TripoSR furniture meshes...")
                try:
                    import local_3d_ai
                    local_3d_ai.prepare_local_assets(configs, update_loading)
                except Exception as exc:
                    if loading.winfo_exists():
                        loading.destroy()
                    dlg.deiconify()
                    dlg.grab_set()
                    messagebox.showerror(
                        "TripoSR furniture could not be built",
                        f"{exc}\n\nNo fallback was opened, so you can retry or "
                        "select the procedural furniture engine.",
                        parent=dlg,
                    )
                    return
            dlg.destroy()
            loading_status.set("Assembling the measured apartment and room designs…")
            loading.update()
            plan_walkthrough.launch_walkthrough(
                rooms, state.doors, state.windows, px_per_m=None,
                room_configs=configs, furnished=True,
                wall_pass=wall_pass_var.get())
        finally:
            if loading.winfo_exists():
                loading.destroy()

    # Footer
    tk.Frame(dlg, bg=BORDER, height=1).pack(fill=tk.X)
    btns = tk.Frame(dlg, bg=SURFACE)
    btns.pack(fill=tk.X)
    inner = tk.Frame(btns, bg=SURFACE)
    inner.pack(anchor="e", padx=20, pady=14)
    make_btn(inner, "Cancel", SECONDARY_COLOR, SECONDARY_HOVER, fg=TEXT_COLOR,
             command=dlg.destroy).pack(side=tk.LEFT, padx=6)
    make_btn(inner, "Build & start walkthrough", PURPLE_COLOR, PURPLE_HOVER,
             command=launch, font_size=11).pack(side=tk.LEFT, padx=6)


# ---------- TOOLBAR BUTTONS (created after their commands exist) ----------
# Edit tools (workflow bar)
make_btn(edit_frame, "↶  Undo", SECONDARY_COLOR, SECONDARY_HOVER, fg=TEXT_COLOR,
         command=undo).pack(side=tk.LEFT, padx=3)
make_btn(edit_frame, "🗑  Reset", "#fee2e2", "#fecaca", fg=DANGER_COLOR,
         command=reset_all).pack(side=tk.LEFT, padx=3)

# Primary actions (header, right side; packed right-to-left)
make_btn(action_frame, "✦  3D Walkthrough", PURPLE_COLOR, PURPLE_HOVER,
         command=open_walkthrough_dialog, font_size=11).pack(side=tk.RIGHT,
                                                             padx=(8, 0))
make_btn(action_frame, "🎨  Gallery", ACCENT_COLOR, ACCENT_HOVER,
         command=open_design_gallery).pack(side=tk.RIGHT, padx=4)
make_btn(action_frame, "📂  Rooms", PRIMARY_COLOR, PRIMARY_HOVER,
         command=open_room_viewer).pack(side=tk.RIGHT, padx=4)
make_btn(action_frame, "🪄  Auto-Detect", SECONDARY_COLOR, SECONDARY_HOVER,
         fg=TEXT_COLOR, command=auto_detect_plan).pack(side=tk.RIGHT, padx=4)

# ---------- KEYBOARD SHORTCUTS ----------
root.bind("<Control-z>", lambda e: undo())
root.bind("<Escape>", cancel_current)
root.bind("<Return>", finish_room)
root.bind("1", lambda e: set_mode("inner"))
root.bind("2", lambda e: set_mode("door"))
root.bind("3", lambda e: set_mode("window"))

# ---------- START ----------
set_mode("inner")
load_image(IMAGE_PATH)
if load_layout():
    lbl_info.config(text=f"✓ Restored saved plan: {len(state.inners)} rooms, "
                         f"{len(state.doors)} doors, {len(state.windows)} windows "
                         f"— no need to redraw",
                    fg=SUCCESS_COLOR)
redraw()
root.mainloop()
