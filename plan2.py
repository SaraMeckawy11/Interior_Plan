"""
Tkinter plan editor (extended, non-destructive additions)

Added:
- Per-room storage (Room 1, Room 2, ...)
- Room viewer window
- Per-room 3D view (same pipeline as Segment & 3D)
- 3D options dialog with configurable rendering
"""

import os, math, json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from io import BytesIO
import threading

import planAI as ai_furnisher
from plan3d import build_3d_apartment_and_walk
from room_gallery import (
    DesignGallery, save_screenshot_entry, load_metadata, save_metadata,
    DesignEntry, SCREENSHOTS_DIR, AI_DESIGNS_DIR, COLORS as GALLERY_COLORS,
    update_design_entry
)

# ---------------- CONFIG ----------------
IMAGE_PATH = r"C:\Users\Lenovo\Desktop\Interior_plan\plan\1.jpg"
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
root.title("Floor Plan Editor")
root.geometry("1400x900")

# Modern color scheme
BG_COLOR = "#f5f5f5"
PRIMARY_COLOR = "#0066cc"
SECONDARY_COLOR = "#f0f0f0"
TEXT_COLOR = "#333333"
ACCENT_COLOR = "#ff6b35"
DANGER_COLOR = "#ff6666"
SUCCESS_COLOR = "#28a745"

root.configure(bg=BG_COLOR)

# Top toolbar with horizontal layout
top_toolbar = tk.Frame(root, bg="white", height=70)
top_toolbar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
top_toolbar.pack_propagate(False)

# Title
title_label = tk.Label(top_toolbar, text="Floor Plan Editor", font=("Segoe UI", 18, "bold"), 
                       bg="white", fg=TEXT_COLOR)
title_label.pack(side=tk.LEFT, padx=20, pady=15)

# Mode buttons in horizontal layout
mode_frame = tk.Frame(top_toolbar, bg="white")
mode_frame.pack(side=tk.LEFT, padx=20, pady=10)

tk.Label(mode_frame, text="Draw Mode:", font=("Segoe UI", 11, "bold"), 
         bg="white", fg=TEXT_COLOR).pack(side=tk.LEFT, padx=(0, 15))

mode_buttons = {}

def create_mode_btn(text, mode_val):
    btn = tk.Button(mode_frame, text=text, font=("Segoe UI", 10), 
                    width=14, bg=SECONDARY_COLOR, fg=TEXT_COLOR,
                    relief=tk.FLAT, padx=10, pady=6, activebackground=PRIMARY_COLOR,
                    command=lambda: set_mode(mode_val))
    btn.pack(side=tk.LEFT, padx=5)
    mode_buttons[mode_val] = btn
    return btn

create_mode_btn("🔲 Rooms", "inner")
create_mode_btn("🚪 Doors", "door")
create_mode_btn("🪟 Windows", "window")

# Action buttons on the right
action_frame = tk.Frame(top_toolbar, bg="white")
action_frame.pack(side=tk.RIGHT, padx=20, pady=10)

# Status bar at bottom
status_frame = tk.Frame(root, bg="white", height=40, relief=tk.RAISED, bd=1)
status_frame.pack(side=tk.BOTTOM, fill=tk.X)
status_frame.pack_propagate(False)

lbl_mode = tk.Label(status_frame, text="Mode: Rooms", font=("Segoe UI", 10),
                    bg="white", fg=TEXT_COLOR)
lbl_mode.pack(side=tk.LEFT, padx=20, pady=10)

lbl_info = tk.Label(status_frame, text="Click to add points", font=("Segoe UI", 9), fg="#666666")
lbl_info.pack(side=tk.LEFT, padx=20, pady=10)

# Canvas in the middle
canvas = tk.Canvas(root, width=CANVAS_MAX_W, height=700, bg="#eeeeee", highlightthickness=0)
canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)

def set_mode(m):
    state.mode = m
    if m == "window":
        state.window_type = "normal"
    
    mode_text = {
        "inner": "🔲 Rooms",
        "door": "🚪 Doors",
        "window": "🪟 Windows"
    }
    lbl_mode.config(text=f"Mode: {mode_text.get(m, m)}")
    
    # Visual feedback on buttons with checkmark
    for mode_val, btn in mode_buttons.items():
        if mode_val == m:
            btn.config(bg=PRIMARY_COLOR, fg="white", text=mode_text[mode_val] + " ✓")
        else:
            text_clean = mode_text[mode_val].replace(" ✓", "")
            btn.config(bg=SECONDARY_COLOR, fg=TEXT_COLOR, text=text_clean)
    
    redraw()

# ---------- UNDO ----------
def undo():
    if not state.undo_stack:
        messagebox.showinfo("Info", "Nothing to undo")
        return

    action, data = state.undo_stack.pop()

    if action == "outer_point":
        state.outer.pop()
    elif action == "outer_close":
        state.outer_closed = False
    elif action == "inner_point":
        state.current_inner.pop()
    elif action == "inner_finish":
        last = state.inners.pop()
        state.current_inner = last
        state.room_segments.pop(f"Room {len(state.inners)+1}", None)
    elif action == "door":
        state.doors.pop()
    elif action == "window":
        state.windows.pop()

    redraw()

# ---------- DRAW ----------
def redraw():
    canvas.delete("all")
    if not state.image:
        load_image(IMAGE_PATH)

    imtk = ImageTk.PhotoImage(state.image)
    canvas.image = imtk
    canvas.create_image(0, 0, anchor="nw", image=imtk)

    def draw_poly(poly, col):
        pts = [img2canvas(p) for p in poly]
        if len(pts) > 1:
            canvas.create_line(*sum(pts, ()), fill=col, width=2)
        for x, y in pts:
            canvas.create_oval(x-4, y-4, x+4, y+4, fill="white")

    for p in state.inners:
        draw_poly(p, "#00aa33")
    draw_poly(state.current_inner, "#00aa33")

    for p1, p2 in state.doors:
        canvas.create_line(*img2canvas(p1), *img2canvas(p2), fill="#ff8800", width=6)

    for p1, p2, t in state.windows:
        color = "#00cccc" if t == "normal" else "#0099ff"
        canvas.create_line(*img2canvas(p1), *img2canvas(p2), fill=color, width=8)

# ---------- INPUT ----------
mouse_start = None
def click(e):
    global mouse_start
    pt = canvas2img((e.x, e.y))

    if state.mode == "inner":
        if state.current_inner and dist(img2canvas(state.current_inner[0]), (e.x, e.y)) < POLYGON_CLOSE_DIST:
            poly = state.current_inner[:]
            state.inners.append(poly)

            room_id = len(state.inners)
            state.room_segments[f"Room {room_id}"] = poly

            state.undo_stack.append(("inner_finish", None))
            state.current_inner = []
            lbl_info.config(text=f"✓ Room {room_id} created")
        else:
            state.current_inner.append(pt)
            state.undo_stack.append(("inner_point", None))
            lbl_info.config(text=f"Room points: {len(state.current_inner)}")
        redraw()

    elif state.mode in ("door", "window"):
        if mouse_start is None:
            mouse_start = pt
            mode_icon = "🚪" if state.mode == "door" else "🪟"
            lbl_info.config(text=f"{mode_icon} Click end point to finish", fg="#ff8800")
        else:
            if state.mode == "door":
                state.doors.append((mouse_start, pt))
                state.undo_stack.append(("door", None))
                lbl_info.config(text=f"✓ Door added! ({len(state.doors)} total)", fg="#28a745")
            else:
                state.windows.append((mouse_start, pt, state.window_type))
                state.undo_stack.append(("window", None))
                lbl_info.config(text=f"✓ Window added! ({len(state.windows)} total)", fg="#28a745")
            mouse_start = None
            redraw()

canvas.bind("<Button-1>", click)

# Add motion tracking for better UX
def on_motion(e):
    if state.mode in ("door", "window") and mouse_start is not None:
        pt = canvas2img((e.x, e.y))
        canvas.delete("preview_line")
        
        p1_canvas = img2canvas(mouse_start)
        p2_canvas = (e.x, e.y)
        
        color = "#ff8800" if state.mode == "door" else "#00cccc"
        canvas.create_line(*p1_canvas, *p2_canvas, fill=color, width=8, dash=(4, 4), tags="preview_line")
        
        # Show length
        dist_px = math.hypot(pt[0] - mouse_start[0], pt[1] - mouse_start[1])
        lbl_info.config(text=f"Length: {dist_px:.1f}px", fg="#0066cc")

canvas.bind("<Motion>", on_motion)


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
            camera_params=entry.camera_params
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
        btn_frame2.pack(fill=tk.X, padx=10, pady=(3, 10))

        tk.Button(btn_frame2, text="� Capture Perspective", font=("Segoe UI", 9), width=26,
                  bg=SUCCESS_COLOR, fg="white", relief=tk.FLAT, cursor="hand2",
                  command=lambda n=name: take_room_perspective_screenshot(state.room_segments[n], n)).pack(fill=tk.X, padx=2)

# Create action buttons now that functions are defined
tk.Button(action_frame, text="↶ Undo", font=("Segoe UI", 10), width=10,
          bg=DANGER_COLOR, fg="white", relief=tk.FLAT, padx=10, pady=6,
          activebackground="#ff4444", cursor="hand2",
          command=undo).pack(side=tk.LEFT, padx=5)

tk.Button(action_frame, text="📂 View Rooms", font=("Segoe UI", 10), width=12,
          bg=PRIMARY_COLOR, fg="white", relief=tk.FLAT, padx=10, pady=6,
          activebackground="#0052a3", cursor="hand2",
          command=open_room_viewer).pack(side=tk.LEFT, padx=5)

tk.Button(action_frame, text="🎨 Gallery", font=("Segoe UI", 10), width=10,
          bg=ACCENT_COLOR, fg="white", relief=tk.FLAT, padx=10, pady=6,
          activebackground="#d83a50", cursor="hand2",
          command=open_design_gallery).pack(side=tk.LEFT, padx=5)

# Initialize with "outer" mode selected
set_mode("inner")

# ---------- START ----------
load_image(IMAGE_PATH)
redraw()
root.mainloop()
