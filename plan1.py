"""
Interactive floorplan editor (desktop) — improved

Features:
- Outer polygon (connected, auto-close when last point near first)
- Multiple inner polygons (each becomes an explicit room)
- Doors/windows drawn by click-and-drag lines (preview while dragging)
- Move vertices by dragging
- Undo, Reset, Export JSON
- Segmentation that respects user geometry and DOES NOT thicken walls/furniture
- After segmentation: asks user whether to keep furniture, empty rooms, or export masks for AI
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import os
import math
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    SHAPELY = True
except Exception:
    SHAPELY = False

# --------- Config ----------
IMAGE_PATH = r"C:\Users\Lenovo\Desktop\Interior_plan\plan\1.jpg"  # default; change via Open Image
OUT_DIR = os.path.join(os.getcwd(), "seg_output")
os.makedirs(OUT_DIR, exist_ok=True)

CANVAS_MAX_W = 1000
CANVAS_MAX_H = 800
SNAP_RADIUS = 10          # px for selecting vertices on canvas
POLYGON_CLOSE_DIST = 20   # px threshold to auto-close polygon (canvas coords)
DOOR_WINDOW_THICKNESS = 10  # px thickness when erasing walls at door/window during segmentation

# --------- App State ----------
class AppState:
    def __init__(self):
        self.image = None            # PIL Image (resized for display)
        self.orig_img_cv = None      # original image as grayscale (cv2)
        self.orig_img_color = None   # original color image (cv2)
        self.scale = 1.0
        self.offset = (0, 0)

        self.outer = []            # list of (x,y) ints (image coords)
        self.outer_closed = False

        self.inners = []           # list of polygons (each list of (x,y) image coords)
        self.current_inner = []    # polygon in progress

        self.doors = []            # list of ((x,y),(x,y))
        self.windows = []          # list of ((x,y),(x,y))

        self.mode = 'outer'        # outer, inner, door, window, move
        self.moving = False
        self.move_ref = None       # (poly_type, poly_index, vert_index, coords, dist)
        self.undo_stack = []

state = AppState()

# --------- Utilities ----------
def load_image(path):
    img_pil = Image.open(path).convert("RGB")
    w,h = img_pil.size
    scale = min(1.0, CANVAS_MAX_W / w, CANVAS_MAX_H / h)
    state.scale = scale
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = img_pil.resize((new_w,new_h), Image.LANCZOS)
    state.image = img_resized
    state.orig_img_cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    state.orig_img_color = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    state.offset = (0,0)
    return img_resized

def img2canvas_coords(pt):
    x = int(pt[0]*state.scale + state.offset[0])
    y = int(pt[1]*state.scale + state.offset[1])
    return (x,y)

def canvas2img_coords(pt):
    x = int((pt[0] - state.offset[0]) / state.scale)
    y = int((pt[1] - state.offset[1]) / state.scale)
    return (x,y)

def distance(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def find_nearest_vertex(canvas_pt):
    cx, cy = canvas_pt
    best = (None, None, None, None, 9999.0)
    # outer
    for i,p in enumerate(state.outer):
        c = img2canvas_coords(p)
        d = distance((cx,cy), c)
        if d < best[4] and d <= SNAP_RADIUS:
            best = ('outer', 0, i, p, d)
    # inners (including current)
    for pi,poly in enumerate(state.inners + ([state.current_inner] if state.current_inner else [])):
        for vi,p in enumerate(poly):
            c = img2canvas_coords(p)
            d = distance((cx,cy), c)
            if d < best[4] and d <= SNAP_RADIUS:
                best = ('inner', pi, vi, p, d)
    # doors
    for di,(p1,p2) in enumerate(state.doors):
        for vi,p in enumerate([p1,p2]):
            c = img2canvas_coords(p)
            d = distance((cx,cy), c)
            if d < best[4] and d <= SNAP_RADIUS:
                best = ('door', di, vi, p, d)
    # windows
    for wi,(p1,p2) in enumerate(state.windows):
        for vi,p in enumerate([p1,p2]):
            c = img2canvas_coords(p)
            d = distance((cx,cy), c)
            if d < best[4] and d <= SNAP_RADIUS:
                best = ('window', wi, vi, p, d)
    if best[0] is None:
        return None
    return best  # (type, poly_index, vert_index, coords, dist)

# --------- Tk UI ----------
root = tk.Tk()
root.title("Floorplan Editor — Toolbar Mode")

toolbar = tk.Frame(root)
toolbar.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

canvas_frame = tk.Frame(root)
canvas_frame.pack(side=tk.RIGHT, padx=6, pady=6)
canvas = tk.Canvas(canvas_frame, width=CANVAS_MAX_W, height=CANVAS_MAX_H, bg='#eee')
canvas.pack()

# --------- Toolbar buttons ----------
def choose_image():
    path = filedialog.askopenfilename(title="Select floorplan image", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
    if not path:
        return
    global IMAGE_PATH
    IMAGE_PATH = path
    load_image(path)
    redraw_all()

btn_choose = tk.Button(toolbar, text="Open Image", width=18, command=choose_image)
btn_choose.pack(pady=4)

def set_mode(m):
    state.mode = m
    lbl_mode.config(text=f"Mode: {m}")

btn_outer = tk.Button(toolbar, text="Outer (Polygon)", width=18, command=lambda:set_mode('outer'))
btn_outer.pack(pady=2)

btn_inner = tk.Button(toolbar, text="Inner (Polygon)", width=18, command=lambda:set_mode('inner'))
btn_inner.pack(pady=2)

btn_door = tk.Button(toolbar, text="Door (drag)", width=18, command=lambda:set_mode('door'))
btn_door.pack(pady=2)

btn_window = tk.Button(toolbar, text="Window (drag)", width=18, command=lambda:set_mode('window'))
btn_window.pack(pady=2)

btn_move = tk.Button(toolbar, text="Move Vertex", width=18, command=lambda:set_mode('move'))
btn_move.pack(pady=6)

def finish_polygon():
    if state.mode == 'outer':
        if len(state.outer) >= 3:
            state.outer_closed = True
            state.undo_stack.append(('outer_close', None))
            redraw_all()
    elif state.mode == 'inner':
        if len(state.current_inner) >= 3:
            state.inners.append(list(state.current_inner))
            state.undo_stack.append(('inner_finish', None))
            state.current_inner = []
            redraw_all()

btn_finish = tk.Button(toolbar, text="Finish Polygon", width=18, command=finish_polygon)
btn_finish.pack(pady=2)

def undo_action():
    # context-aware undo
    if state.mode == 'outer':
        if state.outer and not state.outer_closed:
            state.outer.pop()
            redraw_all()
            return
        if state.undo_stack:
            typ, val = state.undo_stack.pop()
            if typ == 'outer_close':
                state.outer_closed = False
                redraw_all()
                return
    if state.mode == 'inner':
        if state.current_inner:
            state.current_inner.pop()
            redraw_all()
            return
        if state.inners:
            state.inners.pop()
            redraw_all()
            return
    # fallback: undo doors/windows
    if state.doors:
        state.doors.pop()
        redraw_all()
        return
    if state.windows:
        state.windows.pop()
        redraw_all()
        return

btn_undo = tk.Button(toolbar, text="Undo", width=18, command=undo_action)
btn_undo.pack(pady=2)

def reset_all():
    if messagebox.askyesno("Reset", "Reset all geometry?"):
        state.outer = []
        state.outer_closed = False
        state.inners = []
        state.current_inner = []
        state.doors = []
        state.windows = []
        state.undo_stack = []
        redraw_all()

btn_reset = tk.Button(toolbar, text="Reset", width=18, command=reset_all)
btn_reset.pack(pady=2)

def export_json():
    data = {
        'outer': [{'x':int(x),'y':int(y)} for (x,y) in state.outer],
        'inners': [[{'x':int(x),'y':int(y)} for (x,y) in poly] for poly in state.inners],
        'doors': [[{'x':int(p[0]),'y':int(p[1])} for p in seg] for seg in state.doors],
        'windows': [[{'x':int(p[0]),'y':int(p[1])} for p in seg] for seg in state.windows],
    }
    out_path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')], initialfile='plan_geometry.json')
    if not out_path: return
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    messagebox.showinfo("Export", f"Saved {out_path}")

btn_export = tk.Button(toolbar, text="Export JSON", width=18, command=export_json)
btn_export.pack(pady=6)

lbl_mode = tk.Label(toolbar, text=f"Mode: {state.mode}")
lbl_mode.pack(pady=6)

# --------- Segmentation & post-choice logic ----------
def ask_post_seg_choice_and_act(room_masks, seg_color_img, poly_mask):
    """
    room_masks: list of binary masks (H,W) for detected rooms (user inners first, then auto rooms)
    seg_color_img: HxWx3 colored segmentation (rooms colored)
    poly_mask: polygon mask of outer polygon (H,W)
    """
    # Ask user: Keep furniture / Empty rooms / Export masks for AI
    dlg = tk.Toplevel(root)
    dlg.title("Post-segmentation options")
    tk.Label(dlg, text="What do you want to do with furniture inside rooms?").pack(padx=12, pady=8)

    result = {'choice': None}

    def keep():
        result['choice'] = 'keep'
        dlg.destroy()
    def empty():
        result['choice'] = 'empty'
        dlg.destroy()
    def export_masks():
        result['choice'] = 'export'
        dlg.destroy()

    btnf = tk.Frame(dlg); btnf.pack(padx=8, pady=8)
    tk.Button(btnf, text="Keep original furniture (show preview)", width=28, command=keep).grid(row=0,column=0,padx=6,pady=6)
    tk.Button(btnf, text="Empty rooms (furniture removed)", width=28, command=empty).grid(row=1,column=0,padx=6,pady=6)
    tk.Button(btnf, text="Export room masks for AI", width=28, command=export_masks).grid(row=2,column=0,padx=6,pady=6)

    dlg.transient(root)
    dlg.grab_set()
    root.wait_window(dlg)

    choice = result['choice']
    H,W = seg_color_img.shape[:2]
    if choice == 'keep':
        # Composite original image (color) inside each room mask and overlay thin segmentation border
        color_orig = state.orig_img_color.copy()
        color_orig_rgb = cv2.cvtColor(color_orig, cv2.COLOR_BGR2RGB)
        out = seg_color_img.copy()
        for i,m in enumerate(room_masks):
            # overlay original image into that room
            out[m==255] = color_orig_rgb[m==255]
        # show and save
        out_path = os.path.join(OUT_DIR, "rooms_with_furniture.png")
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_bgr)
        # popup view
        im = Image.fromarray(out)
        im.thumbnail((900,700), Image.LANCZOS)
        win = tk.Toplevel(root)
        win.title("Rooms with furniture (preview)")
        tk_img = ImageTk.PhotoImage(im)
        lbl = tk.Label(win, image=tk_img); lbl.image = tk_img; lbl.pack()
        messagebox.showinfo("Saved", f"Saved preview with furniture to {out_path}")
        
    elif choice == 'empty':
        # Uses the user-provided room_masks (list of binary masks) to aggressively remove ANY content
        # inside rooms while preserving walls/doors/windows (which are black strokes).
        H, W = state.orig_img_cv.shape
        img_rgb = cv2.cvtColor(state.orig_img_color, cv2.COLOR_BGR2RGB)
        black_thr = 20
        black_mask = np.all(img_rgb <= black_thr, axis=2)  # True where black lines exist

        clean_seg = seg_color_img.copy()  # colored segmentation RGB

        # kernel: small closing to fill thin furniture strokes inside each room
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

        for rm in room_masks:
            room_bin = (rm == 255).astype(np.uint8) * 255
            if room_bin.sum() == 0:
                continue

            # close inside this room to bridge thin black furniture lines (but we operate per room mask only)
            # To avoid modifying outside room, mask the region and apply closing on the mask crop
            x,y,w,h = cv2.boundingRect(room_bin)
            crop_mask = room_bin[y:y+h, x:x+w]
            closed_crop = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            closed_room = np.zeros_like(room_bin)
            closed_room[y:y+h, x:x+w] = closed_crop

            # Optionally erase any pixels that are actually walls (black lines) before filling:
            closed_room[black_mask] = 0

            # choose room color robustly from seg_color_img (the colored segmentation preview)
            ys, xs = np.where(room_bin == 255)
            room_color = None
            for yy,xx in zip(ys, xs):
                px = clean_seg[yy, xx]
                if not (px[0] == 0 and px[1] == 0 and px[2] == 0):
                    room_color = px.copy()
                    break
            if room_color is None:
                room_color = np.median(clean_seg[room_bin==255].reshape(-1,3), axis=0).astype(np.uint8)

            # Fill the closed_room area with the room color (this removes furniture inside)
            closed_inds = (closed_room == 255)
            clean_seg[closed_inds] = room_color

        # Save and preview
        out_path = os.path.join(OUT_DIR, "rooms_empty_preview.png")
        cv2.imwrite(out_path, cv2.cvtColor(clean_seg, cv2.COLOR_RGB2BGR))
        # ... then display/save as you already do in your code

    elif choice == 'export':
        # Save each room mask and a JSON mapping for AI replacement
        masks_dir = os.path.join(OUT_DIR, "room_masks")
        os.makedirs(masks_dir, exist_ok=True)
        meta = {'rooms': []}
        for i,m in enumerate(room_masks):
            fname = f"room_{i+1:02d}.png"
            path = os.path.join(masks_dir, fname)
            cv2.imwrite(path, m)
            meta['rooms'].append({'mask': fname})
        with open(os.path.join(masks_dir, "rooms_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        messagebox.showinfo("Export", f"Wrote {len(room_masks)} masks to {masks_dir}")
    else:
        messagebox.showinfo("Cancelled", "No post-segmentation action taken.")

def segment_and_show():
    if not state.outer_closed:
        messagebox.showwarning("Segment", "Finish outer polygon first.")
        return
    if state.orig_img_cv is None:
        messagebox.showwarning("Segment", "Load an image first.")
        return

    H,W = state.orig_img_cv.shape
    # rasterize outer polygon (image coords)
    poly_pts = np.array([[p[0], p[1]] for p in state.outer], dtype=np.int32)
    poly_mask = np.zeros((H,W), np.uint8)
    cv2.fillPoly(poly_mask, [poly_pts], 255)

    # build candidate walls from original image but do NOT thicken
    th = cv2.threshold(state.orig_img_cv, 200, 255, cv2.THRESH_BINARY_INV)[1]
    # only consider walls inside polygon
    walls = cv2.bitwise_and(th, poly_mask)

    # ensure outer polygon boundary exists as wall (draw thin boundary, but do not thicken everything)
    cv2.polylines(walls, [poly_pts], isClosed=True, color=255, thickness=2)

    # apply doors/windows: erase the wall where user drew them (respect thickness param)
    for (p1,p2) in state.doors:
        p1_i = (int(p1[0]), int(p1[1])); p2_i = (int(p2[0]), int(p2[1]))
        cv2.line(walls, p1_i, p2_i, 0, DOOR_WINDOW_THICKNESS)
    for (p1,p2) in state.windows:
        p1_i = (int(p1[0]), int(p1[1])); p2_i = (int(p2[0]), int(p2[1]))
        cv2.line(walls, p1_i, p2_i, 0, DOOR_WINDOW_THICKNESS)

    # final walls are now the exact thin walls; do not dilate or open/close aggressively
    final_walls = walls.copy()

    # interiors = polygon area minus walls
    interiors_all = cv2.bitwise_and(cv2.bitwise_not(final_walls), poly_mask)

    # ---- Create room masks from user inner polygons first (each becomes its own room) ----
    room_masks = []
    for poly in state.inners:
        if len(poly) < 3: continue
        pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
        mask = np.zeros((H,W), np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        # intersect with outer polygon area
        mask = cv2.bitwise_and(mask, poly_mask)
        # remove walls
        mask[final_walls == 255] = 0
        if cv2.countNonZero(mask) > 0:
            room_masks.append(mask)

    # subtract inner polygons from interiors to avoid double-segmentation
    interiors_remainder = interiors_all.copy()
    for m in room_masks:
        interiors_remainder[m == 255] = 0

    # auto-segment remaining interiors (connected components)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(interiors_remainder)
    auto_room_masks = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 500:
            continue
        mask_i = (labels == i).astype(np.uint8) * 255
        auto_room_masks.append(mask_i)

    # combine user inner rooms + auto rooms
    seg_color = np.zeros((H,W,3), dtype=np.uint8)
    all_room_masks = []
    for m in room_masks + auto_room_masks:
        col = [int(x) for x in np.random.randint(60,255,3)]
        seg_color[m == 255] = col
        all_room_masks.append(m)

    # Save preliminary segmentation image
    seg_preview_path = os.path.join(OUT_DIR, "segmented_preview.png")
    cv2.imwrite(seg_preview_path, cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))

    # Show preview
    im = Image.fromarray(seg_color)
    im.thumbnail((900,700), Image.LANCZOS)
    win = tk.Toplevel(root)
    win.title("Segmentation Preview")
    tk_img = ImageTk.PhotoImage(im)
    lbl = tk.Label(win, image=tk_img); lbl.image = tk_img; lbl.pack()

    # Ask the user what to do with furniture (keep/empty/export masks)
    ask_post_seg_choice_and_act(all_room_masks, seg_color, poly_mask)

btn_segment = tk.Button(toolbar, text="Segment", width=18, command=segment_and_show)
btn_segment.pack(pady=4)

# --------- Canvas drawing & interactions ----------
def redraw_all():
    canvas.delete("all")
    if state.image is None:
        try:
            load_image(IMAGE_PATH)
        except Exception as e:
            print("Could not load image:", e)
            return
    imtk = ImageTk.PhotoImage(state.image)
    canvas.image = imtk
    canvas.create_image(0,0,anchor='nw',image=imtk)

    # draw outer polygon lines & verts
    if state.outer:
        pts = [img2canvas_coords(p) for p in state.outer]
        if len(pts) > 1:
            canvas.create_line(*sum(pts,()), fill='#0033aa', width=2)
        for (x,y) in pts:
            canvas.create_oval(x-5,y-5,x+5,y+5,fill='white',outline='black')
    if state.outer_closed:
        if len(state.outer) >= 3:
            a = img2canvas_coords(state.outer[-1])
            b = img2canvas_coords(state.outer[0])
            canvas.create_line(a[0],a[1], b[0],b[1], fill='#0033aa', width=2)

    # inner polygons (finished)
    for poly in state.inners:
        pts = [img2canvas_coords(p) for p in poly]
        if len(pts) > 1:
            canvas.create_line(*sum(pts,()), fill='#00aa33', width=2)
        for (x,y) in pts:
            canvas.create_oval(x-4,y-4,x+4,y+4, fill='white', outline='black')
    # current inner polygon (in-progress)
    if state.current_inner:
        pts = [img2canvas_coords(p) for p in state.current_inner]
        if len(pts) > 1:
            canvas.create_line(*sum(pts,()), fill='#00aa33', width=2, dash=(4,3))
        for (x,y) in pts:
            canvas.create_oval(x-4,y-4,x+4,y+4, fill='white', outline='black')

    # doors
    for (p1,p2) in state.doors:
        a = img2canvas_coords(p1); b = img2canvas_coords(p2)
        canvas.create_line(a[0],a[1], b[0],b[1], fill='#ff8800', width=6)

    # windows
    for (p1,p2) in state.windows:
        a = img2canvas_coords(p1); b = img2canvas_coords(p2)
        canvas.create_line(a[0],a[1], b[0],b[1], fill='#00cccc', width=6)


# mouse state for drag preview and moving
mouse_state = {
    'dragging': False,
    'drawing_door': False,
    'drawing_window': False,
    'start_pt': None
}

def on_canvas_click(event):
    cx, cy = event.x, event.y
    img_pt = canvas2img_coords((cx,cy))

    # MOVE MODE
    if state.mode == 'move':
        n = find_nearest_vertex((cx,cy))
        if n:
            state.moving = True
            state.move_ref = n
            mouse_state['dragging'] = True
        return

   # DOOR: select 2 points (no drag)
    if state.mode == 'door':
        if mouse_state['start_pt'] is None:
            mouse_state['start_pt'] = img_pt
        else:
            state.doors.append((mouse_state['start_pt'], img_pt))
            mouse_state['start_pt'] = None
            redraw_all()
        return

    # WINDOW: select 2 points (no drag)
    if state.mode == 'window':
        if mouse_state['start_pt'] is None:
            mouse_state['start_pt'] = img_pt
        else:
            state.windows.append((mouse_state['start_pt'], img_pt))
            mouse_state['start_pt'] = None
            redraw_all()
        return


    # OUTER polygon drawing (auto-close when near first point)
    if state.mode == 'outer' and not state.outer_closed:
        if len(state.outer) >= 2:
            first_canvas = img2canvas_coords(state.outer[0])
            if distance((cx,cy), first_canvas) <= POLYGON_CLOSE_DIST:
                state.outer_closed = True
                redraw_all()
                return
        state.outer.append(img_pt)
        redraw_all()
        return

    # INNER polygon drawing (auto-close when near first)
    if state.mode == 'inner':
        if len(state.current_inner) >= 2:
            first_canvas = img2canvas_coords(state.current_inner[0])
            if distance((cx,cy), first_canvas) <= POLYGON_CLOSE_DIST:
                state.inners.append(list(state.current_inner))
                state.current_inner = []
                redraw_all()
                return
        state.current_inner.append(img_pt)
        redraw_all()
        return

def on_canvas_motion(event):
    cx, cy = event.x, event.y

    # dragging vertex
    if state.moving and state.move_ref:
        ix, iy = canvas2img_coords((cx,cy))
        typ, pidx, vidx, coords, d = state.move_ref
        if typ == 'outer':
            state.outer[vidx] = (ix, iy)
        elif typ == 'inner':
            if pidx < len(state.inners):
                state.inners[pidx][vidx] = (ix, iy)
            else:
                state.current_inner[vidx] = (ix, iy)
        elif typ == 'door':
            p1,p2 = state.doors[pidx]
            if vidx == 0:
                state.doors[pidx] = ((ix,iy), p2)
            else:
                state.doors[pidx] = (p1, (ix,iy))
        elif typ == 'window':
            p1,p2 = state.windows[pidx]
            if vidx == 0:
                state.windows[pidx] = ((ix,iy), p2)
            else:
                state.windows[pidx] = (p1, (ix,iy))
        state.move_ref = (typ,pidx,vidx,(ix,iy),0)
        redraw_all()
        return

    # preview line for door/window while dragging
    if mouse_state['drawing_door'] or mouse_state['drawing_window']:
        canvas.delete("preview_line")
        sp = mouse_state['start_pt']
        sp_canvas = img2canvas_coords(sp)
        canvas.create_line(sp_canvas[0], sp_canvas[1], cx, cy,
                           fill="#ff8800" if mouse_state['drawing_door'] else "#00cccc",
                           width=6, tag="preview_line")
        return

def on_canvas_release(event):
    cx, cy = event.x, event.y
    img_pt = canvas2img_coords((cx,cy))

    # finish moving
    if state.moving:
        state.moving = False
        state.move_ref = None
        mouse_state['dragging'] = False
        redraw_all()
        return

    # finish door
    if mouse_state['drawing_door']:
        start = mouse_state['start_pt']
        state.doors.append((start, img_pt))
        mouse_state['drawing_door'] = False
        mouse_state['start_pt'] = None
        canvas.delete("preview_line")
        redraw_all()
        return

    # finish window
    if mouse_state['drawing_window']:
        start = mouse_state['start_pt']
        state.windows.append((start, img_pt))
        mouse_state['drawing_window'] = False
        mouse_state['start_pt'] = None
        canvas.delete("preview_line")
        redraw_all()
        return

canvas.bind("<Button-1>", on_canvas_click)
canvas.bind("<B1-Motion>", on_canvas_motion)
canvas.bind("<ButtonRelease-1>", on_canvas_release)

# initialize
try:
    load_image(IMAGE_PATH)
except Exception as e:
    print("No default image loaded; use Open Image button.", e)
redraw_all()

root.mainloop()
