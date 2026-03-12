"""
Room Gallery & AI Design Manager
Handles screenshot capture, AI generation, and gallery display with modern UI/UX
"""

import os
import json
from datetime import datetime
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Directory structure
GALLERY_DIR = os.path.join(os.getcwd(), "room_designs")
SCREENSHOTS_DIR = os.path.join(GALLERY_DIR, "screenshots")
AI_DESIGNS_DIR = os.path.join(GALLERY_DIR, "ai_generated")
METADATA_FILE = os.path.join(GALLERY_DIR, "designs_metadata.json")

# Create directories
for d in [GALLERY_DIR, SCREENSHOTS_DIR, AI_DESIGNS_DIR]:
    os.makedirs(d, exist_ok=True)

# Modern color scheme (dark theme)
COLORS = {
    "bg": "#1a1a2e",
    "surface": "#16213e",
    "primary": "#0f3460",
    "accent": "#e94560",
    "text": "#ffffff",
    "text_secondary": "#a0a0a0",
    "success": "#00d26a",
    "warning": "#ff9f1c",
    "card": "#1f2940",
    "hover": "#2a3f5f"
}


def load_metadata():
    """Load existing design metadata"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"designs": []}


def save_metadata(data):
    """Save design metadata"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def update_design_entry(design_id: str, updates: dict):
    """Update a design entry in metadata"""
    metadata = load_metadata()
    for i, d in enumerate(metadata["designs"]):
        if d["design_id"] == design_id:
            metadata["designs"][i].update(updates)
            save_metadata(metadata)
            print(f"[OK] Updated design entry: {design_id}")
            return True
    print(f"[WARNING] Design entry not found: {design_id}")
    return False


def generate_design_id():
    """Generate unique design ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class DesignEntry:
    """Represents a single design (screenshot + AI generation)"""
    def __init__(self, design_id, room_name, wall_index, screenshot_path, 
                 ai_path=None, room_type="Living Room", style="Modern",
                 has_doors=False, has_windows=False, visible_openings=None,
                 camera_params=None):
        self.design_id = design_id
        self.room_name = room_name
        self.wall_index = wall_index
        self.screenshot_path = screenshot_path
        self.ai_path = ai_path
        self.room_type = room_type
        self.style = style
        # Actual layout info from the floor plan editor (not auto-detected)
        self.has_doors = has_doors
        self.has_windows = has_windows
        # List of visible openings with 3D geometry:
        # [{"type": "door"/"window", "wall_index": int, "corners_3d": [[x,y,z]×4]}, ...]
        self.visible_openings = visible_openings or []
        # Camera intrinsics + extrinsics for 3D→2D projection
        # {"width": int, "height": int,
        #  "intrinsic": [[3×3 row-major]], "extrinsic": [[4×4 row-major]]}
        self.camera_params = camera_params or {}
        self.created_at = datetime.now().isoformat()
        self.status = "pending" if ai_path is None else "completed"
    
    def to_dict(self):
        return {
            "design_id": self.design_id,
            "room_name": self.room_name,
            "wall_index": self.wall_index,
            "screenshot_path": self.screenshot_path,
            "ai_path": self.ai_path,
            "room_type": self.room_type,
            "style": self.style,
            "has_doors": self.has_doors,
            "has_windows": self.has_windows,
            "visible_openings": self.visible_openings,
            "camera_params": self.camera_params,
            "created_at": self.created_at,
            "status": self.status
        }
    
    @staticmethod
    def from_dict(d):
        entry = DesignEntry(
            d["design_id"], d["room_name"], d["wall_index"],
            d["screenshot_path"], d.get("ai_path"),
            d.get("room_type", "Living Room"), d.get("style", "Modern"),
            d.get("has_doors", False), d.get("has_windows", False),
            d.get("visible_openings", []),
            d.get("camera_params", {})
        )
        entry.created_at = d.get("created_at", datetime.now().isoformat())
        entry.status = d.get("status", "pending")
        return entry


class ModernScrollableFrame(tk.Frame):
    """A scrollable frame with modern styling"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.canvas = tk.Canvas(self, bg=COLORS["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS["bg"])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        
        # Resize canvas window
        self.canvas.bind("<Configure>", self._on_canvas_configure)
    
    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)


class DesignCard(tk.Frame):
    """Modern design card widget"""
    def __init__(self, parent, entry: DesignEntry, on_generate=None, on_view=None, on_delete=None):
        super().__init__(parent, bg=COLORS["card"], relief=tk.FLAT)
        self.entry = entry
        self.on_generate = on_generate
        self.on_view = on_view
        self.on_delete = on_delete
        
        self.configure(padx=2, pady=2)
        
        # Inner frame with rounded corners effect
        inner = tk.Frame(self, bg=COLORS["card"])
        inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header = tk.Frame(inner, bg=COLORS["card"])
        header.pack(fill=tk.X, pady=(0, 10))
        
        # Room name and status
        tk.Label(header, text=f"🏠 {entry.room_name}", font=("Segoe UI", 14, "bold"),
                 bg=COLORS["card"], fg=COLORS["text"]).pack(side=tk.LEFT)
        
        status_color = COLORS["success"] if entry.status == "completed" else COLORS["warning"]
        status_text = "✓ Generated" if entry.status == "completed" else "⏳ Pending"
        tk.Label(header, text=status_text, font=("Segoe UI", 10),
                 bg=COLORS["card"], fg=status_color).pack(side=tk.RIGHT)
        
        # Image container
        img_container = tk.Frame(inner, bg=COLORS["surface"])
        img_container.pack(fill=tk.X, pady=10)
        
        # Screenshot thumbnail
        self.screenshot_label = tk.Label(img_container, bg=COLORS["surface"], 
                                          text="📷 Loading...", fg=COLORS["text_secondary"])
        self.screenshot_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Arrow
        tk.Label(img_container, text="→", font=("Segoe UI", 20, "bold"),
                 bg=COLORS["surface"], fg=COLORS["accent"]).pack(side=tk.LEFT, padx=10)
        
        # AI result thumbnail
        self.ai_label = tk.Label(img_container, bg=COLORS["surface"],
                                  text="🎨 AI Design", fg=COLORS["text_secondary"])
        self.ai_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Load thumbnails
        self.load_thumbnails()
        
        # Info section
        info_frame = tk.Frame(inner, bg=COLORS["card"])
        info_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(info_frame, text=f"Wall {entry.wall_index + 1} • {entry.room_type} • {entry.style}",
                 font=("Segoe UI", 10), bg=COLORS["card"], fg=COLORS["text_secondary"]).pack(side=tk.LEFT)
        
        # Created date
        try:
            created = datetime.fromisoformat(entry.created_at)
            date_str = created.strftime("%b %d, %Y %H:%M")
        except:
            date_str = "Unknown"
        
        tk.Label(info_frame, text=date_str, font=("Segoe UI", 9),
                 bg=COLORS["card"], fg=COLORS["text_secondary"]).pack(side=tk.RIGHT)
        
        # Action buttons
        btn_frame = tk.Frame(inner, bg=COLORS["card"])
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        if entry.status == "pending":
            generate_btn = tk.Button(btn_frame, text="🎨 Generate AI Design",
                                      font=("Segoe UI", 10, "bold"),
                                      bg=COLORS["accent"], fg=COLORS["text"],
                                      relief=tk.FLAT, padx=15, pady=8,
                                      cursor="hand2",
                                      command=lambda: on_generate(entry) if on_generate else None)
            generate_btn.pack(side=tk.LEFT, padx=2)
        
        view_btn = tk.Button(btn_frame, text="👁️ View Full",
                              font=("Segoe UI", 10),
                              bg=COLORS["primary"], fg=COLORS["text"],
                              relief=tk.FLAT, padx=15, pady=8,
                              cursor="hand2",
                              command=lambda: on_view(entry) if on_view else None)
        view_btn.pack(side=tk.LEFT, padx=2)
        
        delete_btn = tk.Button(btn_frame, text="🗑️",
                                font=("Segoe UI", 10),
                                bg=COLORS["surface"], fg=COLORS["accent"],
                                relief=tk.FLAT, padx=10, pady=8,
                                cursor="hand2",
                                command=lambda: on_delete(entry) if on_delete else None)
        delete_btn.pack(side=tk.RIGHT, padx=2)
    
    def load_thumbnails(self):
        """Load and display thumbnail images"""
        try:
            if os.path.exists(self.entry.screenshot_path):
                img = Image.open(self.entry.screenshot_path)
                img.thumbnail((200, 150))
                photo = ImageTk.PhotoImage(img)
                self.screenshot_label.configure(image=photo, text="")
                self.screenshot_label.image = photo
        except Exception as e:
            print(f"Error loading screenshot: {e}")
        
        try:
            if self.entry.ai_path and os.path.exists(self.entry.ai_path):
                img = Image.open(self.entry.ai_path)
                img.thumbnail((200, 150))
                photo = ImageTk.PhotoImage(img)
                self.ai_label.configure(image=photo, text="")
                self.ai_label.image = photo
            else:
                self.ai_label.configure(text="🎨 Not yet\ngenerated",
                                         font=("Segoe UI", 10))
        except Exception as e:
            print(f"Error loading AI image: {e}")


class DesignGallery(tk.Toplevel):
    """Modern gallery window for viewing all designs"""
    def __init__(self, parent, ai_generator_callback=None):
        super().__init__(parent)
        self.title("🎨 Design Gallery")
        self.geometry("900x700")
        self.configure(bg=COLORS["bg"])
        
        self.ai_generator = ai_generator_callback
        self.entries = []
        
        self._build_ui()
        self.load_designs()
    
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=COLORS["surface"], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="🎨 Design Gallery", font=("Segoe UI", 22, "bold"),
                 bg=COLORS["surface"], fg=COLORS["text"]).pack(side=tk.LEFT, padx=25, pady=20)
        
        # Stats
        self.stats_label = tk.Label(header, text="", font=("Segoe UI", 11),
                                     bg=COLORS["surface"], fg=COLORS["text_secondary"])
        self.stats_label.pack(side=tk.RIGHT, padx=25)
        
        # Toolbar
        toolbar = tk.Frame(self, bg=COLORS["bg"], height=50)
        toolbar.pack(fill=tk.X, padx=20, pady=10)
        
        # Filter buttons
        tk.Label(toolbar, text="Filter:", font=("Segoe UI", 10),
                 bg=COLORS["bg"], fg=COLORS["text_secondary"]).pack(side=tk.LEFT)
        
        for text, filter_val in [("All", "all"), ("Pending", "pending"), ("Completed", "completed")]:
            btn = tk.Button(toolbar, text=text, font=("Segoe UI", 9),
                           bg=COLORS["primary"], fg=COLORS["text"],
                           relief=tk.FLAT, padx=12, pady=4,
                           command=lambda f=filter_val: self.filter_designs(f))
            btn.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        tk.Button(toolbar, text="🔄 Refresh", font=("Segoe UI", 10),
                  bg=COLORS["accent"], fg=COLORS["text"],
                  relief=tk.FLAT, padx=15, pady=5,
                  command=self.load_designs).pack(side=tk.RIGHT)
        
        # Scrollable content area
        self.scroll_frame = ModernScrollableFrame(self, bg=COLORS["bg"])
        self.scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.cards_container = self.scroll_frame.scrollable_frame
    
    def load_designs(self, filter_status="all"):
        """Load all designs from metadata"""
        # Clear existing cards
        for widget in self.cards_container.winfo_children():
            widget.destroy()
        
        metadata = load_metadata()
        self.entries = [DesignEntry.from_dict(d) for d in metadata.get("designs", [])]
        
        # Apply filter
        if filter_status != "all":
            filtered = [e for e in self.entries if e.status == filter_status]
        else:
            filtered = self.entries
        
        # Update stats
        total = len(self.entries)
        completed = len([e for e in self.entries if e.status == "completed"])
        self.stats_label.config(text=f"📊 {total} designs • {completed} completed")
        
        # Create cards
        if not filtered:
            tk.Label(self.cards_container, 
                     text="No designs yet.\nCapture screenshots from room perspective views!",
                     font=("Segoe UI", 14), bg=COLORS["bg"], fg=COLORS["text_secondary"],
                     justify=tk.CENTER).pack(pady=50)
        else:
            for entry in reversed(filtered):  # Show newest first
                card = DesignCard(self.cards_container, entry,
                                  on_generate=self.generate_ai_design,
                                  on_view=self.view_full_design,
                                  on_delete=self.delete_design)
                card.pack(fill=tk.X, pady=10)
    
    def filter_designs(self, status):
        self.load_designs(status)
    
    def generate_ai_design(self, entry: DesignEntry):
        """Generate AI design for an entry"""
        if self.ai_generator:
            self.show_generation_dialog(entry)
        else:
            messagebox.showinfo("Info", "AI generator not available")
    
    def show_generation_dialog(self, entry: DesignEntry):
        """Show dialog for AI generation options"""
        dialog = tk.Toplevel(self)
        dialog.title("🎨 Generate AI Design")
        dialog.geometry("450x550")
        dialog.configure(bg=COLORS["bg"])
        dialog.transient(self)
        dialog.grab_set()
        
        # Center dialog
        dialog.geometry(f"+{self.winfo_x() + 200}+{self.winfo_y() + 50}")
        
        # Title
        tk.Label(dialog, text="🎨 AI Interior Design", font=("Segoe UI", 18, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(pady=20)
        
        # Preview
        preview_frame = tk.Frame(dialog, bg=COLORS["surface"])
        preview_frame.pack(fill=tk.X, padx=30, pady=10)
        
        try:
            img = Image.open(entry.screenshot_path)
            img.thumbnail((380, 200))
            photo = ImageTk.PhotoImage(img)
            preview_label = tk.Label(preview_frame, image=photo, bg=COLORS["surface"])
            preview_label.image = photo
            preview_label.pack(pady=10)
        except:
            pass
        
        # Options
        options_frame = tk.Frame(dialog, bg=COLORS["bg"])
        options_frame.pack(fill=tk.X, padx=30, pady=20)
        
        # Room type
        tk.Label(options_frame, text="Room Type:", font=("Segoe UI", 11, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        
        room_types = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Office", "Dining Room"]
        room_var = tk.StringVar(value=entry.room_type)
        room_menu = ttk.Combobox(options_frame, textvariable=room_var, values=room_types,
                                  font=("Segoe UI", 10), state="readonly")
        room_menu.pack(fill=tk.X, pady=(5, 15))
        
        # Style
        tk.Label(options_frame, text="Design Style:", font=("Segoe UI", 11, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        
        styles = ["Modern Minimalist", "Scandinavian", "Industrial", "Bohemian", 
                  "Mid-Century Modern", "Contemporary", "Traditional", "Japandi"]
        style_var = tk.StringVar(value=entry.style)
        style_menu = ttk.Combobox(options_frame, textvariable=style_var, values=styles,
                                   font=("Segoe UI", 10), state="readonly")
        style_menu.pack(fill=tk.X, pady=(5, 15))
        
        # Progress
        progress_label = tk.Label(dialog, text="", font=("Segoe UI", 10),
                                   bg=COLORS["bg"], fg=COLORS["text_secondary"])
        progress_label.pack(pady=5)
        
        progress_bar = ttk.Progressbar(dialog, mode="indeterminate", length=300)
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=COLORS["bg"])
        btn_frame.pack(pady=20)
        
        def start_generation():
            entry.room_type = room_var.get()
            entry.style = style_var.get()
            
            progress_label.config(text="Generating AI design... This may take a few minutes.")
            progress_bar.pack(pady=10)
            progress_bar.start(10)
            
            # Run AI generation in thread
            def generate():
                try:
                    print("[INFO] Starting AI generation thread...")
                    result = self.ai_generator(entry)
                    print(f"[INFO] AI generation returned: {result}")
                    dialog.after(0, lambda: on_generation_complete(result))
                except Exception as e:
                    print(f"[ERROR] AI generation exception: {e}")
                    import traceback
                    traceback.print_exc()
                    dialog.after(0, lambda err=str(e): on_generation_error(err))
            
            threading.Thread(target=generate, daemon=True).start()
        
        def on_generation_complete(result_path):
            progress_bar.stop()
            progress_bar.pack_forget()
            
            if result_path and os.path.exists(result_path):
                entry.ai_path = result_path
                entry.status = "completed"
                self.update_metadata(entry)
                
                progress_label.config(text="Design generated successfully!", fg=COLORS["success"])
                dialog.after(1500, dialog.destroy)
                dialog.after(1600, self.load_designs)
            else:
                progress_label.config(text="Generation failed - no output", fg=COLORS["accent"])
        
        def on_generation_error(error_msg):
            progress_bar.stop()
            progress_bar.pack_forget()
            progress_label.config(text=f"Error: {error_msg}", fg=COLORS["accent"])
        
        tk.Button(btn_frame, text="🎨 Generate", font=("Segoe UI", 11, "bold"),
                  bg=COLORS["accent"], fg=COLORS["text"],
                  relief=tk.FLAT, padx=25, pady=10,
                  command=start_generation).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 11),
                  bg=COLORS["surface"], fg=COLORS["text"],
                  relief=tk.FLAT, padx=25, pady=10,
                  command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def update_metadata(self, entry: DesignEntry):
        """Update entry in metadata file"""
        metadata = load_metadata()
        for i, d in enumerate(metadata["designs"]):
            if d["design_id"] == entry.design_id:
                metadata["designs"][i] = entry.to_dict()
                break
        save_metadata(metadata)
    
    def view_full_design(self, entry: DesignEntry):
        """View full-size images"""
        viewer = tk.Toplevel(self)
        viewer.title(f"🖼️ {entry.room_name} - Wall {entry.wall_index + 1}")
        viewer.geometry("1200x700")
        viewer.configure(bg=COLORS["bg"])
        
        # Header
        tk.Label(viewer, text=f"🏠 {entry.room_name} - Wall {entry.wall_index + 1}",
                 font=("Segoe UI", 18, "bold"),
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(pady=20)
        
        # Images container
        img_frame = tk.Frame(viewer, bg=COLORS["bg"])
        img_frame.pack(fill=tk.BOTH, expand=True, padx=30)
        
        # Screenshot
        left_frame = tk.Frame(img_frame, bg=COLORS["surface"])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(left_frame, text="📷 Original Screenshot", font=("Segoe UI", 12, "bold"),
                 bg=COLORS["surface"], fg=COLORS["text"]).pack(pady=10)
        
        try:
            img1 = Image.open(entry.screenshot_path)
            img1.thumbnail((550, 500))
            photo1 = ImageTk.PhotoImage(img1)
            lbl1 = tk.Label(left_frame, image=photo1, bg=COLORS["surface"])
            lbl1.image = photo1
            lbl1.pack(pady=10)
        except:
            tk.Label(left_frame, text="Image not found", bg=COLORS["surface"],
                     fg=COLORS["text_secondary"]).pack(pady=50)
        
        # AI Design
        right_frame = tk.Frame(img_frame, bg=COLORS["surface"])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(right_frame, text="🎨 AI Generated Design", font=("Segoe UI", 12, "bold"),
                 bg=COLORS["surface"], fg=COLORS["text"]).pack(pady=10)
        
        if entry.ai_path and os.path.exists(entry.ai_path):
            try:
                img2 = Image.open(entry.ai_path)
                img2.thumbnail((550, 500))
                photo2 = ImageTk.PhotoImage(img2)
                lbl2 = tk.Label(right_frame, image=photo2, bg=COLORS["surface"])
                lbl2.image = photo2
                lbl2.pack(pady=10)
            except:
                tk.Label(right_frame, text="Image not found", bg=COLORS["surface"],
                         fg=COLORS["text_secondary"]).pack(pady=50)
        else:
            tk.Label(right_frame, text="Not yet generated\n\nClick 'Generate AI Design'\nin the gallery",
                     font=("Segoe UI", 11), bg=COLORS["surface"],
                     fg=COLORS["text_secondary"], justify=tk.CENTER).pack(pady=50)
        
        # Info bar
        info_bar = tk.Frame(viewer, bg=COLORS["surface"], height=50)
        info_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Label(info_bar, text=f"Room Type: {entry.room_type} • Style: {entry.style}",
                 font=("Segoe UI", 10), bg=COLORS["surface"],
                 fg=COLORS["text_secondary"]).pack(pady=15)
    
    def delete_design(self, entry: DesignEntry):
        """Delete a design entry"""
        if messagebox.askyesno("Delete Design", 
                               f"Delete design for {entry.room_name}?\n\nThis will remove both the screenshot and any AI-generated images."):
            # Remove files
            try:
                if os.path.exists(entry.screenshot_path):
                    os.remove(entry.screenshot_path)
                if entry.ai_path and os.path.exists(entry.ai_path):
                    os.remove(entry.ai_path)
            except Exception as e:
                print(f"Error deleting files: {e}")
            
            # Update metadata
            metadata = load_metadata()
            metadata["designs"] = [d for d in metadata["designs"] 
                                   if d["design_id"] != entry.design_id]
            save_metadata(metadata)
            
            self.load_designs()


def save_screenshot_entry(room_name: str, wall_index: int, screenshot_image,
                          room_type: str = "Living Room",
                          has_doors: bool = False, has_windows: bool = False,
                          visible_openings: list = None,
                          camera_params: dict = None) -> DesignEntry:
    """Save a screenshot and create a new design entry
    
    Args:
        room_name: Name of the room
        wall_index: Index of the removed wall
        screenshot_image: PIL Image or path string
        room_type: Type of room (Living Room, Bedroom, etc.)
        has_doors: Whether the visible walls contain doors (from floor plan data)
        has_windows: Whether the visible walls contain windows (from floor plan data)
        visible_openings: List of dicts with type, wall_index, corners_3d
        camera_params: Dict with intrinsic/extrinsic matrices for 3D→2D projection
    """
    design_id = generate_design_id()
    
    # Save screenshot
    screenshot_filename = f"{design_id}_{room_name.replace(' ', '_')}_wall{wall_index}.png"
    screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_filename)
    
    if isinstance(screenshot_image, Image.Image):
        screenshot_image.save(screenshot_path)
    elif isinstance(screenshot_image, str):  # It's already a path
        screenshot_path = screenshot_image
    
    # Create entry with layout info
    entry = DesignEntry(
        design_id=design_id,
        room_name=room_name,
        wall_index=wall_index,
        screenshot_path=screenshot_path,
        room_type=room_type,
        has_doors=has_doors,
        has_windows=has_windows,
        visible_openings=visible_openings or [],
        camera_params=camera_params or {}
    )
    
    # Save to metadata
    metadata = load_metadata()
    metadata["designs"].append(entry.to_dict())
    save_metadata(metadata)
    
    return entry
