import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import threading
from typing import List, Dict, Callable, Optional
from scipy.ndimage import zoom, gaussian_filter

# Try to import the backend logic
try:
    import tpms_hybrid_v2 as backend
    # Verify backend has rotation support
    if not hasattr(backend, 'generate_rotation_field_from_3x3x3'):
        raise ImportError("Backend missing rotation support")
except ImportError:
    try:
        import tpms_hybrid_expand as backend
    except ImportError:
        # Add current directory to path if running directly
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            import tpms_hybrid_v2 as backend
        except ImportError:
            try:
                import tpms_hybrid as backend
            except ImportError as e:
                messagebox.showerror("Error", f"Could not import backend modules: {e}")
                sys.exit(1)

# Available TPMS types from the backend
AVAILABLE_TPMS = list(backend.TPMS_FUNCTIONS.keys())
TPMS_COLORS = backend.TPMS_COLOR_PAIRS


class ToolTip:
    """简单的工具提示类"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)
    
    def show(self, event=None):
        if self.tip_window:
            return
        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
        label.pack()
    
    def hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class TPMSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TPMS Hybrid Generator UI")
        self.root.geometry("1200x800")
        
        # --- Data Model ---
        self.grid_size = [5, 5, 6]  # X, Y, Z
        self.resolution = 120
        self.selected_tpms_names = ['Gyroid', 'Schoen_IWP']  # Default selection
        
        # Initialize Grids
        self.init_grids()
        
        # UI State
        self.current_z = 0
        self.selected_cells = set()  # Set of (x, y) tuples
        self.mode = "weights"  # "weights" or "density"
        self.is_generating = False  # 生成状态标志
        
        # Undo/Redo 历史
        self.history: List[Dict] = []
        self.history_index = -1
        self.max_history = 50
        
        # --- UI Layout ---
        self.create_menu()
        
        # Main Container
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel: Configuration
        self.left_frame = ttk.Frame(self.main_paned, width=300)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Center Panel: Grid Editor
        self.center_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.center_frame, weight=3)
        
        # Right Panel: Cell Inspector / Tools
        self.right_frame = ttk.Frame(self.main_paned, width=250)
        self.main_paned.add(self.right_frame, weight=1)
        
        self.setup_left_panel()
        self.setup_center_panel()
        self.setup_right_panel()
        self.setup_status_bar()
        
        # 绑定快捷键
        self.bind_shortcuts()
        
        # Initial Draw
        self.update_grid_view()
        self.update_inspector()
        self.save_state()  # 保存初始状态
        self.update_status()

    def init_grids(self):
        """Initialize weight and density grids based on current size and TPMS selection."""
        n_tpms = len(self.selected_tpms_names)
        
        # Weights: (X, Y, Z, N)
        # Default: First TPMS has 1.0 weight
        self.weights = np.zeros((*self.grid_size, n_tpms), dtype=np.float32)
        if n_tpms > 0:
            self.weights[..., 0] = 1.0
            
        # Density: (X, Y, Z)
        # Default: 0.3
        self.density = np.full(self.grid_size, 0.3, dtype=np.float32)

        # Rotation: (X, Y, Z, 3) -> [rot_x, rot_y, rot_z] in degrees
        self.rotation = np.zeros((*self.grid_size, 3), dtype=np.float32)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # File 菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Configuration", command=self.save_config, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Configuration", command=self.load_config, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export STL", command=self.generate_stl)
        file_menu.add_command(label="Export Python Code", command=self.export_python_code)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Alt+F4")
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit 菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All (Current Layer)", command=self.select_all, accelerator="Ctrl+A")
        edit_menu.add_command(label="Clear Selection", command=self.clear_selection, accelerator="Escape")
        edit_menu.add_separator()
        edit_menu.add_command(label="Copy Layer", command=self.copy_layer, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste Layer", command=self.paste_layer, accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="Fill All with Current", command=self.fill_all_cells)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # View 菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Previous Layer", command=self.prev_layer, accelerator="Page Up")
        view_menu.add_command(label="Next Layer", command=self.next_layer, accelerator="Page Down")
        view_menu.add_separator()
        view_menu.add_command(label="3D Preview", command=self.show_3d_preview, accelerator="F5")
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help 菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts_help)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def setup_left_panel(self):
        # --- Grid Settings ---
        grp_grid = ttk.LabelFrame(self.left_frame, text="Grid Settings")
        grp_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(grp_grid, text="Size (X, Y, Z):").grid(row=0, column=0, padx=5, pady=2)
        
        frame_size = ttk.Frame(grp_grid)
        frame_size.grid(row=0, column=1, padx=5, pady=2)
        
        self.var_x = tk.IntVar(value=self.grid_size[0])
        self.var_y = tk.IntVar(value=self.grid_size[1])
        self.var_z = tk.IntVar(value=self.grid_size[2])
        
        ttk.Entry(frame_size, textvariable=self.var_x, width=3).pack(side=tk.LEFT)
        ttk.Label(frame_size, text="x").pack(side=tk.LEFT)
        ttk.Entry(frame_size, textvariable=self.var_y, width=3).pack(side=tk.LEFT)
        ttk.Label(frame_size, text="x").pack(side=tk.LEFT)
        ttk.Entry(frame_size, textvariable=self.var_z, width=3).pack(side=tk.LEFT)
        
        btn_resize = ttk.Button(grp_grid, text="Resize Grid", command=self.resize_grid)
        btn_resize.grid(row=1, column=0, columnspan=2, pady=5)
        ToolTip(btn_resize, "调整网格尺寸 (会保留现有数据)")
        
        # --- TPMS Selection ---
        grp_tpms = ttk.LabelFrame(self.left_frame, text="TPMS Types")
        grp_tpms.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tpms_listbox = tk.Listbox(grp_tpms, selectmode=tk.MULTIPLE, height=10)
        self.tpms_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        for name in AVAILABLE_TPMS:
            self.tpms_listbox.insert(tk.END, name)
            
        # Select current ones
        self.update_listbox_selection()
        
        btn_update_tpms = ttk.Button(grp_tpms, text="Update TPMS List", command=self.update_tpms_selection)
        btn_update_tpms.pack(fill=tk.X, padx=5, pady=5)
        ToolTip(btn_update_tpms, "应用选中的 TPMS 类型到网格")
        
        # --- Generation Settings ---
        grp_gen = ttk.LabelFrame(self.left_frame, text="Generation")
        grp_gen.pack(fill=tk.X, padx=5, pady=5)
        
        # Resolution
        ttk.Label(grp_gen, text="Resolution:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.var_res = tk.IntVar(value=self.resolution)
        ttk.Entry(grp_gen, textvariable=self.var_res, width=8).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        
        # Replicate
        ttk.Label(grp_gen, text="Replicate (X,Y,Z):").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        frame_rep = ttk.Frame(grp_gen)
        frame_rep.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        
        self.var_rep_x = tk.IntVar(value=1)
        self.var_rep_y = tk.IntVar(value=1)
        self.var_rep_z = tk.IntVar(value=1)
        
        ttk.Entry(frame_rep, textvariable=self.var_rep_x, width=3).pack(side=tk.LEFT)
        ttk.Label(frame_rep, text="x").pack(side=tk.LEFT)
        ttk.Entry(frame_rep, textvariable=self.var_rep_y, width=3).pack(side=tk.LEFT)
        ttk.Label(frame_rep, text="x").pack(side=tk.LEFT)
        ttk.Entry(frame_rep, textvariable=self.var_rep_z, width=3).pack(side=tk.LEFT)

        # Desired Size
        ttk.Label(grp_gen, text="Desired Size:").grid(row=2, column=0, padx=5, pady=2, sticky="e")
        self.var_size = tk.DoubleVar(value=17.0)
        ttk.Entry(grp_gen, textvariable=self.var_size, width=8).grid(row=2, column=1, padx=5, pady=2, sticky="w")
        
        # Solid Threshold
        ttk.Label(grp_gen, text="Solid Threshold:").grid(row=3, column=0, padx=5, pady=2, sticky="e")
        self.var_threshold = tk.DoubleVar(value=0.3)
        ttk.Entry(grp_gen, textvariable=self.var_threshold, width=8).grid(row=3, column=1, padx=5, pady=2, sticky="w")
        
        # Smooth Sigma
        ttk.Label(grp_gen, text="Smooth Sigma:").grid(row=4, column=0, padx=5, pady=2, sticky="e")
        self.var_smooth_sigma = tk.DoubleVar(value=0.5)
        ttk.Entry(grp_gen, textvariable=self.var_smooth_sigma, width=8).grid(row=4, column=1, padx=5, pady=2, sticky="w")
        
        self.btn_generate = ttk.Button(grp_gen, text="Generate STL", command=self.generate_stl)
        self.btn_generate.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        ToolTip(self.btn_generate, "生成 STL 文件 (耗时操作)")
        
        btn_preview = ttk.Button(grp_gen, text="3D Grid Preview", command=self.show_3d_preview)
        btn_preview.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")
        ToolTip(btn_preview, "预览 3D 权重网格 (F5)")

        # --- Fluid Domain Settings ---
        grp_fluid = ttk.LabelFrame(self.left_frame, text="Fluid Domain")
        grp_fluid.pack(fill=tk.X, padx=5, pady=5)
        
        self.var_gen_fluid = tk.BooleanVar(value=True)
        ttk.Checkbutton(grp_fluid, text="Generate Fluid Domain", variable=self.var_gen_fluid).grid(row=0, column=0, columnspan=2, sticky="w", padx=5)
        
        ttk.Label(grp_fluid, text="Z-Extension:").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.var_z_ext = tk.DoubleVar(value=0.1)
        ttk.Entry(grp_fluid, textvariable=self.var_z_ext, width=8).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        # --- Presets & IO ---
        grp_io = ttk.LabelFrame(self.left_frame, text="Presets & Export")
        grp_io.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(grp_io)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Save .npz", command=self.save_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Load .npz", command=self.load_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(grp_io, text="Export as Python Code", command=self.export_python_code).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(grp_io, text="Load from Code (5x5x6)", command=self.load_from_code_module).pack(fill=tk.X, padx=5, pady=2)

    def setup_center_panel(self):
        # --- Toolbar ---
        toolbar = ttk.Frame(self.center_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=5)
        self.var_mode = tk.StringVar(value="weights")
        ttk.Radiobutton(toolbar, text="Weights", variable=self.var_mode, value="weights", command=self.change_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(toolbar, text="Density", variable=self.var_mode, value="density", command=self.change_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(toolbar, text="Rotation", variable=self.var_mode, value="rotation", command=self.change_mode).pack(side=tk.LEFT)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(toolbar, text="Z-Layer:").pack(side=tk.LEFT, padx=5)
        self.scale_z = tk.Scale(toolbar, from_=0, to=self.grid_size[2]-1, orient=tk.HORIZONTAL, command=self.on_z_change)
        self.scale_z.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # --- Grid Canvas ---
        self.canvas_frame = ttk.Frame(self.center_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#f0f0f0")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Control-Button-1>", self.on_canvas_ctrl_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)  # 右键菜单
        
        # 创建右键菜单
        self.context_menu = tk.Menu(self.canvas, tearoff=0)
        self.context_menu.add_command(label="Select All", command=self.select_all)
        self.context_menu.add_command(label="Clear Selection", command=self.clear_selection)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Fill Selected with Default", command=self.fill_selected_default)
        self.context_menu.add_command(label="Copy to All Layers", command=self.copy_to_all_layers)

    def setup_right_panel(self):
        self.inspector_label = ttk.Label(self.right_frame, text="Cell Inspector", font=("Arial", 12, "bold"))
        self.inspector_label.pack(pady=10)
        
        self.inspector_content = ttk.Frame(self.right_frame)
        self.inspector_content.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # TPMS 颜色图例
        legend_frame = ttk.LabelFrame(self.right_frame, text="TPMS Legend")
        legend_frame.pack(fill=tk.X, padx=5, pady=10, side=tk.BOTTOM)
        self.legend_content = ttk.Frame(legend_frame)
        self.legend_content.pack(fill=tk.X, padx=5, pady=5)
        self.update_legend()
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, length=150, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        self.progress_bar.pack_forget()  # 初始隐藏
    
    def update_legend(self):
        """更新 TPMS 颜色图例"""
        for widget in self.legend_content.winfo_children():
            widget.destroy()
        
        for name in self.selected_tpms_names:
            frame = ttk.Frame(self.legend_content)
            frame.pack(fill=tk.X, pady=1)
            
            hex_color = TPMS_COLORS.get(name, ("#888888",))[0]
            color_box = tk.Label(frame, bg=hex_color, width=2, height=1)
            color_box.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(frame, text=name, font=("Arial", 8)).pack(side=tk.LEFT, padx=2)
    
    def update_status(self, message: Optional[str] = None):
        """更新状态栏"""
        if message:
            self.status_label.config(text=message)
        else:
            sel_count = len(self.selected_cells)
            grid_info = f"Grid: {self.grid_size[0]}x{self.grid_size[1]}x{self.grid_size[2]}"
            layer_info = f"Layer: {self.current_z + 1}/{self.grid_size[2]}"
            sel_info = f"Selected: {sel_count} cells" if sel_count else "No selection"
            self.status_label.config(text=f"{grid_info} | {layer_info} | {sel_info}")
    
    def bind_shortcuts(self):
        """绑定键盘快捷键"""
        self.root.bind("<Control-s>", lambda e: self.save_config())
        self.root.bind("<Control-o>", lambda e: self.load_config())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-a>", lambda e: self.select_all())
        self.root.bind("<Escape>", lambda e: self.clear_selection())
        self.root.bind("<Control-c>", lambda e: self.copy_layer())
        self.root.bind("<Control-v>", lambda e: self.paste_layer())
        self.root.bind("<Prior>", lambda e: self.prev_layer())  # Page Up
        self.root.bind("<Next>", lambda e: self.next_layer())   # Page Down
        self.root.bind("<F5>", lambda e: self.show_3d_preview())

    # --- Logic ---

    def update_listbox_selection(self):
        self.tpms_listbox.selection_clear(0, tk.END)
        for i, name in enumerate(AVAILABLE_TPMS):
            if name in self.selected_tpms_names:
                self.tpms_listbox.selection_set(i)

    def update_tpms_selection(self):
        indices = self.tpms_listbox.curselection()
        new_selection = [AVAILABLE_TPMS[i] for i in indices]
        
        if not new_selection:
            messagebox.showwarning("Warning", "Please select at least one TPMS type.")
            return
            
        # If selection changed, we need to update weights array
        if new_selection != self.selected_tpms_names:
            old_names = self.selected_tpms_names
            self.selected_tpms_names = new_selection
            
            # Create new weights array
            new_weights = np.zeros((*self.grid_size, len(new_selection)), dtype=np.float32)
            
            # Try to preserve existing weights if names match
            for i, name in enumerate(new_selection):
                if name in old_names:
                    old_idx = old_names.index(name)
                    new_weights[..., i] = self.weights[..., old_idx]
                else:
                    # New type, default to 0 (unless it's the only one, handled below)
                    pass
            
            # If we added a new type and it's the only one, set to 1
            if len(new_selection) == 1:
                new_weights[..., 0] = 1.0
            
            # Normalize? Maybe not strictly necessary here, but good practice
            # sum_w = np.sum(new_weights, axis=-1, keepdims=True)
            # new_weights = np.divide(new_weights, sum_w, out=np.zeros_like(new_weights), where=sum_w!=0)
            
            self.weights = new_weights
            self.save_state()
            self.update_inspector()
            self.update_grid_view()
            self.update_legend()

    def resize_grid(self):
        try:
            nx = self.var_x.get()
            ny = self.var_y.get()
            nz = self.var_z.get()
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid grid dimensions.")
            return
            
        if [nx, ny, nz] == self.grid_size:
            return
            
        # Resize arrays
        new_weights = np.zeros((nx, ny, nz, len(self.selected_tpms_names)), dtype=np.float32)
        new_density = np.full((nx, ny, nz), 0.3, dtype=np.float32)
        new_rotation = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        
        # Copy old data where it fits
        ox, oy, oz = self.grid_size
        min_x, min_y, min_z = min(nx, ox), min(ny, oy), min(nz, oz)
        
        new_weights[:min_x, :min_y, :min_z, :] = self.weights[:min_x, :min_y, :min_z, :]
        new_density[:min_x, :min_y, :min_z] = self.density[:min_x, :min_y, :min_z]
        new_rotation[:min_x, :min_y, :min_z, :] = self.rotation[:min_x, :min_y, :min_z, :]
        
        # If expanding, fill new areas with default? 
        if len(self.selected_tpms_names) > 0:
             # Ensure at least some weight if completely new
             pass 
             
        self.grid_size = [nx, ny, nz]
        self.weights = new_weights
        self.density = new_density
        self.rotation = new_rotation
        
        self.scale_z.config(to=nz-1)
        if self.current_z >= nz:
            self.current_z = nz - 1
            self.scale_z.set(self.current_z)
            
        self.selected_cells.clear()
        self.save_state()
        self.update_grid_view()
        self.update_inspector()
        self.update_status()

    def change_mode(self):
        self.mode = self.var_mode.get()
        self.update_grid_view()
        self.update_inspector()

    def on_z_change(self, val):
        self.current_z = int(val)
        self.update_grid_view()
        # Clear selection when changing layers? Or keep it?
        # Let's keep it but it refers to x,y on the NEW layer.
        self.update_inspector()

    def on_canvas_resize(self, event):
        self.update_grid_view()

    def get_cell_rect(self, x, y, width, height):
        nx, ny = self.grid_size[0], self.grid_size[1]
        cell_w = width / nx
        cell_h = height / ny
        return x * cell_w, y * cell_h, (x + 1) * cell_w, (y + 1) * cell_h

    def get_cell_at_coords(self, screen_x, screen_y):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        nx, ny = self.grid_size[0], self.grid_size[1]
        
        x = int(screen_x / (w / nx))
        y = int(screen_y / (h / ny))
        
        if 0 <= x < nx and 0 <= y < ny:
            return x, y
        return None

    def on_canvas_click(self, event):
        cell = self.get_cell_at_coords(event.x, event.y)
        if cell:
            self.selected_cells = {cell}
            self.update_grid_view()
            self.update_inspector()
            self.update_status()

    def on_canvas_ctrl_click(self, event):
        cell = self.get_cell_at_coords(event.x, event.y)
        if cell:
            if cell in self.selected_cells:
                self.selected_cells.remove(cell)
            else:
                self.selected_cells.add(cell)
            self.update_grid_view()
            self.update_inspector()
            self.update_status()

    def on_canvas_drag(self, event):
        # Simple drag selection (adds to selection)
        cell = self.get_cell_at_coords(event.x, event.y)
        if cell and cell not in self.selected_cells:
            self.selected_cells.add(cell)
            self.update_grid_view()
            self.update_inspector()
            self.update_status()
    
    def on_canvas_right_click(self, event):
        """右键菜单"""
        cell = self.get_cell_at_coords(event.x, event.y)
        if cell and cell not in self.selected_cells:
            self.selected_cells = {cell}
            self.update_grid_view()
            self.update_inspector()
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def update_grid_view(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1: return
        
        nx, ny = self.grid_size[0], self.grid_size[1]
        
        for x in range(nx):
            for y in range(ny):
                x1, y1, x2, y2 = self.get_cell_rect(x, y, w, h)
                
                # Determine color
                if self.mode == "weights":
                    # Mix colors
                    cell_weights = self.weights[x, y, self.current_z]
                    # Normalize for display
                    total = np.sum(cell_weights)
                    if total > 0:
                        norm_w = cell_weights / total
                    else:
                        norm_w = cell_weights # Should be zeros
                    
                    r, g, b = 0, 0, 0
                    for i, weight in enumerate(norm_w):
                        tpms_name = self.selected_tpms_names[i]
                        # Get color from TPMS_COLORS (use light color)
                        hex_color = TPMS_COLORS.get(tpms_name, ("#888888",))[0]
                        # Convert hex to rgb
                        cr = int(hex_color[1:3], 16)
                        cg = int(hex_color[3:5], 16)
                        cb = int(hex_color[5:7], 16)
                        
                        r += cr * weight
                        g += cg * weight
                        b += cb * weight
                    
                    color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                
                elif self.mode == "rotation":
                    # Map Rotation (0-360) to RGB
                    rot = self.rotation[x, y, self.current_z]
                    # Normalize: display absolute rotation modulo 360
                    # R=X, G=Y, B=Z
                    rr = int(((rot[0] % 360) / 360.0) * 255)
                    gg = int(((rot[1] % 360) / 360.0) * 255)
                    bb = int(((rot[2] % 360) / 360.0) * 255)
                    color = f"#{rr:02x}{gg:02x}{bb:02x}"

                else:  # Density
                    d = self.density[x, y, self.current_z]
                    # 使用蓝色渐变：密度越大颜色越深
                    # 低密度: 浅蓝色 (#E3F2FD), 高密度: 深蓝色 (#1565C0)
                    d_clamped = max(0.0, min(1.0, d))
                    # 插值计算 RGB
                    r = int(227 - d_clamped * (227 - 21))   # 227 -> 21
                    g = int(242 - d_clamped * (242 - 101))  # 242 -> 101
                    b = int(253 - d_clamped * (253 - 192))  # 253 -> 192
                    color = f"#{r:02x}{g:02x}{b:02x}"
                
                # Draw rect
                tag = f"cell_{x}_{y}"
                
                # Highlight if selected
                width_outline = 1
                outline = "gray"
                if (x, y) in self.selected_cells:
                    width_outline = 3
                    outline = "red"
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=outline, width=width_outline, tags=tag)
                
                # Text info?
                if nx < 10 and ny < 10:
                    if self.mode == "density":
                        d = self.density[x, y, self.current_z]
                        # 根据密度选择文字颜色：密度高时用白色，低时用黑色
                        text_color = "white" if d > 0.5 else "black"
                        self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f"{d:.2f}", fill=text_color)

    def update_inspector(self):
        # Clear previous
        for widget in self.inspector_content.winfo_children():
            widget.destroy()
            
        if not self.selected_cells:
            ttk.Label(self.inspector_content, text="No cells selected").pack()
            return
            
        ttk.Label(self.inspector_content, text=f"Selected: {len(self.selected_cells)} cells").pack(pady=5)
        ttk.Label(self.inspector_content, text=f"Layer Z: {self.current_z}").pack(pady=2)
        
        if self.mode == "weights":
            self.create_weight_sliders()
        elif self.mode == "rotation":
            self.create_rotation_sliders()
        else:
            self.create_density_slider()

    def create_rotation_sliders(self):
        # Average rotation
        avg_rot = np.zeros(3)
        count = len(self.selected_cells)
        if count > 0:
            for x, y in self.selected_cells:
                avg_rot += self.rotation[x, y, self.current_z]
            avg_rot /= count
        
        labels = ["Rot X", "Rot Y", "Rot Z"]
        
        for i in range(3):
            frame = ttk.Frame(self.inspector_content)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=labels[i], width=6).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=avg_rot[i])
            
            scale = ttk.Scale(frame, from_=0.0, to=360.0, variable=var, orient=tk.HORIZONTAL)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            val_lbl = ttk.Label(frame, text=f"{avg_rot[i]:.0f}°", width=5)
            val_lbl.pack(side=tk.LEFT)
            
            def on_slide(v, idx=i, label=val_lbl):
                val = float(v)
                label.config(text=f"{val:.0f}°")
                for x, y in self.selected_cells:
                    self.rotation[x, y, self.current_z, idx] = val
                self.update_grid_view()
            
            scale.config(command=on_slide)
            scale.bind("<ButtonRelease-1>", lambda e: self.save_state())

    def create_weight_sliders(self):
        # We need to show sliders for each TPMS type.
        # If multiple cells selected, show average? Or just set all?
        # We will set all to the slider value.
        
        # Get average values for initialization
        avg_weights = np.zeros(len(self.selected_tpms_names))
        for x, y in self.selected_cells:
            avg_weights += self.weights[x, y, self.current_z]
        avg_weights /= len(self.selected_cells)
        
        self.weight_vars = []
        
        for i, name in enumerate(self.selected_tpms_names):
            frame = ttk.Frame(self.inspector_content)
            frame.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(frame, text=name, width=10)
            lbl.pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=avg_weights[i])
            self.weight_vars.append(var)
            
            scale = ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL,
                              command=lambda v, idx=i: self.on_weight_slide(idx, v))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Value label
            val_lbl = ttk.Label(frame, text=f"{avg_weights[i]:.2f}", width=4)
            val_lbl.pack(side=tk.LEFT)
            
            # Update label on slide
            def update_lbl(v, l=val_lbl):
                l.config(text=f"{float(v):.2f}")
            scale.config(command=lambda v, idx=i, l=val_lbl: [self.on_weight_slide(idx, v), update_lbl(v)])

        ttk.Button(self.inspector_content, text="Normalize Weights", command=self.normalize_selected_weights).pack(pady=10)

    def on_weight_slide(self, idx, value):
        val = float(value)
        for x, y in self.selected_cells:
            self.weights[x, y, self.current_z, idx] = val
        self.update_grid_view()
    
    def on_weight_slide_release(self, event=None):
        """滑块释放时保存状态"""
        self.save_state()

    def normalize_selected_weights(self):
        for x, y in self.selected_cells:
            w = self.weights[x, y, self.current_z]
            total = np.sum(w)
            if total > 0:
                self.weights[x, y, self.current_z] = w / total
        self.save_state()
        self.update_inspector()  # Refresh sliders
        self.update_grid_view()

    def create_density_slider(self):
        # Average density
        avg_d = 0
        for x, y in self.selected_cells:
            avg_d += self.density[x, y, self.current_z]
        avg_d /= len(self.selected_cells)
        
        frame = ttk.Frame(self.inspector_content)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text="Density").pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=avg_d)
        scale = ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        val_lbl = ttk.Label(frame, text=f"{avg_d:.2f}", width=4)
        val_lbl.pack(side=tk.LEFT)
        
        def on_slide(v):
            val = float(v)
            val_lbl.config(text=f"{val:.2f}")
            for x, y in self.selected_cells:
                self.density[x, y, self.current_z] = val
            self.update_grid_view()
            
        scale.config(command=on_slide)

    def save_config(self):
        filename = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy Archive", "*.npz")])
        if filename:
            np.savez(filename, 
                     weights=self.weights, 
                     density=self.density, 
                     rotation=self.rotation,
                     tpms_names=self.selected_tpms_names,
                     grid_size=self.grid_size)
            messagebox.showinfo("Saved", "Configuration saved successfully.")

    def load_config(self):
        filename = filedialog.askopenfilename(filetypes=[("NumPy Archive", "*.npz")])
        if filename:
            try:
                data = np.load(filename)
                self.weights = data['weights']
                self.density = data['density']
                if 'rotation' in data:
                    self.rotation = data['rotation']
                else:
                    self.rotation = np.zeros(list(data['grid_size']) + [3], dtype=np.float32)

                self.selected_tpms_names = list(data['tpms_names'])
                self.grid_size = list(data['grid_size'])
                
                # Update UI vars
                self.var_x.set(self.grid_size[0])
                self.var_y.set(self.grid_size[1])
                self.var_z.set(self.grid_size[2])
                self.scale_z.config(to=self.grid_size[2]-1)
                
                self.update_listbox_selection()
                self.update_grid_view()
                self.update_inspector()
                messagebox.showinfo("Loaded", "Configuration loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def show_3d_preview(self):
        try:
            tpms_names = self.selected_tpms_names
            # Get colors using the helper from backend
            tpms_colors_light, tpms_colors_dark = backend.get_tpms_color_pairs(tpms_names)
            
            # Normalize weights for visualization
            w_grid = self.weights.copy()
            sum_w = np.sum(w_grid, axis=-1, keepdims=True)
            w_grid = np.divide(w_grid, sum_w, out=np.zeros_like(w_grid), where=sum_w!=0)
            
            # Decide which visualization function to use
            # Prefer tpms_hybrid_expand for arbitrary size support
            viz_func = None
            kw_density = 'density_grid'
            
            try:
                import tpms_hybrid_expand
                viz_func = tpms_hybrid_expand.visualize_weight_grid
            except ImportError:
                # Fallback to backend (likely v2)
                if hasattr(backend, 'visualize_weight_cube_3d'):
                    viz_func = backend.visualize_weight_cube_3d
                    kw_density = 'density_grid_3x3x3'
                elif hasattr(backend, 'visualize_weight_grid'):
                    viz_func = backend.visualize_weight_grid
            
            if not viz_func:
                messagebox.showwarning("Warning", "Visualization module not available.")
                return

            kwargs = {
                'title': f"Preview ({self.grid_size[0]}x{self.grid_size[1]}x{self.grid_size[2]})",
                'base_colors_light': tpms_colors_light,
                'base_colors_dark': tpms_colors_dark,
                'density_darken': True,
                kw_density: self.density
            }
            
            viz_func(w_grid, tpms_names, **kwargs)
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed: {e}")

    def export_python_code(self):
        # Generate string representation
        code = "import numpy as np\n\n"
        
        # Weights
        code += f"# Shape: {self.weights.shape}\n"
        code += "MANUAL_WEIGHT_GRID = np.array("
        code += np.array2string(self.weights, separator=', ', threshold=sys.maxsize)
        code += ", dtype=np.float32)\n\n"
        
        # Density
        code += f"# Shape: {self.density.shape}\n"
        code += "MANUAL_DENSITY_GRID = np.array("
        code += np.array2string(self.density, separator=', ', threshold=sys.maxsize)
        code += ", dtype=np.float32)\n"
        
        # Rotation
        code += f"\n# Shape: {self.rotation.shape}\n"
        code += "MANUAL_ROTATION_GRID = np.array("
        code += np.array2string(self.rotation, separator=', ', threshold=sys.maxsize)
        code += ", dtype=np.float32)\n"
        
        # Show in a new window with text area
        top = tk.Toplevel(self.root)
        top.title("Python Code Export")
        top.geometry("600x500")
        
        text_area = tk.Text(top, wrap=tk.NONE)
        text_area.pack(fill=tk.BOTH, expand=True)
        
        text_area.insert(tk.END, code)
        
        # Add scrollbars
        scroll_y = ttk.Scrollbar(text_area, orient=tk.VERTICAL, command=text_area.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        text_area.config(yscrollcommand=scroll_y.set)
        
        scroll_x = ttk.Scrollbar(text_area, orient=tk.HORIZONTAL, command=text_area.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        text_area.config(xscrollcommand=scroll_x.set)

    def load_from_code_module(self):
        try:
            # Try to use backend module
            import importlib
            module = backend
            if hasattr(module, '__file__'): # It is a module
                 importlib.reload(module)
            
            # Try 5x5x6 first (expand version feature) then standard (v2 feature)
            w_grid = getattr(module, 'MANUAL_WEIGHT_GRID_5x5x6', None)
            if w_grid is None:
                w_grid = getattr(module, 'MANUAL_WEIGHT_GRID', None)
            
            # If still None, and backend was not v2/expand directly (maybe wrapper), try explicit
            if w_grid is None:
                 try:
                     import tpms_hybrid_v2
                     importlib.reload(tpms_hybrid_v2)
                     w_grid = getattr(tpms_hybrid_v2, 'MANUAL_WEIGHT_GRID', None)
                     module = tpms_hybrid_v2
                 except ImportError:
                     pass

            if w_grid is not None:
                d_grid = getattr(module, 'MANUAL_DENSITY_GRID_5x5x6', None)
                if d_grid is None:
                    d_grid = getattr(module, 'MANUAL_DENSITY_GRID', None)
                
                r_grid = getattr(module, 'MANUAL_ROTATION_GRID', None)
                
                # Check dimensions
                if w_grid.ndim != 4:
                    raise ValueError("Weight grid must be 4D")
                    
                # Update Grid Size
                nx, ny, nz, n_channels = w_grid.shape
                self.grid_size = [nx, ny, nz]
                self.var_x.set(nx)
                self.var_y.set(ny)
                self.var_z.set(nz)
                self.scale_z.config(to=nz-1)
                
                # Update Weights
                # Note: This assumes the channel count matches current selection or we need to warn
                if n_channels != len(self.selected_tpms_names):
                    messagebox.showwarning("Warning", 
                        f"Loaded grid has {n_channels} channels, but {len(self.selected_tpms_names)} TPMS types selected.\n"
                        "Please adjust TPMS selection to match.")
                
                # Resize/Reset arrays first
                self.weights = np.zeros((nx, ny, nz, n_channels), dtype=np.float32)
                self.density = np.zeros((nx, ny, nz), dtype=np.float32)
                self.rotation = np.zeros((nx, ny, nz, 3), dtype=np.float32)
                
                self.weights = w_grid.astype(np.float32)
                
                # Update Density
                if d_grid is not None and d_grid.shape == (nx, ny, nz):
                    self.density = d_grid.astype(np.float32)
                else:
                    self.density[:] = 0.3
                
                # Update Rotation
                if r_grid is not None and r_grid.shape == (nx, ny, nz, 3):
                    self.rotation = r_grid.astype(np.float32)
                
                self.selected_cells.clear()
                self.update_grid_view()
                self.update_inspector()
                messagebox.showinfo("Success", f"Loaded grid from {module.__name__}.")
            else:
                messagebox.showerror("Error", "MANUAL_WEIGHT_GRID not found in backend module.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load from code: {e}")

    def generate_stl(self):
        if self.is_generating:
            messagebox.showwarning("Warning", "Generation already in progress.")
            return
        # Run in thread
        threading.Thread(target=self._generate_process, daemon=True).start()

    def _generate_process(self):
        self.is_generating = True
        self.root.after(0, self._show_progress)
        
        try:
            self.root.after(0, lambda: self.update_status("Generating STL..."))
            
            # Prepare data
            tpms_funcs = [tpms_hybrid.TPMS_FUNCTIONS[name] for name in self.selected_tpms_names]
            resolution = self.var_res.get()
            
            # Replicate settings
            replicate = (self.var_rep_x.get(), self.var_rep_y.get(), self.var_rep_z.get())
            
            # Desired Size
            desired_size = self.var_size.get()
            if desired_size <= 0: desired_size = None

            # Base Ranges
            base_x = (-1.5, 1.5)
            base_y = (-1.5, 1.5)
            base_z = (-1.5, 1.5)
            
            # Expand ranges if needed
            if replicate != (1, 1, 1):
                x_range, y_range, z_range = tpms_hybrid_expand.expanded_ranges(base_x, base_y, base_z, replicate)
                
                # Adjust resolution to maintain voxel density
                total_res_x = resolution * replicate[0] // 3
                total_res_y = resolution * replicate[1] // 3
                total_res_z = resolution * replicate[2] // 3
                resolution = max(total_res_x, total_res_y, total_res_z)
            else:
                x_range, y_range, z_range = base_x, base_y, base_z
            
            # Generate Fields
            self.root.after(0, lambda: self.update_status("Generating fields..."))
            
            # Helper to generate field from arbitrary grid
            def generate_field_from_grid(grid, res_shape, order=1, smooth_sigma=None, normalize=False):
                # grid shape: (Nx, Ny, Nz, [Channels])
                # res_shape: (Rx, Ry, Rz)
                # Compute zoom factors
                factors = [res_shape[i] / grid.shape[i] for i in range(3)]
                if grid.ndim > 3:
                     factors.append(1) # Don't zoom channels
                
                # Zoom
                field = zoom(grid.astype(np.float32), factors, order=order)
                
                # Smooth
                if smooth_sigma:
                    if field.ndim == 3:
                        field = gaussian_filter(field, sigma=smooth_sigma)
                    else:
                        for i in range(field.shape[-1]):
                            field[..., i] = gaussian_filter(field[..., i], sigma=smooth_sigma)
                            
                # Normalize
                if normalize and field.ndim > 3 and field.shape[-1] > 1:
                    f_sum = field.sum(axis=-1, keepdims=True)
                    field = np.divide(field, f_sum, out=np.zeros_like(field), where=f_sum!=0)
                    
                return np.clip(field, 0.0, 1.0) if normalize else field
            
            # 1. Weights
            w_grid = self.weights.copy()
            # Pre-normalize grid? Not strictly needed if we normalize field, but good practice
            sum_w = np.sum(w_grid, axis=-1, keepdims=True)
            w_grid = np.divide(w_grid, sum_w, out=np.zeros_like(w_grid), where=sum_w!=0)
            
            weight_volume = generate_field_from_grid(w_grid, (resolution, resolution, resolution), 
                                                    smooth_sigma=self.var_smooth_sigma.get(), normalize=True)
            
            # 2. Density
            density_field = generate_field_from_grid(self.density, (resolution, resolution, resolution),
                                                    smooth_sigma=None) # Density usually linear interp without extra smooth?
            
            # 3. Rotation
            # Rotation needs to be scaled? No, interpolation of degrees is fine for small changes, 
            # but wrapping around 360 is an issue. For now assume linear interpolation is okay.
            rotation_field = generate_field_from_grid(self.rotation, (resolution, resolution, resolution),
                                                     smooth_sigma=None)

            self.root.after(0, lambda: self.update_status("Creating hybrid TPMS solid..."))
            
            # Call backend
            # Note: create_hybrid_tpms_solid in v2 returns (mesh, porosity, threshold, mask) if return_mask=True
            # If backend is older, it might raise error on unknown kwargs.
            # But we imported v2 specifically or checked for rotation support.
            
            solid_mesh, actual_porosity, _, solid_mask = backend.create_hybrid_tpms_solid(
                tpms_funcs, x_range, y_range, z_range,
                weight_volume=weight_volume,
                density_field=density_field,
                rotation_field=rotation_field,  # Pass rotation
                resolution=resolution,
                solid_threshold=self.var_threshold.get(),
                smooth=True,
                return_mask=True
            )
            
            self.root.after(0, lambda: self.update_status("Finalizing solid mesh..."))
            solid_mesh = backend.finalize_mesh(solid_mesh, smooth_taubin_iter=10, do_clean=True)
            
            # Handle Fluid Domain
            fluid_mesh = None
            if self.var_gen_fluid.get():
                self.root.after(0, lambda: self.update_status("Generating fluid domain..."))
                fluid_mesh, _ = backend.create_fluid_domain_from_solid_mask(
                    solid_mask,
                    x_range, y_range, z_range,
                    resolution=resolution,
                    z_extension=self.var_z_ext.get(),
                    verbose=True
                )
                self.root.after(0, lambda: self.update_status("Finalizing fluid mesh..."))
                fluid_mesh = backend.finalize_mesh(fluid_mesh, smooth_taubin_iter=10, do_clean=True)

            # Scale and Save
            # ... (Scaling logic same as before, apply to both) ...
            
            # Desired Size scaling
            if desired_size is not None and solid_mesh is not None and solid_mesh.n_points > 0:
                b = solid_mesh.bounds
                # Use max dimension or Z? Original code used Lz.
                # v2 code uses max(lx, ly, lz). Let's stick to max dimension for safety.
                dims = [b[1]-b[0], b[3]-b[2], b[5]-b[4]]
                max_dim = max(dims)
                if max_dim > 0:
                    scale_factor = desired_size / max_dim
                    solid_mesh.scale([scale_factor]*3, inplace=True)
                    if fluid_mesh:
                         fluid_mesh.scale([scale_factor]*3, inplace=True)
                    
                    # Align origin
                    nb = solid_mesh.bounds
                    offset = [-nb[0], -nb[2], -nb[4]]
                    solid_mesh.translate(offset, inplace=True)
                    if fluid_mesh:
                        fluid_mesh.translate(offset, inplace=True)

            # Save
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            suffix = f"_{self.grid_size[0]}x{self.grid_size[1]}x{self.grid_size[2]}"
            if replicate != (1,1,1):
                suffix += f"_rep{replicate[0]}x{replicate[1]}x{replicate[2]}"
                
            filename_solid = os.path.join(output_dir, f"ui_generated_solid{suffix}.stl")
            backend.export_stl(solid_mesh, filename_solid)
            
            msg = f"Solid STL generated at:\n{filename_solid}"
            
            if fluid_mesh:
                filename_fluid = os.path.join(output_dir, f"ui_generated_fluid{suffix}.stl")
                backend.export_stl(fluid_mesh, filename_fluid)
                msg += f"\n\nFluid STL generated at:\n{filename_fluid}"
            
            self.root.after(0, lambda: messagebox.showinfo("Success", msg))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda err=str(e): messagebox.showerror("Error", f"Generation failed: {err}"))
        finally:
            self.is_generating = False
            self.root.after(0, self._hide_progress)
            self.root.after(0, lambda: self.root.config(cursor=""))
            self.root.after(0, self.update_status)
    
    def _show_progress(self):
        """显示进度条"""
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        self.progress_bar.start(10)
        self.btn_generate.config(state=tk.DISABLED)
    
    def _hide_progress(self):
        """隐藏进度条"""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.btn_generate.config(state=tk.NORMAL)
    
    # === Undo/Redo 功能 ===
    
    def save_state(self):
        """保存当前状态到历史"""
        state = {
            'weights': self.weights.copy(),
            'density': self.density.copy(),
            'rotation': self.rotation.copy(),
        }
        # 删除当前位置之后的历史
        self.history = self.history[:self.history_index + 1]
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
    
    def undo(self):
        """撤销"""
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.weights = state['weights'].copy()
            self.density = state['density'].copy()
            self.rotation = state.get('rotation', np.zeros((*self.grid_size, 3))).copy()
            self.update_grid_view()
            self.update_inspector()
            self.update_status("Undo")
    
    def redo(self):
        """重做"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.weights = state['weights'].copy()
            self.density = state['density'].copy()
            self.rotation = state.get('rotation', np.zeros((*self.grid_size, 3))).copy()
            self.update_grid_view()
            self.update_inspector()
            self.update_status("Redo")
    
    # === 选择功能 ===
    
    def select_all(self):
        """选择当前层所有单元格"""
        self.selected_cells = {(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])}
        self.update_grid_view()
        self.update_inspector()
        self.update_status()
    
    def clear_selection(self):
        """清除选择"""
        self.selected_cells.clear()
        self.update_grid_view()
        self.update_inspector()
        self.update_status()
    
    # === 层操作 ===
    
    def prev_layer(self):
        """上一层"""
        if self.current_z > 0:
            self.current_z -= 1
            self.scale_z.set(self.current_z)
            self.update_grid_view()
            self.update_inspector()
            self.update_status()
    
    def next_layer(self):
        """下一层"""
        if self.current_z < self.grid_size[2] - 1:
            self.current_z += 1
            self.scale_z.set(self.current_z)
            self.update_grid_view()
            self.update_inspector()
            self.update_status()
    
    def copy_layer(self):
        """复制当前层"""
        self.copied_weights = self.weights[:, :, self.current_z, :].copy()
        self.copied_density = self.density[:, :, self.current_z].copy()
        self.copied_rotation = self.rotation[:, :, self.current_z, :].copy()
        self.update_status(f"Layer {self.current_z + 1} copied")
    
    def paste_layer(self):
        """粘贴到当前层"""
        if hasattr(self, 'copied_weights') and hasattr(self, 'copied_density') and hasattr(self, 'copied_rotation'):
            # 检查尺寸是否匹配
            if self.copied_weights.shape[:2] == (self.grid_size[0], self.grid_size[1]):
                self.weights[:, :, self.current_z, :] = self.copied_weights.copy()
                self.density[:, :, self.current_z] = self.copied_density.copy()
                self.rotation[:, :, self.current_z, :] = self.copied_rotation.copy()
                self.save_state()
                self.update_grid_view()
                self.update_inspector()
                self.update_status(f"Pasted to layer {self.current_z + 1}")
            else:
                messagebox.showwarning("Warning", "Layer size mismatch. Cannot paste.")
        else:
            messagebox.showinfo("Info", "No layer copied. Use Ctrl+C first.")
    
    def copy_to_all_layers(self):
        """将选中单元格的值复制到所有层"""
        if not self.selected_cells:
            return
        for x, y in self.selected_cells:
            for z in range(self.grid_size[2]):
                self.weights[x, y, z, :] = self.weights[x, y, self.current_z, :]
                self.density[x, y, z] = self.density[x, y, self.current_z]
                self.rotation[x, y, z, :] = self.rotation[x, y, self.current_z, :]
        self.save_state()
        self.update_status("Copied to all layers")
    
    # === 填充功能 ===
    
    def fill_selected_default(self):
        """将选中单元格填充为默认值"""
        if not self.selected_cells:
            return
        for x, y in self.selected_cells:
            self.weights[x, y, self.current_z, :] = 0
            if len(self.selected_tpms_names) > 0:
                self.weights[x, y, self.current_z, 0] = 1.0
            self.density[x, y, self.current_z] = 0.3
            self.rotation[x, y, self.current_z, :] = 0.0
        self.save_state()
        self.update_grid_view()
        self.update_inspector()
    
    def fill_all_cells(self):
        """用当前选中单元格的值填充所有单元格"""
        if not self.selected_cells:
            messagebox.showinfo("Info", "Please select a cell first.")
            return
        # 使用第一个选中单元格的值
        x, y = next(iter(self.selected_cells))
        src_weights = self.weights[x, y, self.current_z, :].copy()
        src_density = self.density[x, y, self.current_z]
        src_rotation = self.rotation[x, y, self.current_z, :].copy()
        
        for xi in range(self.grid_size[0]):
            for yi in range(self.grid_size[1]):
                self.weights[xi, yi, self.current_z, :] = src_weights
                self.density[xi, yi, self.current_z] = src_density
                self.rotation[xi, yi, self.current_z, :] = src_rotation
        
        self.save_state()
        self.update_grid_view()
        self.update_status("Filled all cells in current layer")
    
    # === 帮助对话框 ===
    
    def show_shortcuts_help(self):
        """显示快捷键帮助"""
        help_text = """Keyboard Shortcuts:

File:
  Ctrl+S    Save configuration
  Ctrl+O    Open configuration

Edit:
  Ctrl+Z    Undo
  Ctrl+Y    Redo
  Ctrl+A    Select all cells (current layer)
  Escape    Clear selection
  Ctrl+C    Copy current layer
  Ctrl+V    Paste to current layer

View:
  Page Up   Previous layer
  Page Down Next layer
  F5        3D Preview

Mouse:
  Left Click       Select cell
  Ctrl+Left Click  Toggle cell selection
  Drag             Multi-select cells
  Right Click      Context menu
"""
        messagebox.showinfo("Keyboard Shortcuts", help_text)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = """TPMS Hybrid Generator UI

Version: 1.0

A tool for creating hybrid TPMS (Triply Periodic Minimal Surface) structures with customizable weight and density grids.
"""
        messagebox.showinfo("About", about_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TPMSApp(root)
    root.mainloop()
