"""
Tkinter GUI for interactive X-ray image simulation.

All business logic lives in simulator.simulate.simulate().
GUI merely gathers parameters from the use and renders the
resulting Numpy array with matplotlib.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from simulator.simulate import simulate

class ParamSlider(ttk.Frame):
    """
    Helper wiget: label + Tk Scale + read-only entry showing current value.
    """

    def __init__(self, master, text, from_, to, resolution, default,
                 orient=tk.HORIZONTAL, command=None, unit=""):
        super().__init__(master)
        self.var = tk.DoubleVar(value=default)
        ttk.Label(self, text=text).pack(side=tk.LEFT, padx=(0,4))
        self.scale = ttk.Scale(self, orient=orient, from_=from_, to=to,
                               variable=self.var, command=self._on_move)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry = ttk.Entry(self, width=6, textvariable=self.var, state='readonly')
        self.entry.pack(side=tk.LEFT, padx=(4,0))
        self.unit_label = ttk.Label(self, text=unit)
        self.unit_label.pack(side=tk.LEFT, padx=(2,0))
        self.user_command = command
        # Ensure resolution
        self.resolution = resolution
    
    def _on_move(self, *_):
        # Snap to resolution
        value = round(self.var.get() / self.resolution) * self.resolution
        self.var.set(value)
        if self.user_command:
            self.user_command()

    def get(self):
        return self.var.get()

class XrayGUI(tk.Tk):
    """Main aoolication window."""

    IMG_W = 512
    IMG_H = 512

    def __init__(self):
        super().__init__()
        self.title("X-ray Image Simulator")
        self._build_controls()
        self._build_canvas()
        self._update_image()
    
    # --- UI layout ---

    def _build_controls(self):
        ctrl = ttk.Frame(self, padding=8)
        ctrl.grid(row=0, column=0, sticky="nsew")

        # kVp
        self.kvp = ParamSlider(ctrl, "kVp", 40, 120, 1, 80,
                               command=self._update_image, unit="kV")
        self.kvp.pack(fill=tk.X, pady=2)

        # mAs
        self.mas = ParamSlider(ctrl, "mAs", 1, 20, 0.1, 10,
                               command=self._update_image, unit="mAs")
        self.mas.pack(fill=tk.X, pady=2)

        # Cone scale
        self.scale = ParamSlider(ctrl, "Cone scale", 0.1, 1.0, 0.05, 0.5,
                               command=self._update_image)
        self.scale.pack(fill=tk.X, pady=2)
        
        # Offset X
        self.offset_x = ParamSlider(ctrl, "Offset X", 0, self.IMG_W, 1, 100,
                               command=self._update_image, unit="px")
        self.offset_x.pack(fill=tk.X, pady=2)
        
        # Offset Y
        self.offset_y = ParamSlider(ctrl, "Offset Y", 0, self.IMG_H, 1, 150,
                               command=self._update_image)
        self.offset_y.pack(fill=tk.X, pady=2)
        
        # Photons per pixel (quantum noise)
        self.photons = ParamSlider(ctrl, "Photons/pixel", 1e3, 1e7, 1e3, 1e6,
                               command=self._update_image)
        self.photons.pack(fill=tk.X, pady=2)
        
        # System noise sigma
        self.sigma = ParamSlider(ctrl, "System sigma", 0.0, 0.05, 0.001, 0.02,
                               command=self._update_image)
        self.sigma.pack(fill=tk.X, pady=2)
        
        # Save button
        ttk.Button(ctrl, text="Save PNG", command=self._save_png).pack(
            pady=(6, 0), fill=tk.X)
        
    def _build_canvas(self):
        fig = plt.Figure(figsize=(5,5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.axis("off")
        self.im = self.ax.imshow(np.zeros((self.IMG_H, self.IMG_W)), cmap="gray", vmin=0, vmax=1)
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=1, sticky="nsew")
        self.canvas = canvas
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    # --- Rendering ---
    def _get_params(self):
        return dict(
            kvp=self.kvp.get(),
            mas=self.mas.get(),
            scale=self.scale.get(),
            offset=(int(self.offset_x.get()), int(self.offset_y.get())),
            photons=(self.photons.get()),
            sigma=self.sigma.get(),
        )

    def _update_image(self, *_):
        p = self._get_params()
        img = simulate(p["kvp"], p["mas"],
                    height_pix=self.IMG_H, width_pix=self.IMG_W,
                    cone_scale=p["scale"], cone_offset=p["offset"],
                    photons=p["photons"], sigma=p["sigma"])
        self.im.set_data(img)
        self.canvas.draw_idle()

    def _save_png(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        
        if not path:
            return
        try:
            matplotlib.image.imsave(path, self.img.get_array(), cmap='gray')
            messagebox.showinfo("Saved", f"Image saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    # ... build_controls, build_canvas, get_params, update_image ...
if __name__ == "__main__":
    XrayGUI().mainloop()