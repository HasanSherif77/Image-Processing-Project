import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

import feature1
import feature2
import feature3
import feature4


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title(" Image Processor")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1b1b1d")

        self.original = None
        self.final = None
        self.feature_images = [None]*4

        self.features = [feature1.apply, feature2.apply,
                         feature3.apply, feature4.apply]
        self.feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]

        self.build_ui()

    def build_ui(self):
        # Scrollable frame setup
        main_canvas = tk.Canvas(self.root, bg="#1b1b1d")
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=main_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        main_canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame = tk.Frame(main_canvas, bg="#1b1b1d")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Title
        tk.Label(
            scrollable_frame,
            text="Image Processor",
            bg="#1b1b1d",
            fg="white",
            font=("Segoe UI", 28, "bold")
        ).pack(pady=15)

        # Load Image Button
        tk.Button(
            scrollable_frame,
            text="Load Image",
            command=self.load_image,
            bg="#190c28",
            fg="white",
            font=("Segoe UI", 13, "bold"),
            bd=0,
            height=2
        ).pack(pady=10)

        # Top frame: Before / After final
        top_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        top_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)

        tk.Label(top_frame, text="Before / After Final",
                 bg="#1b1b1d", fg="white", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.top_canvas = tk.Canvas(top_frame, bg="#0e0e0f", height=400)
        self.top_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom frame: 4 feature previews
        bottom_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        bottom_frame.pack(fill=tk.X, expand=False, padx=20, pady=10)

        tk.Label(bottom_frame, text="Feature Previews",
                 bg="#1b1b1d", fg="white", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.feature_frame = tk.Frame(bottom_frame, bg="#1b1b1d")
        self.feature_frame.pack(fill=tk.X, expand=True)

        self.feature_canvases = []
        for i in range(4):
            c = tk.Canvas(self.feature_frame, bg="#0e0e0f", width=300, height=200)
            c.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
            self.feature_canvases.append(c)

        # Buttons for applying features
        btn_frame = tk.Frame(scrollable_frame, bg="#1b1b1d")
        btn_frame.pack(pady=10)

        for i in range(4):
            tk.Button(
                btn_frame,
                text=self.feature_names[i],
                command=lambda idx=i: self.apply_feature(idx),
                bg="#3a3aff",
                fg="white",
                font=("Segoe UI", 12, "bold"),
                bd=0,
                height=2,
                width=15
            ).pack(side=tk.LEFT, padx=10)

        tk.Button(
            btn_frame,
            text="Apply All Features",
            command=self.apply_all,
            bg="#ff9900",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            bd=0,
            height=2,
            width=15
        ).pack(side=tk.LEFT, padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return
        self.original = cv2.imread(path)
        self.final = self.original.copy()
        self.feature_images = [None]*4
        self.update_top_canvas()
        self.update_feature_canvases()

    def apply_feature(self, idx):
        if self.original is None:
            return
        # Apply individual feature preview
        self.feature_images[idx] = self.features[idx](self.original.copy())
        # Update final image cumulatively
        self.final = self.features[idx](self.final.copy())
        self.update_top_canvas()
        self.update_feature_canvases()

    def apply_all(self):
        if self.original is None:
            return
        # Reset final and feature previews
        self.final = self.original.copy()
        for i in range(4):
            self.feature_images[i] = self.features[i](self.original.copy())
            self.final = self.features[i](self.final.copy())
        self.update_top_canvas()
        self.update_feature_canvases()

    def update_top_canvas(self):
        if self.original is None or self.final is None:
            return
        before = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(self.final, cv2.COLOR_BGR2RGB)
        h = min(before.shape[0], after.shape[0])
        before_resized = cv2.resize(before, (int(before.shape[1]*h/before.shape[0]), h))
        after_resized = cv2.resize(after, (int(after.shape[1]*h/after.shape[0]), h))
        combined = cv2.hconcat([before_resized, after_resized])
        self.display_image(combined, self.top_canvas)

    def update_feature_canvases(self):
        for i, c in enumerate(self.feature_canvases):
            img = self.feature_images[i]
            if img is None:
                c.delete("all")
                c.create_rectangle(0, 0, 300, 200, fill="#0e0e0f")
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.display_image(img_rgb, c)

    def display_image(self, img, canvas):
        pil = Image.fromarray(img)
        w = canvas.winfo_width() or canvas.winfo_reqwidth()
        h = canvas.winfo_height() or canvas.winfo_reqheight()
        pil.thumbnail((w, h))
        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image(w//2, h//2, image=tk_img, anchor="center")
        canvas.image = tk_img


def main():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()



