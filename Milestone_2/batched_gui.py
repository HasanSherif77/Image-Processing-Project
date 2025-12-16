"""
Enhanced Batch Puzzle Solver GUI - Modern Design
Professional interface for batch processing jigsaw puzzles
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import threading
import sys
import os
import cv2
import numpy as np
from pathlib import Path
import time
from batch_process_puzzles import process_puzzle_folder


class ModernBatchPuzzleGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Modern color scheme
        self.colors = {
            'bg_dark': '#1e1e2e',
            'bg_medium': '#2e3440',
            'bg_light': '#3b4252',
            'accent': '#88c0d0',
            'success': '#a3be8c',
            'warning': '#ebcb8b',
            'error': '#bf616a',
            'text': '#eceff4',
            'text_dim': '#d8dee9'
        }

        # Variables
        self.puzzle_folder = None
        self.output_folder = tk.StringVar(value="output/batch_solved")
        self.grid_size = tk.IntVar(value=2)
        self.beam_width = tk.IntVar(value=80)
        self.downsample_size = tk.IntVar(value=48)

        self.is_processing = False
        self.preview_images = []
        self.preview_index = 0
        self.current_stats = {'total': 0, 'success': 0, 'failed': 0, 'costs': []}

        # Style configuration
        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure button style
        style.configure('Modern.TButton',
                        background=self.colors['accent'],
                        foreground='white',
                        borderwidth=0,
                        focuscolor='none',
                        padding=10)

        style.map('Modern.TButton',
                  background=[('active', '#5e81ac')])

        # Configure label style
        style.configure('Title.TLabel',
                        background=self.colors['bg_dark'],
                        foreground=self.colors['text'],
                        font=('Arial', 14, 'bold'))

        style.configure('Subtitle.TLabel',
                        background=self.colors['bg_medium'],
                        foreground=self.colors['text_dim'],
                        font=('Arial', 10))

        # Configure notebook style
        style.configure('TNotebook',
                        background=self.colors['bg_medium'],
                        borderwidth=0)

        style.configure('TNotebook.Tab',
                        background=self.colors['bg_light'],
                        foreground=self.colors['text'],
                        padding=[20, 10],
                        borderwidth=0)

        style.map('TNotebook.Tab',
                  background=[('selected', self.colors['accent'])],
                  foreground=[('selected', 'white')])

    def create_widgets(self):
        """Create all GUI widgets"""


        # Main container
        main_container = tk.Frame(self, bg=self.colors['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=20, pady=10)

        # Control panel
        self.create_control_panel(main_container)

        # Content area with preview and logs
        self.create_content_area(main_container)

        # Statistics panel
        self.create_stats_panel(main_container)

        # Status bar
        self.create_status_bar()


    def create_control_panel(self, parent):
        """Create enhanced control panel"""
        panel = tk.Frame(parent, bg=self.colors['bg_medium'], relief='flat', bd=2)
        panel.pack(fill='x', pady=(0, 15))

        # Inner padding
        inner = tk.Frame(panel, bg=self.colors['bg_medium'])
        inner.pack(fill='x', padx=10, pady=10)

        # Folder selection section
        folder_frame = tk.LabelFrame(inner,
                                     text="  📁 Input Configuration  ",
                                     bg=self.colors['bg_medium'],
                                     fg=self.colors['accent'],
                                     font=('Arial', 11, 'bold'),
                                     relief='flat')
        folder_frame.pack(fill='x', pady=(0, 5))

        folder_inner = tk.Frame(folder_frame, bg=self.colors['bg_medium'])
        folder_inner.pack(fill='x', padx=5, pady=5)

        # Puzzle folder
        tk.Label(folder_inner,
                 text="Puzzle Folder:",
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text'],
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)

        self.folder_label = tk.Label(folder_inner,
                                     text="No folder selected",
                                     bg=self.colors['bg_light'],
                                     fg=self.colors['text_dim'],
                                     anchor='w',
                                     relief='flat',
                                     padx=10,
                                     pady=8,
                                     width=50)
        self.folder_label.grid(row=0, column=1, sticky='ew', padx=5)

        self.browse_btn = tk.Button(folder_inner,
                                    text="📂 Browse",
                                    command=self.select_folder,
                                    bg=self.colors['accent'],
                                    fg='white',
                                    font=('Arial', 10, 'bold'),
                                    relief='flat',
                                    padx=20,
                                    pady=8,
                                    cursor='hand2')
        self.browse_btn.grid(row=0, column=2, padx=5)

        # Output folder
        tk.Label(folder_inner,
                 text="Output Folder:",
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text'],
                 font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=5)

        output_entry = tk.Entry(folder_inner,
                                textvariable=self.output_folder,
                                bg=self.colors['bg_light'],
                                fg=self.colors['text'],
                                relief='flat',
                                font=('Arial', 10),
                                insertbackground=self.colors['text'])
        output_entry.grid(row=1, column=1, sticky='ew', padx=5, ipady=5)

        tk.Button(folder_inner,
                  text="📂 Browse",
                  command=self.select_output_folder,
                  bg=self.colors['accent'],
                  fg='white',
                  font=('Arial', 10, 'bold'),
                  relief='flat',
                  padx=20,
                  pady=8,
                  cursor='hand2').grid(row=1, column=2, padx=5)

        folder_inner.grid_columnconfigure(1, weight=1)

        # Settings section
        settings_frame = tk.LabelFrame(inner,
                                       text="  ⚙️ Solver Settings  ",
                                       bg=self.colors['bg_medium'],
                                       fg=self.colors['accent'],
                                       font=('Arial', 11, 'bold'),
                                       relief='flat')
        settings_frame.pack(fill='x', pady=(0, 5))

        settings_inner = tk.Frame(settings_frame, bg=self.colors['bg_medium'])
        settings_inner.pack(fill='x', padx=5, pady=5)

        # Grid size
        self.create_setting_control(settings_inner, 0,
                                    "Grid Size:",
                                    "Puzzle dimension (2x2, 4x4, or 8x8)",
                                    'combobox', [2, 4, 8],
                                    self.grid_size)

        # Beam width
        self.create_setting_control(settings_inner, 1,
                                    "Beam Width:",
                                    "Higher = better quality, slower (recommended: 80-150)",
                                    'spinbox', (10, 200),
                                    self.beam_width)

        # Downsample size
        self.create_setting_control(settings_inner, 2,
                                    "Downsample Size:",
                                    "Edge matching resolution (recommended: 48-64)",
                                    'spinbox', (16, 128),
                                    self.downsample_size)

        # Process button
        button_frame = tk.Frame(inner, bg=self.colors['bg_medium'])
        button_frame.pack(fill='x', pady=10)

        self.process_btn = tk.Button(button_frame,
                                     text="▶  Start Processing",
                                     command=self.start_processing,
                                     bg=self.colors['success'],
                                     fg='white',
                                     font=('Arial', 10, 'bold'),
                                     relief='flat',
                                     padx=10,
                                     pady=5,
                                     cursor='hand2',
                                     state=tk.DISABLED)
        self.process_btn.pack()

    def create_setting_control(self, parent, row, label, tooltip, control_type, values, variable):
        """Create a setting control with label and tooltip"""
        # Label
        tk.Label(parent,
                 text=label,
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text'],
                 font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='w', padx=(0, 15), pady=8)

        # Control
        if control_type == 'combobox':
            control = ttk.Combobox(parent,
                                   textvariable=variable,
                                   values=values,
                                   width=15,
                                   state='readonly')
        else:  # spinbox
            control = ttk.Spinbox(parent,
                                  from_=values[0],
                                  to=values[1],
                                  textvariable=variable,
                                  width=15)

        control.grid(row=row, column=1, sticky='w', pady=4)

        # Tooltip
        tk.Label(parent,
                 text=f"ℹ️  {tooltip}",
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text_dim'],
                 font=('Arial', 9, 'italic')).grid(row=row, column=2, sticky='w', padx=15)

    def create_content_area(self, parent):
        """Create content area with tabs"""
        content = tk.Frame(parent, bg=self.colors['bg_dark'])
        content.pack(fill='both', expand=True, pady=(0, 15))

        # Paned window for resizable sections
        paned = tk.PanedWindow(content,
                               orient=tk.HORIZONTAL,
                               bg=self.colors['bg_dark'],
                               sashwidth=5,
                               sashrelief='flat')
        paned.pack(fill='both', expand=True)

        # Left: Image list
        left_frame = tk.Frame(paned, bg=self.colors['bg_medium'], width=300)
        paned.add(left_frame, minsize=250)

        tk.Label(left_frame,
                 text="📋 Processed Images",
                 bg=self.colors['bg_medium'],
                 fg=self.colors['text'],
                 font=('Arial', 11, 'bold'),
                 pady=10).pack()

        list_frame = tk.Frame(left_frame, bg=self.colors['bg_medium'])
        list_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.image_listbox = tk.Listbox(list_frame,
                                        yscrollcommand=scrollbar.set,
                                        bg=self.colors['bg_light'],
                                        fg=self.colors['text'],
                                        selectbackground=self.colors['accent'],
                                        selectforeground='white',
                                        font=('Consolas', 9),
                                        relief='flat',
                                        highlightthickness=0)
        self.image_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.image_listbox.yview)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)

        # Right: Notebook with tabs
        right_frame = tk.Frame(paned, bg=self.colors['bg_medium'])
        paned.add(right_frame, minsize=600)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Preview tab
        self.create_preview_tab()

        # Log tab
        self.create_log_tab()


    def create_preview_tab(self):
        """Create preview tab with image viewer"""
        preview_tab = tk.Frame(self.notebook, bg=self.colors['bg_light'])
        self.notebook.add(preview_tab, text="🖼️  Preview")

        # Preview canvas
        self.preview_canvas = tk.Canvas(preview_tab,
                                        bg='#000000',
                                        highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True, padx=10, pady=10)

        # Navigation controls
        nav_frame = tk.Frame(preview_tab, bg=self.colors['bg_light'])
        nav_frame.pack(fill='x', padx=10, pady=10)

        tk.Button(nav_frame,
                  text="◀  Previous",
                  command=self.prev_image,
                  bg=self.colors['accent'],
                  fg='white',
                  font=('Arial', 10, 'bold'),
                  relief='flat',
                  padx=20,
                  pady=8,
                  cursor='hand2').pack(side='left', padx=5)

        self.image_label = tk.Label(nav_frame,
                                    text="No images",
                                    bg=self.colors['bg_light'],
                                    fg=self.colors['text'],
                                    font=('Arial', 10))
        self.image_label.pack(side='left', expand=True)

        tk.Button(nav_frame,
                  text="Next  ▶",
                  command=self.next_image,
                  bg=self.colors['accent'],
                  fg='white',
                  font=('Arial', 10, 'bold'),
                  relief='flat',
                  padx=20,
                  pady=8,
                  cursor='hand2').pack(side='right', padx=5)

    def create_log_tab(self):
        """Create log tab"""
        log_tab = tk.Frame(self.notebook, bg=self.colors['bg_light'])
        self.notebook.add(log_tab, text="📝  Processing Log")

        self.log_text = ScrolledText(log_tab,
                                     wrap=tk.WORD,
                                     bg=self.colors['bg_dark'],
                                     fg=self.colors['text'],
                                     font=('Consolas', 9),
                                     relief='flat',
                                     insertbackground=self.colors['text'])
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure text tags for colored output
        self.log_text.tag_config('success', foreground=self.colors['success'])
        self.log_text.tag_config('error', foreground=self.colors['error'])
        self.log_text.tag_config('warning', foreground=self.colors['warning'])
        self.log_text.tag_config('info', foreground=self.colors['accent'])


    def create_stats_panel(self, parent):
        """Create live statistics panel"""
        stats_frame = tk.Frame(parent, bg=self.colors['bg_medium'], relief='flat', bd=2)
        stats_frame.pack(fill='x', pady=(0, 15))

        inner = tk.Frame(stats_frame, bg=self.colors['bg_medium'])
        inner.pack(fill='x', padx=20, pady=15)

        tk.Label(inner,
                 text="📈 Live Statistics",
                 bg=self.colors['bg_medium'],
                 fg=self.colors['accent'],
                 font=('Arial', 11, 'bold')).pack(anchor='w', pady=(0, 10))

        # Stats grid
        stats_grid = tk.Frame(inner, bg=self.colors['bg_medium'])
        stats_grid.pack(fill='x')

        self.stat_widgets = {}
        stats_config = [
            ('Total', '0', self.colors['accent']),
            ('Successful', '0', self.colors['success']),
            ('Failed', '0', self.colors['error']),
            ('Avg Cost', '0.00', self.colors['warning'])
        ]

        for idx, (label, value, color) in enumerate(stats_config):
            frame = tk.Frame(stats_grid, bg=self.colors['bg_light'], relief='flat')
            frame.grid(row=0, column=idx, padx=10, sticky='ew')

            tk.Label(frame,
                     text=label,
                     bg=self.colors['bg_light'],
                     fg=self.colors['text_dim'],
                     font=('Arial', 9)).pack(pady=(10, 2))

            value_label = tk.Label(frame,
                                   text=value,
                                   bg=self.colors['bg_light'],
                                   fg=color,
                                   font=('Arial', 16, 'bold'))
            value_label.pack(pady=(0, 10))

            self.stat_widgets[label] = value_label
            stats_grid.grid_columnconfigure(idx, weight=1)

    def create_status_bar(self):
        """Create modern status bar"""
        status_frame = tk.Frame(self, bg=self.colors['bg_medium'], height=40)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame,
                                     text="Ready to process puzzles",
                                     bg=self.colors['bg_medium'],
                                     fg=self.colors['text'],
                                     font=('Arial', 9),
                                     anchor='w',
                                     padx=20)
        self.status_label.pack(side='left', fill='y')

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame,
                                            variable=self.progress_var,
                                            length=300,
                                            mode='determinate')
        self.progress_bar.pack(side='right', padx=20)

    # --------------------
    # Event Handlers
    # --------------------

    def select_folder(self):
        """Select puzzle folder"""
        folder = filedialog.askdirectory(title="Select Puzzle Folder")
        if folder:
            self.puzzle_folder = folder
            basename = os.path.basename(folder)
            self.folder_label.config(text=basename, fg=self.colors['text'])
            self.process_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Loaded folder: {basename}")

            # Count images
            image_count = len([f for f in os.listdir(folder)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            self.update_stat('Total', str(image_count))

    def select_output_folder(self):
        """Select output folder"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def start_processing(self):
        """Start batch processing"""
        if self.is_processing:
            return

        if not self.puzzle_folder:
            messagebox.showerror("Error", "Please select a puzzle folder first")
            return

        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED, text="⏸  Processing...")
        self.log_text.delete('1.0', tk.END)
        self.preview_images = []
        self.preview_index = 0
        self.progress_var.set(0)
        self.current_stats = {'total': 0, 'success': 0, 'failed': 0, 'costs': []}
        self.notebook.select(1)  # Switch to log tab

        thread = threading.Thread(target=self._run_process_thread, daemon=True)
        thread.start()

    def _run_process_thread(self):
        """Run processing in background thread"""
        try:
            # Redirect output
            sys.stdout = TextRedirector(self.log_text, self)
            sys.stderr = TextRedirector(self.log_text, self)

            process_puzzle_folder(
                folder_path=self.puzzle_folder,
                grid_size=self.grid_size.get(),
                output_base_dir=self.output_folder.get(),
                beam_width=self.beam_width.get(),
                downsample_size=self.downsample_size.get()
            )

            # Load results
            output_dir = self.output_folder.get()
            if os.path.exists(output_dir):
                self.preview_images = sorted([
                    os.path.join(output_dir, f)
                    for f in os.listdir(output_dir)
                    if f.endswith("_solved.png")
                ])

                if self.preview_images:
                    self.preview_index = 0
                    self.after(0, self.show_preview)
                    self.after(0, self.load_image_list)

            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Batch processing completed!\n\nProcessed {len(self.preview_images)} images"
            ))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.is_processing = False
            self.after(0, lambda: self.process_btn.config(
                state=tk.NORMAL,
                text="▶  Start Processing"
            ))
            self.after(0, lambda: self.progress_var.set(100))
            self.after(0, lambda: self.status_label.config(
                text="Processing complete"
            ))

    def load_image_list(self):
        """Load processed images into list"""
        self.image_listbox.delete(0, tk.END)
        for img_path in self.preview_images:
            self.image_listbox.insert(tk.END, os.path.basename(img_path))

    def on_listbox_select(self, event):
        """Handle image selection"""
        selection = event.widget.curselection()
        if selection:
            self.preview_index = selection[0]
            self.show_preview()

    def show_preview(self):
        """Show current image in preview"""
        if not self.preview_images:
            return

        try:
            img_path = self.preview_images[self.preview_index]
            img = Image.open(img_path)

            # Get canvas size
            canvas_w = self.preview_canvas.winfo_width()
            canvas_h = self.preview_canvas.winfo_height()

            if canvas_w < 10:
                canvas_w, canvas_h = 800, 600

            # Resize to fit
            img_w, img_h = img.size
            scale = min((canvas_w - 20) / img_w, (canvas_h - 20) / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)

            # Display
            self.preview_canvas.delete('all')
            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2
            self.preview_canvas.create_image(x, y, anchor='nw', image=self.tk_img)

            # Update label
            self.image_label.config(
                text=f"{self.preview_index + 1} / {len(self.preview_images)} - {os.path.basename(img_path)}"
            )

        except Exception as e:
            print(f"Error loading preview: {e}")

    def next_image(self):
        """Show next image"""
        if not self.preview_images:
            return
        self.preview_index = (self.preview_index + 1) % len(self.preview_images)
        self.show_preview()

    def prev_image(self):
        """Show previous image"""
        if not self.preview_images:
            return
        self.preview_index = (self.preview_index - 1) % len(self.preview_images)
        self.show_preview()

    def update_stat(self, stat_name, value):
        """Update a statistic widget"""
        if stat_name in self.stat_widgets:
            self.stat_widgets[stat_name].config(text=value)


class TextRedirector:
    """Redirect stdout/stderr to text widget with color coding"""

    def __init__(self, widget, gui):
        self.widget = widget
        self.gui = gui

    def write(self, text):
        # Color code based on content
        tag = None
        if '✓' in text or 'Success' in text:
            tag = 'success'
        elif '✗' in text or 'Error' in text or 'Failed' in text:
            tag = 'error'
        elif '⚠' in text or 'Warning' in text:
            tag = 'warning'
        elif 'Step' in text or 'Position' in text:
            tag = 'info'

        self.widget.insert(tk.END, text, tag)
        self.widget.see(tk.END)

        # Update stats from log
        if 'Successfully processed' in text:
            self.gui.current_stats['success'] += 1
            self.gui.after(0, lambda: self.gui.update_stat(
                'Successful',
                str(self.gui.current_stats['success'])
            ))
        elif 'Failed to process' in text:
            self.gui.current_stats['failed'] += 1
            self.gui.after(0, lambda: self.gui.update_stat(
                'Failed',
                str(self.gui.current_stats['failed'])
            ))
        elif 'Final cost:' in text:
            try:
                cost = float(text.split('Final cost:')[1].strip())
                self.gui.current_stats['costs'].append(cost)
                if self.gui.current_stats['costs']:
                    avg = np.mean(self.gui.current_stats['costs'])
                    self.gui.after(0, lambda: self.gui.update_stat(
                        'Avg Cost',
                        f"{avg:.2f}"
                    ))
            except:
                pass

    def flush(self):
        pass


def main():
    app = ModernBatchPuzzleGUI()
    app.mainloop()


if __name__ == "__main__":
    main()