
import os
from tkinter import filedialog as fd
from tkinter import messagebox
import tkinter as tk
from tkinter import  messagebox, ttk
from models import HuggingFaceModel1, HuggingFaceModel2

EXPLANATIONS_TEXT = """
OOP Concepts Used in This App
-----------------------------
1) Inheritance:
   - BaseHFModel defines shared behavior (lazy loading, run(), get_info()).
   - HuggingFaceModel1/2 extend BaseHFModel and supply task/model metadata.

2) Encapsulation:
   - The HF pipeline object is stored privately as _pipe and created on-demand
     via _ensure_loaded(); GUI code cannot manipulate it directly.

3) Polymorphism:
   - The GUI calls model.run(input). Different subclasses can implement or
     override run() to handle text vs image transparently.

4) Method Overriding:
   - Subclasses may override run() or get_info() to customize behavior/metadata.

5) Multiple Decorators (example):
   - See utils.py for @logger and @timed applied to functions used by the GUI.
"""

class App:
    def _format_image_results(self, preds):
        lines = [f"Top {len(preds)} predictions:"]
        for i, p in enumerate(preds, 1):
            lines.append(f"{i:>2}. {p['label']:<28} {p['score']*100:5.1f}%")
        return "\n".join(lines)

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HIT137 Assignment 3 - AI GUI")
        self.root.geometry("800x600")

        # --- Notebook with three tabs: Run, Explanations, Model Info
        self.nb = ttk.Notebook(self.root)
        self.tab_run = ttk.Frame(self.nb)
        self.tab_explain = ttk.Frame(self.nb)
        self.tab_info = ttk.Frame(self.nb)
        self.nb.add(self.tab_run, text="Run")
        self.nb.add(self.tab_explain, text="Explanations")
        self.nb.add(self.tab_info, text="Model Info")
        self.nb.pack(fill="both", expand=True)

        # ========== RUN TAB ==========
        top = ttk.Frame(self.tab_run); top.pack(padx=12, pady=12, anchor="w")

        ttk.Label(top, text="Select Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="Sentiment Analysis (Text)")
        self.model_menu = ttk.OptionMenu(
            top, self.model_var, "Sentiment Analysis (Text)",
            "Sentiment Analysis (Text)", "Image Classification",
            command=self._on_model_change
        )
        self.model_menu.grid(row=0, column=1, padx=6)

        self.input_label = ttk.Label(top, text="Enter text below (Model: Sentiment)")
        self.input_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10,2))

        self.text_entry = ttk.Entry(top, width=70)
        self.text_entry.grid(row=2, column=0, columnspan=2, sticky="we")

        self.pick_btn = ttk.Button(top, text="Choose Image…", command=self._pick_image)
        # pick_btn is hidden initially (text model)
        # we will grid/remove it when model changes

        self.run_btn = ttk.Button(top, text="Run Model", command=self._run_model)
        self.run_btn.grid(row=3, column=0, pady=10, sticky="w")

        self.out = tk.Text(self.tab_run, height=18)
        self.out.pack(fill="both", expand=True, padx=12, pady=6)

        # ========== EXPLANATIONS TAB ==========
        self.exp_text = tk.Text(self.tab_explain, wrap="word")
        self.exp_text.pack(fill="both", expand=True, padx=12, pady=12)
        self.exp_text.insert("1.0", EXPLANATIONS_TEXT)
        self.exp_text.configure(state="disabled")

        # ========== MODEL INFO TAB ==========
        self.info_tree = ttk.Treeview(self.tab_info, columns=("value",), show="tree headings")
        self.info_tree.heading("#0", text="Field")
        self.info_tree.heading("value", text="Value")
        self.info_tree.column("#0", width=220)
        self.info_tree.column("value", width=520)
        self.info_tree.pack(fill="both", expand=True, padx=12, pady=12)

        # state
        self._image_path = None
        self._models = {
            "Sentiment Analysis (Text)": HuggingFaceModel1,
            "Image Classification": HuggingFaceModel2,
        }

        # initialize Model Info for default selection
        self._refresh_model_info()

    # ------- UI helpers -------
    def _on_model_change(self, *_):
        model_name = self.model_var.get()
        if model_name == "Sentiment Analysis (Text)":
            self.input_label.config(text="Enter text below (Model: Sentiment)")
            self.text_entry.grid()           # show
            self.pick_btn.grid_forget()      # hide image picker
        else:
            self.input_label.config(text="Pick an image file (Model: Image Classification)")
            self.text_entry.grid_remove()    # hide text box
            self.pick_btn.grid(row=2, column=0, sticky="w", pady=4)
        self._refresh_model_info()

    def _pick_image(self):
        try:
            # IMPORTANT: pass parent and use space-separated patterns (or a tuple)
            path = fd.askopenfilename(
                parent=self.root,
                title="Select image",
                initialdir=os.path.expanduser("~"),
                filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
                # or: filetypes=[("Image files", (".png", ".jpg", ".jpeg")), ("All files", "*")]
            )
            if path:
                self._image_path = path
                messagebox.showinfo("Selected", path, parent=self.root)
        except Exception as e:
            messagebox.showerror("File dialog error", str(e), parent=self.root)


    def _run_model(self):
        """Run the selected model and print results to the output box."""
        key = self.model_var.get().strip()
        ModelClass = self._models[key]
        model = ModelClass()

        if key == "Sentiment Analysis (Text)":
            text = self.text_entry.get().strip()
            if not text:
                messagebox.showerror("Input missing", "Please enter some text.")
                return
            try:
                result = model.run(text)  # e.g., 'NEGATIVE (98.2%)'
            except Exception as e:
                result = f"Sentiment error: {e}"
            self.out.insert("end", f"\nSentiment → {result}\n")
            self.out.see("end")
            return

        if key == "Image Classification":
            if not self._image_path:
                messagebox.showerror("Input missing", "Please choose an image.")
                return
            try:
                preds = model.run(self._image_path, top_k=5)
            except TypeError:
                preds = model.run(self._image_path)
            if isinstance(preds, dict):
                preds = [preds]
            self.out.configure(font=("Courier New", 11))
            self.out.insert("end", "\n" + self._format_image_results(preds) + "\n")
            self.out.see("end")
            return


    def _refresh_model_info(self):
        # clear table
        for i in self.info_tree.get_children():
            self.info_tree.delete(i)
        # get metadata
        ModelClass = self._models[self.model_var.get()]
        info = ModelClass().get_info()
        for k, v in info.items():
            self.info_tree.insert("", "end", text=k, values=(str(v),))

    def run(self):
        self.root.mainloop()
