import os, pickle, threading
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from transformers import AutoProcessor, AutoModel

IMAGE_FOLDER = r"./images"
CACHE_FILE = "embeddings_siglip_multi.pkl"
MODEL_ID = "google/siglip-base-patch16-256-multilingual"


class MultilingualPhotoSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Local AI Multilingual Photo Search (English & اردو)")
        self.root.geometry("1050x700")
        self.root.configure(bg="#f4f4f9")

        self.model = None
        self.processor = None
        self.image_paths = []
        self.image_embeddings = None
        self.query_img_path = ""
        self.thumb_refs = []

        self.setup_ui()
        threading.Thread(target=self.init_ai, daemon=True).start()

    def set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def setup_ui(self):
        top = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid", padx=15, pady=12)
        top.pack(fill="x", padx=15, pady=12)

        tk.Label(top, text="Text Query (English / اردو):", font=("Segoe UI", 10, "bold"), bg="#ffffff").grid(row=0, column=0, sticky="w", pady=4)
        self.text_entry = tk.Entry(top, font=("Segoe UI", 12), width=42, bd=1, relief="solid")
        self.text_entry.grid(row=0, column=1, padx=10, pady=4, sticky="w")

        tk.Label(top, text="Reference Image:", font=("Segoe UI", 10, "bold"), bg="#ffffff").grid(row=1, column=0, sticky="w", pady=4)
        self.img_var = tk.StringVar(value="No image selected")
        tk.Label(top, textvariable=self.img_var, font=("Segoe UI", 9, "italic"), fg="#555555", bg="#ffffff", width=48, anchor="w").grid(row=1, column=1, padx=10, pady=4, sticky="w")

        btns = tk.Frame(top, bg="#ffffff")
        btns.grid(row=1, column=2, padx=5)
        tk.Button(btns, text="Browse...", font=("Segoe UI", 9), command=self.browse_img, bg="#e0e0e0", relief="flat", padx=10, pady=4).pack(side="left", padx=3)
        tk.Button(btns, text="Clear", font=("Segoe UI", 9), command=self.clear_img, bg="#ffe0e0", relief="flat", padx=8, pady=4).pack(side="left", padx=3)

        self.search_btn = tk.Button(top, text="Search Photos", font=("Segoe UI", 11, "bold"), bg="#0066cc", fg="white", relief="flat", padx=20, pady=12, state="disabled", command=self.start_search)
        self.search_btn.grid(row=0, column=3, rowspan=2, padx=15)

        status_bar = tk.Frame(self.root, bg="#f4f4f9")
        status_bar.pack(fill="x", padx=15)
        self.status_var = tk.StringVar(value="Loading Multilingual SigLIP Model... Please wait.")
        tk.Label(status_bar, textvariable=self.status_var, font=("Segoe UI", 9, "bold"), bg="#f4f4f9").pack(side="left")
        self.progress = ttk.Progressbar(status_bar, orient="horizontal", length=220, mode="indeterminate")
        self.progress.pack(side="right")
        self.progress.start(10)

        gallery = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid")
        gallery.pack(fill="both", expand=True, padx=15, pady=12)

        self.canvas = tk.Canvas(gallery, bg="#ffffff", highlightthickness=0)
        scrollbar = ttk.Scrollbar(gallery, orient="vertical", command=self.canvas.yview)
        self.grid_frame = tk.Frame(self.canvas, bg="#ffffff")
        self.grid_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def browse_img(self):
        path = filedialog.askopenfilename(title="Select Query Image", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self.query_img_path = path
            self.img_var.set(os.path.basename(path))

    def clear_img(self):
        self.query_img_path = ""
        self.img_var.set("No image selected")
    def init_ai(self):
        try:
            self.model = AutoModel.from_pretrained(MODEL_ID)
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            if os.path.exists(CACHE_FILE):
                self.set_status("Loading cached multilingual embeddings...")
                with open(CACHE_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.image_paths, self.image_embeddings = data["paths"], data["embeddings"]
            else:
                self.set_status("Scanning folders and indexing photos...")
                paths = [os.path.join(r, fn) for r, _, fs in os.walk(IMAGE_FOLDER) for fn in fs if fn.lower().endswith((".png", ".jpg", ".jpeg"))]
                if not paths:
                    self.set_status(f"No images found in {IMAGE_FOLDER}")
                    return
                embs, valid = [], []
                for i, p in enumerate(paths):
                    try:
                        with Image.open(p) as im:
                            im = im.convert("RGB") if im.mode != "RGB" else im
                            with torch.no_grad():
                                out = self.model.get_image_features(**self.processor(images=im, return_tensors="pt"))
                                vec = out.pooler_output if hasattr(out, "pooler_output") else out
                                embs.append(vec / vec.norm(dim=-1, keepdim=True))
                                valid.append(p)
                        if (i + 1) % 5 == 0 or (i + 1) == len(paths):
                            self.set_status(f"Indexed {i + 1}/{len(paths)} images...")
                    except Exception:
                        continue
                self.image_paths, self.image_embeddings = valid, torch.cat(embs, dim=0)
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump({"paths": valid, "embeddings": self.image_embeddings}, f)
            self.set_status(f"Ready! {len(self.image_paths)} photos indexed (Supports English & اردو).")
            self.root.after(0, lambda: self.search_btn.config(state="normal"))
        except Exception as err:
            print("INIT ERROR:", err)
            self.set_status(f"Init Error: {err}")
        finally:
            self.root.after(0, self.progress.stop)

    def start_search(self):
        txt, img = self.text_entry.get().strip(), self.query_img_path
        if not txt and not img:
            messagebox.showwarning("Empty Input", "Enter English/Urdu text, select an image, or both.")
            return
        for w in self.grid_frame.winfo_children():
            w.destroy()
        self.thumb_refs.clear()
        self.search_btn.config(state="disabled")
        self.progress.start(10)
        self.set_status("Searching...")
        threading.Thread(target=self.run_search, args=(txt, img), daemon=True).start()

    def run_search(self, txt, img):
        try:
            vecs = []
            with torch.no_grad():
                if txt:
                    inputs = self.processor(text=[txt], return_tensors="pt", padding="max_length", truncation=True)
                    out = self.model.get_text_features(**inputs)
                    t_vec = out.pooler_output if hasattr(out, "pooler_output") else out
                    vecs.append(t_vec / t_vec.norm(dim=-1, keepdim=True))
                if img:
                    with Image.open(img) as q_im:
                        q_im = q_im.convert("RGB") if q_im.mode != "RGB" else q_im
                        out = self.model.get_image_features(**self.processor(images=q_im, return_tensors="pt"))
                        i_vec = out.pooler_output if hasattr(out, "pooler_output") else out
                        vecs.append(i_vec / i_vec.norm(dim=-1, keepdim=True))

                q_emb = torch.stack(vecs).mean(dim=0)
                q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

                raw_sim = self.image_embeddings @ q_emb.T
                if hasattr(self.model, "logit_scale") and hasattr(self.model, "logit_bias"):
                    logits = raw_sim * self.model.logit_scale.exp() + self.model.logit_bias
                    scores = torch.sigmoid(logits).squeeze(dim=-1).tolist()
                else:
                    scores = raw_sim.squeeze(dim=-1).tolist()

            ranked = sorted(zip(self.image_paths, scores), key=lambda pair: pair[-1], reverse=True)
            vals = [s for _, s in ranked]
            thresh = np.mean(vals) + 1.4 * np.std(vals)
            matches = [(p, s) for p, s in ranked if s >= thresh] or ranked[:3]
            self.root.after(0, lambda: self.show_thumbs(matches))
        except Exception as err:
            print("SEARCH ERROR:", err)
            self.set_status(f"Search Error: {err}")
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.search_btn.config(state="normal"))

    def show_thumbs(self, matches):
        for idx, (path, score) in enumerate(matches):
            card = tk.Frame(self.grid_frame, bg="#f9f9fb", bd=1, relief="solid", padx=8, pady=8)
            card.grid(row=idx // 4, column=idx % 4, padx=10, pady=10, sticky="nsew")
            try:
                with Image.open(path) as im:
                    im.thumbnail((210, 210))
                    tk_im = ImageTk.PhotoImage(im)
                    self.thumb_refs.append(tk_im)
                tk.Button(card, image=tk_im, bd=0, cursor="hand2", command=lambda p=path: os.startfile(p)).pack()
                name = os.path.basename(path)
                tk.Label(card, text=name[:21] + "..." if len(name) > 24 else name, font=("Segoe UI", 9, "bold"), bg="#f9f9fb").pack(pady=(5, 0))
                tk.Label(card, text=f"Score: {score:.3f}", font=("Segoe UI", 8), fg="#0066cc", bg="#f9f9fb").pack()
            except Exception:
                continue
        self.set_status(f"Done! Showing {len(matches)} matches. Click any thumbnail to open.")
        self.root.after(0, self.progress.stop)
        self.search_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    MultilingualPhotoSearchApp(root)
    root.mainloop()
