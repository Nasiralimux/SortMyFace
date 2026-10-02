import os
import sys
import pickle
import shutil
import time
import threading

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox

BASE_FOLDER = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
DATABASE_FILE = os.path.join(BASE_FOLDER, "face_database.pkl")
BACKUP_DIR = os.path.join(BASE_FOLDER, "backups")

DET_SIZE = (640, 640)
SIM_THRESHOLD = 0.36
EMBED_DIM = 512
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

face_app = None

# ---- Color palette ----
COL_BG = "#F3F4F6"
COL_HEADER = "#FFFFFF"
COL_TITLE = "#1E3A8A"
COL_SUBTEXT = "#6B7280"
COL_ACCENT = "#4F46E5"
COL_ACCENT_HOVER = "#4338CA"
COL_SAFE = "#2563EB"
COL_SAFE_HOVER = "#1D4ED8"
COL_WARN = "#D97706"
COL_WARN_HOVER = "#B45309"
COL_DANGER = "#DC2626"
COL_DANGER_HOVER = "#B91C1C"
LOG_COLORS = {
    "info": "#1F2937",
    "success": "#059669",
    "warn": "#B45309",
    "error": "#DC2626",
}


def init_models():
    global face_app
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        messagebox.showerror("SortMyFace", "insightface is not installed.\nRun: pip install -r requirements.txt")
        sys.exit(1)
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=DET_SIZE)


def load_database():
    if not os.path.exists(DATABASE_FILE):
        return {}
    try:
        with open(DATABASE_FILE, "rb") as f:
            db = pickle.load(f)
    except Exception:
        return {}
    for embs in db.values():
        if embs and len(embs[0]) != EMBED_DIM:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            shutil.copy2(DATABASE_FILE, os.path.join(BACKUP_DIR, f"face_database_old_model_{stamp}.pkl"))
            return {}
    return db


def save_database(db):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(DATABASE_FILE):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(DATABASE_FILE, os.path.join(BACKUP_DIR, f"face_database_{stamp}.pkl"))
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(db, f)


def detect_and_embed(img):
    """Returns the normalized embedding of the largest detected face, or None."""
    faces = face_app.get(img)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return largest.normed_embedding


def cosine_sim(a, b):
    return float(np.dot(a, b))


def identify(embedding, db):
    best_name, best_score = None, -1.0
    for name, embs in db.items():
        for e in embs:
            score = cosine_sim(embedding, e)
            if score > best_score:
                best_score, best_name = score, name
    if best_name is not None and best_score >= SIM_THRESHOLD:
        return best_name, best_score
    return None, best_score


def unique_dest(dest_folder, filename):
    os.makedirs(dest_folder, exist_ok=True)
    dest = os.path.join(dest_folder, filename)
    if not os.path.exists(dest):
        return dest
    stem, ext = os.path.splitext(filename)
    i = 1
    while True:
        candidate = os.path.join(dest_folder, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


class SortMyFaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("SortMyFace")
        self.geometry("760x620")
        self.configure(fg_color=COL_BG)

        self.current_folder = sys.argv[1].rstrip("\\/") if len(sys.argv) > 1 else BASE_FOLDER
        self.db = load_database()
        self.busy = False

        self._build_ui()
        self._refresh_stats()

    def _build_ui(self):
        header = ctk.CTkFrame(self, corner_radius=14, fg_color=COL_HEADER,
                               border_width=1, border_color="#E5E7EB")
        header.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(header, text="SORTMYFACE", font=("Segoe UI", 26, "bold"),
                     text_color=COL_TITLE).pack(pady=(12, 0))
        self.folder_label = ctk.CTkLabel(header, text="", font=("Segoe UI", 12), text_color=COL_SUBTEXT)
        self.folder_label.pack(pady=(4, 0))
        self.stats_label = ctk.CTkLabel(header, text="", font=("Segoe UI", 12, "bold"), text_color=COL_SAFE)
        self.stats_label.pack(pady=(2, 12))

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=8)
        actions = [
            ("Sort Photos in This Folder", self.sort_this_folder, COL_ACCENT, COL_ACCENT_HOVER),
            ("Sort Photos in Another Folder", self.sort_another_folder, COL_ACCENT, COL_ACCENT_HOVER),
            ("Add New Person", self.add_person, COL_SAFE, COL_SAFE_HOVER),
            ("View Registered People", self.view_people, "#475569", "#334155"),
            ("Remove Person", self.remove_person, COL_WARN, COL_WARN_HOVER),
            ("Rebuild Face Database", self.rebuild_database, COL_DANGER, COL_DANGER_HOVER),
        ]
        self.action_buttons = []
        for i, (label, cmd, color, hover) in enumerate(actions):
            b = ctk.CTkButton(btns, text=label, height=46, corner_radius=10,
                               font=("Segoe UI", 13, "bold"), anchor="center",
                               fg_color=color, hover_color=hover, text_color="#FFFFFF",
                               command=cmd)
            b.grid(row=i // 2, column=i % 2, padx=8, pady=8, sticky="ew")
            self.action_buttons.append(b)
        btns.grid_columnconfigure((0, 1), weight=1)

        self.progress = ctk.CTkProgressBar(self, progress_color=COL_ACCENT,
                                            fg_color="#E5E7EB")
        self.progress.set(0)
        self.progress.pack(fill="x", padx=16, pady=(4, 8))

        self.log_box = ctk.CTkTextbox(self, height=220, font=("Consolas", 11),
                                       fg_color="#FFFFFF", text_color="#1F2937",
                                       corner_radius=10, border_width=1, border_color="#E5E7EB")
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.log_box.configure(state="disabled")
        for tag, color in LOG_COLORS.items():
            self.log_box._textbox.tag_configure(tag, foreground=color)

    def log(self, text, level="info"):
        tag = level if level in LOG_COLORS else "info"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_progress(self, current, total):
        self.progress.set(current / total if total else 0)

    def _refresh_stats(self):
        self.folder_label.configure(text=f"Current folder: {self.current_folder}")
        total_embs = sum(len(v) for v in self.db.values())
        self.stats_label.configure(text=f"Database: {len(self.db)} people, {total_embs} reference embeddings")

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in self.action_buttons:
            b.configure(state=state)

    def run_async(self, target, *args):
        if self.busy:
            return
        self.set_busy(True)
        threading.Thread(target=self._wrap, args=(target, args), daemon=True).start()

    def _wrap(self, target, args):
        try:
            target(*args)
        except Exception as e:
            self.after(0, lambda: self.log(f"ERROR: {e}"))
        finally:
            self.after(0, lambda: self.set_busy(False))
            self.after(0, self._refresh_stats)

    def sort_this_folder(self):
        self.run_async(self._do_sort, self.current_folder)

    def sort_another_folder(self):
        folder = filedialog.askdirectory(initialdir=self.current_folder, title="Select Photo Folder")
        if folder:
            self.current_folder = folder
            self._refresh_stats()
            self.run_async(self._do_sort, folder)

    def _do_sort(self, folder):
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
                 and os.path.isfile(os.path.join(folder, f))]
        if not files:
            self.after(0, lambda: self.log("No supported images found in this folder."))
            return
        total = len(files)
        counts = {"unknown": 0, "no_face": 0, "skipped": 0}
        per_person = {}
        start = time.time()
        self.after(0, lambda: self.log(f"Scanning {total} images in {folder}"))
        for i, fname in enumerate(files, 1):
            src = os.path.join(folder, fname)
            try:
                img = cv2.imread(src)
                if img is None:
                    raise ValueError("unreadable")
                emb = detect_and_embed(img)
                if emb is None:
                    dest_folder = os.path.join(folder, "Categorized", "No Face Detected")
                    counts["no_face"] += 1
                    self.after(0, lambda f=fname: self.log(f"{f} → No Face Detected", "warn"))
                else:
                    name, score = identify(emb, self.db)
                    if name:
                        dest_folder = os.path.join(folder, "Categorized", name)
                        per_person[name] = per_person.get(name, 0) + 1
                    else:
                        dest_folder = os.path.join(folder, "Categorized", "Unknown")
                        counts["unknown"] += 1
                        self.after(0, lambda f=fname, s=score: self.log(f"{f} → Unknown (best match {s:.2f})", "warn"))
                dest = unique_dest(dest_folder, fname)
                shutil.move(src, dest)
            except Exception:
                counts["skipped"] += 1
            self.after(0, lambda i=i, t=total: self.set_progress(i, t))
        elapsed = time.time() - start
        self.after(0, lambda: self.log("\nSORT COMPLETE", "success"))
        self.after(0, lambda: self.log("-" * 32))
        for name, c in per_person.items():
            self.after(0, lambda name=name, c=c: self.log(f"{name}: {c}", "success"))
        self.after(0, lambda: self.log(f"Unknown: {counts['unknown']}", "warn"))
        self.after(0, lambda: self.log(f"No Face Detected: {counts['no_face']}", "warn"))
        self.after(0, lambda: self.log(f"Skipped: {counts['skipped']}", "error" if counts["skipped"] else "info"))
        self.after(0, lambda: self.log(f"Total: {total}"))
        self.after(0, lambda: self.log(f"Time: {elapsed:.1f}s"))
        self.after(0, lambda: self.set_progress(0, 1))

    def add_person(self):
        dialog = ctk.CTkInputDialog(text="Enter person's name:", title="Add New Person")
        name = dialog.get_input()
        if not name or not name.strip():
            return
        name = name.strip()
        if name in self.db:
            messagebox.showerror("SortMyFace", f'A person named "{name}" already exists.')
            return
        files = filedialog.askopenfilenames(
            initialdir=self.current_folder, title="Select Reference Images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not files:
            return
        self.run_async(self._do_add_person, name, list(files))

    def _do_add_person(self, name, files):
        embeddings = []
        total = len(files)
        for i, path in enumerate(files, 1):
            img = cv2.imread(path)
            if img is not None:
                emb = detect_and_embed(img)
                if emb is not None:
                    embeddings.append(emb)
            self.after(0, lambda i=i, t=total: self.set_progress(i, t))
        if not embeddings:
            self.after(0, lambda: self.log(f'No usable faces found for "{name}".', "error"))
            return
        self.db[name] = embeddings
        save_database(self.db)
        self.after(0, lambda: self.log(f'Added "{name}" with {len(embeddings)} reference embeddings.', "success"))
        self.after(0, lambda: self.set_progress(0, 1))

    def view_people(self):
        if not self.db:
            self.log("No people registered yet.")
            return
        self.log("REGISTERED PEOPLE")
        self.log("-" * 32)
        for i, (name, embs) in enumerate(self.db.items(), 1):
            self.log(f"{i}. {name} — {len(embs)} reference images")

    def remove_person(self):
        if not self.db:
            messagebox.showinfo("SortMyFace", "No people registered.")
            return
        names = list(self.db.keys())
        dialog = ctk.CTkInputDialog(text="Type the exact name to remove:\n" + "\n".join(names), title="Remove Person")
        name = dialog.get_input()
        if not name or name not in self.db:
            messagebox.showwarning("SortMyFace", "No matching person selected.")
            return
        if messagebox.askyesno("Confirm", f'Permanently remove "{name}"?'):
            del self.db[name]
            save_database(self.db)
            self.log(f'Removed "{name}".', "success")
            self._refresh_stats()

    def rebuild_database(self):
        known_dir = os.path.join(BASE_FOLDER, "known_faces")
        if not os.path.isdir(known_dir):
            messagebox.showerror("SortMyFace", f"known_faces folder not found at {known_dir}")
            return
        if not messagebox.askyesno("Confirm", "This rebuilds the ENTIRE database from known_faces\\. Continue?"):
            return
        self.run_async(self._do_rebuild, known_dir)

    def _do_rebuild(self, known_dir):
        new_db = {}
        people = [p for p in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, p))]
        for person in people:
            pdir = os.path.join(known_dir, person)
            embeddings = []
            imgs = [f for f in os.listdir(pdir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]
            for i, fname in enumerate(imgs, 1):
                img = cv2.imread(os.path.join(pdir, fname))
                if img is not None:
                    emb = detect_and_embed(img)
                    if emb is not None:
                        embeddings.append(emb)
                self.after(0, lambda i=i, t=len(imgs): self.set_progress(i, t))
            if embeddings:
                new_db[person] = embeddings
        save_database(new_db)
        self.db = new_db
        self.after(0, lambda: self.log(f"Database rebuilt with {len(new_db)} people.", "success"))
        self.after(0, lambda: self.set_progress(0, 1))


def main():
    init_models()
    app = SortMyFaceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
