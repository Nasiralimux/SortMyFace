# SortMyFace

A Windows desktop app that automatically sorts your photos into folders by the people in them, using facial recognition — no cloud upload, no manual sorting.

## Problem It Solves

Anyone with thousands of unsorted photos (family archives, event photography, phone backups) knows the pain of manually dragging images into per-person folders. SortMyFace automates that: point it at a folder, and it moves every photo into `Categorized\<Person Name>` automatically.

## Features

- **One portable app** — a single `.exe`, no installation, no Python required to run it
- **Three-way sorting** — every photo lands in exactly one of:
  - `Categorized\<Person Name>` — a registered face was recognized
  - `Categorized\Unknown` — a face was found but didn't match anyone registered
  - `Categorized\No Face Detected` — no face could be found in the image
- **Add/Remove people** through simple dialogs — pick reference photos, done
- **View registered people** and how many reference images each has
- **Rebuild the database** from a folder structure if needed
- **Safe by design** — moves (never deletes) files, auto-backs up the database before any change, never overwrites reference photos
- **Clean GUI** — live progress bar, color-coded activity log, sort summary report

## Technology Stack

- **Python 3.14**
- **InsightFace** (SCRFD for face detection, ArcFace/`buffalo_l` for recognition) — the current standard in open-source face recognition
- **OpenCV** for image I/O
- **CustomTkinter** for the interface
- **PyInstaller** to package everything into a single `.exe`

## How Face Recognition Works

1. **Detection** — SCRFD locates every face in a photo; the largest one is used.
2. **Embedding** — ArcFace converts that face into a 512-dimensional numeric vector (an "embedding") that represents its unique features.
3. **Matching** — the embedding is compared against every registered reference embedding using cosine similarity.
4. **Decision** — if the best match clears the similarity threshold, the photo is filed under that person's name; otherwise it's `Unknown`.

Recognition improves with more, varied reference photos per person (different angles, lighting, expressions).

## Installation

**Option 1 — Just run it:**
Download `SortMyFace.exe` and double-click it. First launch needs internet once (to download the recognition model, ~300MB); every run after that works offline.

**Option 2 — Run from source:**
```
pip install -r requirements.txt
python SortMyFace.py
```

## Usage

1. Launch SortMyFace (in any folder, or via a portable copy).
2. **Add New Person** → enter a name → select several clear reference photos of them.
3. **Sort Photos in This Folder** (or **in Another Folder**) → watch the progress bar and log.
4. Check the `Categorized\` subfolder created inside your photo folder.

## How Photo Categorization Works

For every image, SortMyFace detects the largest face, generates its embedding, and compares it against all registered people. The file is then moved (not copied) into the appropriate `Categorized\` subfolder, with automatic handling of duplicate filenames.

## Folder Structure

```
YourPhotoFolder\
└── Categorized\
    ├── Person Name\
    ├── Unknown\
    └── No Face Detected\
```

The app itself keeps its database and backups alongside `SortMyFace.exe`, separate from whatever photo folder you point it at.

## Limitations

- Uses only the **largest** face per photo — group photos are filed by whichever face is biggest, not all faces present
- Recognition accuracy depends on reference photo quality and variety
- No built-in UI (yet) to adjust the similarity threshold — it's a fixed constant in the code

## Future Improvements

- Multi-face photos filed under all recognized people, not just the largest face
- In-app threshold adjustment slider
- Cross-platform builds (macOS/Linux)
- Duplicate photo detection

## License

MIT — see [LICENSE](LICENSE).
