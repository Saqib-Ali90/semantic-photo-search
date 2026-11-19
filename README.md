# 🔍 Local AI Semantic Photo Search

A fast, 100% private desktop application built with Python and Tkinter that searches your local photo library using "natural language text", "reference images", or a "hybrid combination of both".

Powered by local Vision-Language embedding models OpenAI's CLIP, all indexing and search operations happen completely offline on your PC—your personal photos never leave your device.

---

## ✨ Key Features

* 100% Local & Private: Runs completely offline on your CPU or GPU with zero cloud API calls.
* Multimodal Search:
  * Text-to-Image: Describe scenes, objects, or attributes (e.g., `"a kid in a yellow suit"` or `"بلی"`).
  * Image-to-Image: Browse a reference photo to find visually similar images across your gallery.
  * Hybrid Search: Combine both a reference image and a text prompt simultaneously for refined results.
* Multilingual Support: Supports searching in Urdu (اردو), --not yet included--  and 100+ other languages when using Google's multilingual SigLIP model (coming soon).
* Recursive Subfolder Indexing: Automatically scans nested folders and subfolders (`os.walk`) sequentially to keep RAM usage low.
* Instant Vector Caching: Saves computed image embeddings locally (`embeddings.pkl`) so subsequent app launches are near-instant.
* Interactive Thumbnail Gallery: Displays top statistically ranked matches in a scrollable grid. Click any thumbnail to open the full-resolution photo.

---

## 🛠️ Tech Stack

* GUI Framework: Python `Tkinter` & `Pillow` (PIL)
* AI / ML Engine: `PyTorch` & Hugging Face `Transformers` (`google/siglip-base-patch16-256-multilingual` / `openai/clip-vit-base-patch16`)
* Math & Vector Ranking: `NumPy` & Cosine Similarity with statistical outlier thresholding

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.9 or higher
* Git

### 2. Clone the Repository
```bash
git clone https://github.com
cd semantic-photo-search
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Photo Folder
1. Create an `images` folder inside the project directory (or update `IMAGE_FOLDER` at the top of `gui_search.py` to point to your photo directory).
2. Add your test photos (subfolders are supported automatically).

### 5. Run the Application
```bash
python gui_search_CLIP.py
```
(Note: On the very first launch, the app will download the pretrained vision model from Hugging Face and build your local `embeddings.pkl` cache. All future launches will load immediately.)

---

## 📦 Packaging as a Standalone Windows `.exe` (Optional)

To create a double-clickable desktop executable:
```bash
pip install pyinstaller
pyinstaller --noconsole --onefile gui_search_CLIP.py
```
Your executable will be generated inside the `dist/` folder. Keep your `images/` folder in the same directory when running the `.exe`.

---

## 📁 Project Structure

```text
├── gui_search.py       # Main Tkinter desktop application & AI search engine
├── requirements.txt    # Python package dependencies
├── .gitignore          # Excludes local photo folders, cache files, and build artifacts
└── README.md           # Project documentation
```

---

## 🔮 Future Roadmap

* [ ] Add live microphone voice search (English & Urdu)
* [ ] Add a visual folder-picker button inside the UI
* [ ] Port the quantized SigLIP model to Android (Kotlin + ONNX Runtime) for on-device mobile gallery search

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

Regards:
Saqib Ali

You can find me on Youtube and Facebook for demonstartion videos /@astrosaqi
Youtube link: https://youtu.be/ISNI2HTmqeo

<img width="858" height="625" alt="app SS3" src="https://github.com/user-attachments/assets/059dc736-7e82-4057-8b78-ba75642a8d08" />
<img width="852" height="619" alt="app SS2" src="https://github.com/user-attachments/assets/120881f2-90d9-41a4-96fe-715baf1b8e20" />
<img width="1072" height="753" alt="app SS1" src="https://github.com/user-attachments/assets/25877880-fd19-4283-804f-3ddcef8258f5" />
