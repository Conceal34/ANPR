# 🚗 ANPR — Automatic Number Plate Recognition System

A real-time Automatic Number Plate Recognition system built with Python, YOLOv8 for detection, and PaddleOCR for text extraction. The system processes video footage to detect vehicles, isolate license plates, and log recognised plate numbers with frame-level precision.

---

## 🧠 How It Works

1. **Detection** — YOLOv8 model scans each video frame to detect vehicles and license plate regions
2. **Tracking** — Detected plates are tracked across frames using the SORT algorithm for consistent identification
3. **OCR** — PaddleOCR extracts alphanumeric text from each cropped plate region
4. **Interpolation** — Missing or low-confidence detections between frames are filled via CSV interpolation for smoother results
5. **Visualisation** — Final output is rendered as an annotated video with bounding boxes and plate text overlaid per frame

---

## 📁 Project Structure

```
ANPR/
├── Model/                            # YOLOv8 model weights
├── Process_files/                    # Intermediate processing outputs
├── icons/                            # UI assets for Tkinter interface
├── test-files/                       # Sample video inputs for testing
├── main.py                           # Core detection + tracking pipeline
├── add_data.py                       # Data logging and CSV management
├── visualise.py                      # Annotated video rendering
├── tkinter-try.py                    # Desktop GUI interface (Tkinter)
├── version-1_without_pre_process.py  # Baseline version without preprocessing
├── sort                              # SORT multi-object tracker
├── test.csv                          # Raw per-frame detection output
├── test_interpolated.csv             # Interpolated and smoothed results
└── requirements.txt                  # Python dependencies
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **YOLOv8** (Ultralytics 8.2.0) | Vehicle & license plate detection |
| **PaddleOCR** (2.7.3) | License plate text recognition |
| **PaddlePaddle** (2.6.2) | Deep learning backend for PaddleOCR |
| **OpenCV** | Video processing & frame manipulation |
| **SORT** | Multi-object tracking across frames |
| **Tkinter** | Desktop GUI interface |
| **Pandas** | CSV data handling & interpolation |
| **NumPy** | Numerical operations |
| **Pillow** | Image preprocessing |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- GPU recommended for real-time performance (CPU supported)

### Installation

```bash
# Clone the repository
git clone https://github.com/Conceal34/ANPR.git
cd ANPR

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PaddlePaddle installation may vary by platform and CUDA version. Refer to the [official PaddlePaddle install guide](https://www.paddlepaddle.org.cn/install/quick) if you run into issues.

### Running the System

```bash
# Run the main detection + OCR pipeline
python main.py

# Generate annotated output video
python visualise.py

# Launch the desktop GUI
python tkinter-try.py
```

---

## 📊 Output

The pipeline produces three outputs:

- **`test.csv`** — Raw per-frame results including plate text, confidence scores, and bounding box coordinates
- **`test_interpolated.csv`** — Smoothed dataset with interpolated values for frames where detection was missed
- **Annotated video** — Visual output with bounding boxes and recognised plate numbers rendered on each frame

---

## 🔬 Versions

Two implementations are included:

- **`main.py`** — Full pipeline with preprocessing for improved OCR accuracy
- **`version-1_without_pre_process.py`** — Baseline version without image preprocessing, useful for benchmarking

---

## 📌 Notes

- Place input video files inside the `test-files/` directory before running
- YOLOv8 model weights should be present in the `Model/` folder
- PaddleOCR will download required models on first run automatically
- For best results use video with adequate resolution and lighting conditions

---

## 👤 Author

**Vinner**  
MCA Student · Christ University, Delhi NCR  
IEEE Published Researcher · Full-Stack Engineer

[Portfolio](https://vinner-portfolio.netlify.app/) · [GitHub](https://github.com/Conceal34) · [LinkedIn](https://www.linkedin.com/in/vinnerhooda/)
