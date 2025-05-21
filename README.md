# 🎭 Facial Emotion Analyzer

This project utilizes [Py-Feat](https://github.com/cosanlab/py-feat) to automatically analyze micro-expressions in video files. It generates various visualizations including **emotion trend line charts, pie charts, bar charts, radar charts, heatmaps, and clustering diagrams**, which are compiled into a professional PDF report. It is well-suited for teaching demonstrations, research assistance, and facial emotion visualization tasks.

---

## 📁 Project Structure

```
facial-analysis/
├── scripts/                 # Main logic and visualization scripts
├── videos/                  # Videos to be analyzed
├── outputs/                 # Output CSV files, PDF reports, interactive HTML charts
├── test_result/             # Intermediate debugging files
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📅 Clone the Repository (For Beginners)

If you're not familiar with Git or the command line, follow the steps below to download the project:

### Step 0: Install Git

* **Windows**: Download from [git-scm.com](https://git-scm.com/download/win) and install with default settings.
* **Mac**: Open Terminal and type `git`, follow the prompts to install.
* **Linux (Ubuntu)**:

```bash
sudo apt update
sudo apt install git
```

### Step 1: Open your terminal and navigate to a desired directory

For example, to navigate to your Downloads folder:

```bash
cd ~/Downloads
```

### Step 2: Clone the project

```bash
git clone https://github.com/diverHansun/facial-emotion-analyzer.git
```

The project will be downloaded as `facial-emotion-analyzer`. Enter this directory to proceed.

---

## 🛆 Install Dependencies

Make sure you're using Python 3.8 or higher. It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

### Step 1: Open your terminal and go to the project root

**Windows:**

```bash
cd "C:\Users\YourUsername\Downloads\facial-emotion-analyzer"
```

**Mac/Linux:**

```bash
cd ~/Downloads/facial-emotion-analyzer
```

### Step 2: Run the main script from the `facial-analysis` folder

```bash
python -m scripts.emotion_analysis.main videos/xxx.mp4 --process_sampling_rate x --fps x
```

---

## ⚙️ Command-Line Arguments

### ✅ Required

| Argument                  | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `video_path`              | Path to the video file (e.g., `videos/test5.mp4`)    |
| `--process_sampling_rate` | Sampling interval (default: 5 frames)                |
| `--fps`                   | Video frame rate for timeline accuracy (default: 30) |

### ✅ Optional

| Argument                  | Default Value                            | Description                                                                                     |
| ------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--multi_face`            | False                                    | Analyze multiple faces per frame (if True); only the most prominent face is analyzed by default |
| `--start_frame`           | None                                     | Specify starting frame (aligned to nearest sampled frame)                                       |
| `--end_frame`             | None                                     | Specify ending frame                                                                            |
| `--output_csv`            | `outputs/facial_expression_analysis.csv` | Path for CSV output                                                                             |
| `--output_pdf`            | `outputs/emotion_report.pdf`             | Path for PDF report                                                                             |
| `--method`                | `tsne`                                   | Dimensionality reduction method (`tsne` or `umap`)                                              |
| `--cluster_sampling_rate` | Auto (or 5 if specified)                 | Sampling rate for clustering visualization                                                      |
| `--perplexity`            | Auto (or 30 if specified)                | t-SNE hyperparameter                                                                            |
| `--n_neighbors`           | Auto (or 15 if specified)                | Number of neighbors for UMAP clustering                                                         |

---

## 📄 Output

After execution, results will be saved in the `outputs/` folder:

* `facial_expression_analysis.csv`: Frame-by-frame emotion scores
* `emotion_dynamic_face.html`: Interactive line chart visualization
* `emotion_report.pdf`: A comprehensive report with the following charts:

| Chart              | Description                          |
| ------------------ | ------------------------------------ |
| 📈 Line Chart      | Emotion intensity over time          |
| 🥧 Pie Chart       | Proportion of dominant emotions      |
| 📊 Bar Chart       | Frequency of dominant emotions       |
| 🔥 Heatmap         | 2D distribution of emotion intensity |
| 🔸 Radar Chart     | Average emotion intensity            |
| 🧠 Clustering Plot | Emotion clusters via t-SNE/UMAP      |

---

## 📌 Notes

* By default, only the most prominent face is analyzed. Use `--multi_face` to analyze all detected faces. The output will then include a `face_id` field.
* Recommended video length: **2–5 minutes** (too short: not meaningful; too long: time-consuming).
* Ensure `--fps` matches the actual video frame rate; default is 30.
* Default sampling rate is every 10 frames. Adjust `--process_sampling_rate` as needed.
* The interactive HTML chart is not included in the PDF and opens in a browser automatically after generation.
* To display Chinese fonts in the PDF, include `simhei.ttf` in the project root. You can skip this if you don't need Chinese text rendering.
* Temporary images used during generation are automatically deleted.
* Output paths and parameters are customizable via CLI for batch processing or integration.

---

## 📚 References

This project is based on or inspired by:

* [py-feat](https://github.com/cosanlab/py-feat)
* [deepface](https://github.com/serengil/deepface)
* Libraries: `matplotlib`, `seaborn`, `reportlab`, `scikit-learn`, etc.

---

## 📬 Contact & Feedback

For issues or suggestions, please open an [Issue](https://github.com/diverHansun/facial-emotion-analyzer/issues).

---

✅ **Enjoy your AI-powered facial expression analyzer!**
