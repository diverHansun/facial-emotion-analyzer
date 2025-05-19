
# 🎭 Facial Emotion Analyzer（面部情绪分析与可视化工具）

本项目基于 [Py-Feat](https://github.com/cosanlab/py-feat) 实现视频中微表情的自动分析，生成包括 **情绪趋势图、饼状图、柱状图、雷达图、热力图、聚类图** 等分析图，并整合为一份专业的 PDF 报告，适用于教学展示、科研辅助与人脸情绪可视化任务。

---

## 📁 项目结构

```
facial-analysis/
├── scripts/                 # 主程序逻辑文件（main、图表绘制等）
├── videos/                  # 存放待分析视频
├── outputs/                 # 存放输出的 CSV 数据，PDF 报告，可交互HTML
├── test_result/             # 存放中间调试数据
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明文件
```

---

## 📥 克隆项目到本地（初学者）

如果你不熟悉 Git 或命令行操作，可以按照以下步骤将本项目下载到你的电脑：

### Step 0: 安装 Git（若尚未安装）

- **Windows 用户**：访问 [Git 官网](https://git-scm.com/download/win)，下载安装包并默认安装即可；
- **Mac 用户**：打开终端，输入 `git` 后按提示安装；
- **Linux 用户**：在终端中运行如下命令（以 Ubuntu 为例）：

```bash
sudo apt update
sudo apt install git
```

### Step 1: 打开命令行或终端，选择你想保存项目的位置

例如：将项目放在“下载”目录：

```bash
cd ~/Downloads
```

### Step 2: 克隆本项目

```bash
git clone https://github.com/diverHansun/facial-emotion-analyzer.git
```

执行后将自动下载整个项目文件夹，名为 `facial-emotion-analyzer`，你可以进入该目录继续后续操作。

---

## 📦 安装依赖

请使用 Python 3.8+ 环境，建议创建虚拟环境后安装：

```bash
pip install -r requirements.txt
```

---

## 🚀 使用方法

### Step 1: 打开命令行，进入项目主目录：

**Windows：**
```bash
cd "C:\Users\你的用户名\Downloads\facial-emotion-analyzer"
```

**Mac/Linux：**
```bash
cd ~/Downloads/facial-emotion-analyzer
```

### Step 2: 在项目主文件夹facial-analysis下运行主程序，传入视频路径和参数

```bash
python -m scripts.emotion_analysis.main videos/xxx.mp4 --process_sampling_rate x --fps x
```

## ⚙️ 命令行参数说明

### ✅ 必填参数：

| 参数 | 说明 |
|------|------|
| `video_path` | 要分析的视频路径，如 `videos/test5.mp4` |
| `--process_sampling_rate` | 每隔多少帧进行一次检测（默认 5） |
| `--fps` | 视频帧率，用于计算秒数轴（默认 30） |

### ✅ 可选参数（默认值可省略）：

| 参数           | 默认值                                    | 说明                                          |
|--------------|----------------------------------------|---------------------------------------------|
| `--multi_face` | False                                  | 是否需要分析视频内的多张人脸，若（默认）False则每检测帧仅分析检测到的最显著一张人脸 |
| `--start_frame` | None                                   | 指定分析起始帧（自动对齐最近采样帧）                          |
| `--end_frame` | None                                   | 指定分析结束帧                                     |
| `--output_csv` | outputs/facial_expression_analysis.csv | 分析结果的 CSV 路径                                |
| `--output_pdf` | outputs/emotion_report.pdf             | 最终生成的 PDF 报告路径                              |
| `--method`   | tsne                                   | 降维方法（可选 `tsne` 或 `umap`）                    |
| `--cluster_sampling_rate` | 自动（无传参时）或5（命令行设定默认）                    | 聚类图设置聚类降维采样间隔                               |
| `--perplexity` | 自动（无传参时）或30（命令行默认）                     | 聚类图选择t-SNE降维方法的超参数                          |
| `--n_neighbors` | 自动（无传参时）或15（命令行默认）                     | 聚类图选择UMAP降维方法的邻居数（可选，默认自动）                  |

---

##  输出内容

运行后将在 `outputs/` 文件夹中生成：

- `facial_expression_analysis.csv`：包含采样到的每帧的情绪分析结果；
- `emotion_dynamic_face.html`：可交互式折线图；
- `emotion_report.pdf`：包含以下图表的综合情绪分析报告：

| 图表类型 | 内容 |
|----------|------|
| 📈 折线图 | 情绪强度随帧变化趋势 |
| 🥧 饼状图 | 主导情绪比例（指定帧区间） |
| 📊 柱状图 | 主导情绪出现频数 |
| 🔥 热力图 | 情绪强度二维分布图 |
| 🕸️ 雷达图 | 情绪平均强度雷达图 |
| 🧠 聚类图 | 降维聚类结果（t-SNE / UMAP） |

---

## 📌 注意事项

- 默认每检测帧仅分析检测到的最显著一张人脸。如需分析每帧中所有人脸，请添加 `--multi_face` 参数，输出中将包含 `face_id` 字段，图表与 PDF 报告也将逐人脸展示；
- 视频建议长度为 **2-5 分钟**，过短不具分析价值，过长处理耗时较大；
- 请确保传入的 `--fps`（帧率）与视频实际帧率一致，否则图表时间轴会偏差,命令行默认值为30帧；
- 默认人脸采样间隔为 **每 10 帧处理一次**，可根据视频长度和帧率灵活调整 `--process_sampling_rate`；
- 可交互式动态折线图（`emotion_dynamic.html`）不会出现在 PDF 报告中，生成后将自动在浏览器打开，供用户交互查看，也可在outputs文件夹中找到；
- 报告中使用中文字体标题，需提供 `simhei.ttf` 字体文件并放置于项目根目录，若系统已安装 SimHei 字体，或不在意中文标题显示效果，可忽略此要求。
- 所有中间临时图片和图表自动清除，无需手动删除；
- 项目支持命令行参数自定义输出路径和处理参数，适合批量处理和集成脚本使用；

---

## 📚 引用项目

项目参考并基于：
- [py-feat](https://github.com/cosanlab/py-feat)
- [deepface](https://github.com/serengil/deepface)
- [matplotlib / seaborn / reportlab / sklearn 等]

---

## 📮 联系与反馈

如在使用过程中遇到问题或有功能建议，欢迎提交 [Issue](https://github.com/diverHansun/facial-emotion-analyzer/issues)。

---

✅ **Enjoy your AI-powered facial expression analyzer!**
