import os
import shutil
import logging
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .plot_emotion_line import plot_emotion_line
from .plot_emotion_pie import plot_emotion_pie
from .plot_emotion_bar import plot_emotion_bar
from .plot_emotion_heatmap import plot_emotion_heatmap
from .plot_emotion_radar import plot_emotion_radar
from .plot_emotion_clusters import plot_emotion_clusters
from .parse_arguments import parse_arguments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def register_chinese_font():
    font_path = "simhei.ttf"
    try:
        pdfmetrics.registerFont(TTFont('SimHei', font_path))
        logging.info("中文字体 SimHei 注册成功")
    except Exception as e:
        logging.error(f"中文字体注册失败: {e}")

register_chinese_font()

def generate_report(df, args, output_path="outputs/emotion_report.pdf"):
    logging.info("开始生成情绪分析报告 PDF...")

    if "second" in df.columns:
        total_frame = df["frame"].iloc[-1] - df["frame"].iloc[0]
        total_time = df["second"].iloc[-1] - df["second"].iloc[0]
        fps = round(total_frame / total_time, 2) if total_time > 0 else 30
    else:
        fps = 30

    temp_dir = "temp_report_images"
    os.makedirs(temp_dir, exist_ok=True)

    plot_emotion_line(df=df, fps=fps, save_path=os.path.join(temp_dir, "emotion_line.png"))
    plot_emotion_pie(df=df, start_frame=args.start_frame, end_frame=args.end_frame, save_path=os.path.join(temp_dir, "emotion_pie.png"))
    plot_emotion_bar(df=df, start_frame=args.start_frame, end_frame=args.end_frame, save_path=os.path.join(temp_dir, "emotion_bar.png"))
    plot_emotion_heatmap(df=df, fps=fps, save_path=os.path.join(temp_dir, "emotion_heatmap.png"))
    plot_emotion_radar(df=df, fps=fps, start_frame=args.start_frame, end_frame=args.end_frame, save_path=os.path.join(temp_dir, "emotion_radar.png"))
    plot_emotion_clusters(df=df, fps=fps, method=args.method, perplexity=args.perplexity, n_neighbors=args.n_neighbors, cluster_sampling_rate=args.cluster_sampling_rate, start_frame=args.start_frame, end_frame=args.end_frame, save_path=os.path.join(temp_dir, "emotion_clusters.png"))

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    def draw_title_page():
        c.setFont("SimHei", 24)
        c.drawCentredString(width / 2, height - 100, "面部表情情绪分析报告")
        c.setFont("SimHei", 12)
        c.drawCentredString(width / 2, height - 140, "基于 Py-Feat 分析生成")
        c.showPage()

    def draw_image_page(image_path, title):
        c.setFont("SimHei", 16)
        c.drawCentredString(width / 2, height - 50, title)
        img = ImageReader(image_path)
        c.drawImage(img, 50, 100, width=width - 100, height=height - 150, preserveAspectRatio=True, mask='auto')
        c.showPage()

    def safe_draw_images(prefix, title):
        base_path = os.path.join(temp_dir, prefix)
        added = False
        if os.path.exists(base_path):
            draw_image_page(base_path, title)
            added = True
        else:
            # 多人脸图检查（如 emotion_pie_face0.png ...）
            i = 1
            while True:
                face_id_str = str(int(i))  # 统一为整数字符串
                path_i = base_path.replace(".png", f"_face{face_id_str}.png")
                if os.path.exists(path_i):
                    draw_image_page(path_i, f"{title} - Face {face_id_str}")
                    added = True
                    i += 1
                else:
                    break
        if not added:
            logging.warning(f"❌ 报告中缺失图像: {title} -> {base_path}*")

    draw_title_page()
    safe_draw_images("emotion_line.png", "情绪趋势折线图")
    safe_draw_images("emotion_pie.png", "主导情绪饼图")
    safe_draw_images("emotion_bar.png", "主导情绪柱状图")
    safe_draw_images("emotion_heatmap.png", "情绪强度热力图")
    safe_draw_images("emotion_radar.png", "情绪雷达图")
    safe_draw_images("emotion_clusters.png", "情绪空间分布聚类图")

    c.save()
    print("✅ PDF 报告生成完毕！")
    logging.info(f"PDF 报告已保存至：{output_path}")
    shutil.rmtree(temp_dir)