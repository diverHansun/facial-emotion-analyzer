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
    font_path = "simhei.ttf"  # 确保这个文件在项目根目录或指定路径
    try:
        pdfmetrics.registerFont(TTFont('SimHei', font_path))
        logging.info("中文字体 SimHei 注册成功")
    except Exception as e:
        logging.error(f"中文字体注册失败: {e}")



register_chinese_font()

def generate_report(df,args,output_path="outputs/emotion_report.pdf"):
    """
    生成包含多种情绪分析图的 PDF 报告
    :param df: 表情分析 DataFrame，必须包含 frame 与情绪列
    :param output_path: 输出 PDF 文件名，默认 "emotion_report.pdf"
    """
    logging.info("开始生成情绪分析报告 PDF...")

    # 自动判断 fps（如果有 second 列）
    if "second" in df.columns:
        total_frame = df["frame"].iloc[-1] - df["frame"].iloc[0]
        total_time = df["second"].iloc[-1] - df["second"].iloc[0]
        fps = round(total_frame / total_time, 2) if total_time > 0 else 30
    else:
        fps = 30

    # 创建临时目录保存图片
    temp_dir = "temp_report_images"
    os.makedirs(temp_dir, exist_ok=True)

    # 图片文件路径定义
    emotion_line_path = os.path.join(temp_dir, "emotion_line.png")
    emotion_pie_path = os.path.join(temp_dir, "emotion_pie.png")
    emotion_bar_path = os.path.join(temp_dir, "emotion_bar.png")
    emotion_heatmap_path = os.path.join(temp_dir, "emotion_heatmap.png")
    emotion_radar_path = os.path.join(temp_dir, "emotion_radar.png")
    emotion_cluster_path = os.path.join(temp_dir, "emotion_clusters.png")


    # --- 保存图片版本的绘图 ---
    args = parse_arguments()

    plot_emotion_line(
        df=df,
        fps=fps,
        save_path=emotion_line_path
    )

    plot_emotion_pie(
        df=df,
        save_path=emotion_pie_path
    )
    plot_emotion_bar(
        df=df,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        save_path=emotion_bar_path
    )
    plot_emotion_heatmap(
        df=df,
        fps=fps,
        save_path=emotion_heatmap_path
    )

    plot_emotion_radar(
        df=df,
        fps=fps,
        save_path=emotion_radar_path
    )

    plot_emotion_clusters(
        df=df,
        fps=fps,
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        cluster_sampling_rate=args.cluster_sampling_rate,
        save_path=emotion_cluster_path
    )

    # 创建 PDF 对象
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
        c.drawImage(img, 50, 100, width=width - 50, height=height - 50, preserveAspectRatio=True, mask='auto')
        c.showPage()

    def safe_draw_image(image_path, title):
        if os.path.exists(image_path):
            draw_image_page(image_path, title)
        else:
            logging.warning(f"❌ 报告中缺失图像: {title} -> {image_path}")


    # 添加封面和图像页
    draw_title_page()
    safe_draw_image(emotion_line_path, "情绪趋势折线图")
    safe_draw_image(emotion_pie_path, "主导情绪饼图")
    safe_draw_image(emotion_bar_path, "主导情绪柱状图")
    safe_draw_image(emotion_heatmap_path, "情绪强度热力图")
    safe_draw_image(emotion_radar_path,"情绪雷达图")
    safe_draw_image(emotion_cluster_path,"情绪空间分布聚类图")


    # 保存 PDF
    c.save()
    print("✅ PDF 报告生成完毕！")
    logging.info(f"PDF 报告已保存至：{output_path}")

    # 删除临时图片目录
    shutil.rmtree(temp_dir)


