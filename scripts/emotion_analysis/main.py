import logging
from .process_video import process_video
from .plot_emotion_line import plot_emotion_line
from .plot_emotion_pie import plot_emotion_pie
from .plot_emotion_heatmap import plot_emotion_heatmap
from .plot_emotion_dynamic import plot_emotion_dynamic
from .plot_emotion_radar import plot_emotion_radar
from .plot_emotion_clusters import plot_emotion_clusters
from .parse_arguments import parse_arguments
from .generate_report import generate_report

def main():
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    df = process_video(
        video_path=args.video_path,
        process_sampling_rate=args.process_sampling_rate,
        output_csv=args.output_csv,
        multi_face=args.multi_face  # 支持多张人脸
    )

    logging.info("检测结果预览：")
    print(df.head())

    # 绘制情绪折线图
    plot_emotion_line(df=df, fps=args.fps)

    # 绘制指定帧范围内情绪占比饼图
    plot_emotion_pie(df=df, start_frame=args.start_frame, end_frame=args.end_frame)

    # 绘制情绪热力图（横轴显示帧数及秒数）
    plot_emotion_heatmap(df=df, fps=args.fps)

    # 绘制情绪雷达图
    plot_emotion_radar(df=df, fps=args.fps, start_frame=args.start_frame, end_frame=args.end_frame)

    # 绘制可交互折线图
    plot_emotion_dynamic(df=df, fps=args.fps, save_path="outputs/emotion_dynamic.html")

    # 绘制情绪聚类图
    plot_emotion_clusters(
        df=df,
        fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        method=args.method,
        n_neighbors=args.n_neighbors,
        perplexity=args.perplexity,
        cluster_sampling_rate=args.cluster_sampling_rate
    )

    # 生成报告
    generate_report(df=df, args=args)
    print("\n🎉 分析完成，图表已展示，报告已生成。程序退出。\n")

if __name__ == "__main__":
    main()

# 命令行进入 facial-analysis 文件夹 (修改为你的路径)
# cd "D:\basic software\pycharm\code\pythonProject1\facial-analysis"
# 命令行参数传入:
#单个人脸
# python -m scripts.emotion_analysis.main videos/name.mp4 --process_sampling_rate 10 --fps 30
#多张人脸
# python -m scripts.emotion_analysis.main videos/test6.mp4 --process_sampling_rate 12 --fps 24 --multi_face