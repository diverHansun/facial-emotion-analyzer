import argparse
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于 Py-Feat 的视频面部表情分析工具（生成报告）")
    parser.add_argument("video_path", help="待分析视频文件的路径")
    parser.add_argument("--process_sampling_rate", type=int, default=10, help="每隔多少帧进行一次情绪检测（默认 10 帧）")
    parser.add_argument("--output_csv", default="outputs/facial_expression_analysis.csv", help="CSV 文件名称")
    parser.add_argument("--start_frame", type=int, default=None, help="饼图分析的起始帧")
    parser.add_argument("--end_frame", type=int, default=None, help="饼图分析的结束帧")
    parser.add_argument("--fps", type=float, default=30, help="视频帧率（用于帧与秒的转换），默认为30")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"],help="降维方法：tsne或umap（默认tsne）")
    parser.add_argument("--cluster_sampling_rate",type=int,default=5,help="用于聚类图绘制阶段")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE的perplexity参数（默认30）")
    parser.add_argument("--n_neighbors", type=int, default=None,help="UMAP 降维中使用的邻居数量，默认为自动推导")
    parser.add_argument("--output_pdf",type=str,default="outputs/emotion_report.pdf",help="输出 PDF 报告的路径（默认 outputs/emotion_report.pdf）")
    return parser.parse_args()