# 文件：plot_emotion_bar.py
import logging
import matplotlib.pyplot as plt

def plot_emotion_bar(df, start_frame=None, end_frame=None, save_path=None):
    """
    绘制指定帧范围内主导情绪占比的柱状图，可保存为图片或展示。
    """

    emotions = ["anger", "happiness", "sadness", "surprise", "fear", "disgust", "neutral"]
    emotion_colors = {
        "anger": "red",
        "happiness": "gold",
        "sadness": "blue",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "green",
        "neutral": "gray"
    }

    available_emotions = [e for e in emotions if e in df.columns]
    if not available_emotions:
        logging.warning("未在数据中找到情绪列，无法绘制柱状图。")
        return

    # 获取有效帧并自动对齐帧范围
    valid_frames = df["frame"].sort_values().unique()
    if start_frame is None:
        start_frame = valid_frames[0]
    if end_frame is None:
        end_frame = valid_frames[-1]

    valid_starts = valid_frames[valid_frames >= start_frame]
    if valid_starts.size == 0:
        logging.warning(f"起始帧 {start_frame} 之后无数据，跳过柱状图。")
        return
    start_frame = valid_starts[0]

    valid_ends = valid_frames[valid_frames <= end_frame]
    if valid_ends.size == 0:
        logging.warning(f"终止帧 {end_frame} 之前无数据，跳过柱状图。")
        return
    end_frame = valid_ends[-1]

    # 筛选帧范围内的数据
    df_range = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
    if df_range.empty:
        logging.warning(f"帧区间 [{start_frame}, {end_frame}] 内无数据。")
        return

    df_range["dominant_emotion"] = df_range[available_emotions].idxmax(axis=1)
    emotion_counts = df_range["dominant_emotion"].value_counts()

    # 开始绘制柱状图
    colors = [emotion_colors.get(e, "black") for e in emotion_counts.index]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{int(height)}", ha='center', fontsize=11)

    plt.title("指定帧范围内主导情绪频数柱状图", fontsize=14)
    plt.ylabel("出现次数")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"✅ 情绪柱状图已保存至 {save_path}")
        plt.close()
    else:
        plt.show()
