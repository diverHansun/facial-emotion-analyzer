import logging
import matplotlib.pyplot as plt
import numpy as np

def plot_emotion_radar(df, fps, start_frame=None, end_frame=None, save_path=None):

    # 若传入的 fps 无效，则使用默认值30
    if not fps or fps <= 0:
        logging.warning("fps 参数，使用默认值 30。")
        fps = 30


    """
    绘制指定帧范围内的情绪强度雷达图（平均值）

    参数:
    - df: 包含情绪数据的 DataFrame，必须含有 "frame" 列和情绪列
    - fps: 视频帧率
    - start_frame: 起始帧（可选，默认全范围）
    - end_frame: 结束帧（可选，默认全范围）
    - save_path: 如指定则保存图像，否则直接展示
    """

    # 情绪类别与颜色定义（与饼图保持一致）
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
        logging.warning("数据中未包含情绪列，无法绘制雷达图。")
        return

    # 获取所有检测帧
    valid_frames = df["frame"].sort_values().unique()

    # ✅ 安全兜底：默认使用整个范围
    if start_frame is None:
        start_frame = valid_frames[0]
    else:
        valid_start = valid_frames[valid_frames >= start_frame]
        if valid_start.size == 0:
            logging.warning(f"起始帧 {start_frame} 后无有效数据，跳过绘图。")
            return
        start_frame = valid_start[0]

    if end_frame is None:
        end_frame = valid_frames[-1]
    else:
        valid_end = valid_frames[valid_frames <= end_frame]
        if valid_end.size == 0:
            logging.warning(f"终止帧 {end_frame} 前无有效数据，跳过绘图。")
            return
        end_frame = valid_end[-1]

    df_range = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
    if df_range.empty:
        logging.warning(f"帧区间 [{start_frame}, {end_frame}] 内无检测数据。")
        return

    # 计算情绪平均值
    avg_emotions = df_range[available_emotions].mean()
    labels = list(avg_emotions.index)
    values = avg_emotions.values.tolist()
    values += values[:1]  # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # 颜色顺序匹配
    colors = [emotion_colors.get(label, "black") for label in labels]

    # 绘图
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='black')
    ax.fill(angles, values, color='lightblue', alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels([])

    # 每个角标注颜色小圆点
    for angle, label, color in zip(angles[:-1], labels, colors):
        ax.text(angle, max(values) * 1.05, "●", ha='center', va='center', fontsize=18, color=color)

    # 标题
    title = f"帧范围 [{start_frame}, {end_frame}]"
    if fps:
        start_time = round(start_frame / fps, 2)
        end_time = round(end_frame / fps, 2)
        title += f"（{start_time}s - {end_time}s）"
    plt.title(f"情绪平均强度雷达图\n{title}", fontsize=14)

    # 保存或展示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"✅ 雷达图已保存至 {save_path}")
        plt.close()
    else:
        plt.show()
