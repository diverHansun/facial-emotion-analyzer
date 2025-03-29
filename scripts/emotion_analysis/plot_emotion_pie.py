import logging
import matplotlib.pyplot as plt
def plot_emotion_pie(df, start_frame=None, end_frame=None,save_path=None):
    """
    绘制指定帧范围内的主导情绪占比饼状图，自动对齐至检测过的帧

    参数:
    - df: 包含情绪数据的 DataFrame，必须含有 "frame" 列和情绪列
    - start_frame: 分析起始帧（用户指定范围，可自动对齐）
    - end_frame: 分析结束帧（用户指定范围，可自动对齐）
    """

    # 定义情绪类别及对应颜色
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

    # 检查是否有情绪列
    available_emotions = [e for e in emotions if e in df.columns]
    if not available_emotions:
        logging.warning("未在数据中找到情绪列，无法绘制饼图。")
        return

    # 获取所有检测过的帧
    valid_frames = df["frame"].sort_values().unique()

    # 安全兜底：如果用户未传入帧范围，则默认使用全部范围
    if start_frame is None:
        start_frame = valid_frames[0]
    if end_frame is None:
        end_frame = valid_frames[-1]

    # 自动对齐到最近的有效检测帧
    if start_frame is not None:
        valid_starts = valid_frames[valid_frames >= start_frame]
        if valid_starts.size == 0:
            logging.warning(f"起始帧 {start_frame} 之后没有检测数据，跳过绘图。")
            return
        start_frame = valid_starts[0]

    if end_frame is not None:
        valid_ends = valid_frames[valid_frames <= end_frame]
        if valid_ends.size == 0:
            logging.warning(f"终止帧 {end_frame} 之前没有检测数据，跳过绘图。")
            return
        end_frame = valid_ends[-1]

    # 筛选实际使用的帧范围
    df_range = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]

    if df_range.empty:
        logging.warning(f"帧区间 [{start_frame}, {end_frame}] 内无检测数据，无法绘制饼图。")
        return

    # 计算每一帧的主导情绪
    df_range["dominant_emotion"] = df_range[available_emotions].idxmax(axis=1)
    emotion_counts = df_range["dominant_emotion"].value_counts()
    #logging.info(f"绘图帧范围对齐为：[{start_frame}, {end_frame}]")
    #logging.info("情绪饼状图数据统计：" + str(emotion_counts.to_dict()))

    # 获取对应颜色
    colors = [emotion_colors.get(emotion, "black") for emotion in emotion_counts.index]

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        emotion_counts.values,
        labels=emotion_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12}
    )
    plt.title("指定帧范围内主导情绪占比", fontsize=14)
    plt.axis("equal")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"✅ 情绪饼状图已保存至 {save_path}")
        plt.close()
    else:
        plt.show()
