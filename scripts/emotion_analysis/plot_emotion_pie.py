import logging
import matplotlib.pyplot as plt

def plot_emotion_pie(df, start_frame=None, end_frame=None, save_path=None):
    """
    绘制指定帧范围内的主导情绪占比饼状图，自动对齐至检测过的帧。
    若包含 face_id，则为每张人脸分别绘图。

    参数:
    - df: 包含情绪数据的 DataFrame，必须含有 "frame" 列和情绪列
    - start_frame: 分析起始帧（用户指定范围，可自动对齐）
    - end_frame: 分析结束帧（用户指定范围，可自动对齐）
    - save_path: 图片保存路径，若为 None 则直接显示
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
        logging.warning("未在数据中找到情绪列，无法绘制饼图。")
        return

    valid_frames = df["frame"].sort_values().unique()
    if start_frame is None:
        start_frame = valid_frames[0]
    if end_frame is None:
        end_frame = valid_frames[-1]

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

    df_range_all = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
    if df_range_all.empty:
        logging.warning(f"帧区间 [{start_frame}, {end_frame}] 内无检测数据，无法绘制饼图。")
        return

    face_ids = df_range_all["face_id"].unique() if "face_id" in df.columns else [None]

    for fid in face_ids:
        if fid is not None:
            df_range = df_range_all[df_range_all["face_id"] == fid].copy()
            fid_str = str(int(fid))
            title_suffix = f" - Face ID {fid_str}"
        else:
            df_range = df_range_all.copy()
            fid_str = ""
            title_suffix = ""

        if df_range.empty:
            continue

        df_range["dominant_emotion"] = df_range[available_emotions].idxmax(axis=1)
        emotion_counts = df_range["dominant_emotion"].value_counts()
        colors = [emotion_colors.get(emotion, "black") for emotion in emotion_counts.index]

        plt.figure(figsize=(8, 8))
        plt.pie(
            emotion_counts.values,
            labels=emotion_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 12}
        )
        plt.title(f"指定帧范围内主导情绪占比{title_suffix}", fontsize=14)
        plt.axis("equal")

        if save_path:
            specific_path = save_path.replace(".png", f"_face{fid_str}.png") if fid is not None else save_path
            plt.savefig(specific_path, bbox_inches='tight')
            logging.info(f"✅ 情绪饼状图已保存至 {specific_path}")
            plt.close()
        else:
            plt.show()
