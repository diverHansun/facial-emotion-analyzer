# 文件：plot_emotion_bar.py
import logging
import matplotlib.pyplot as plt

def plot_emotion_bar(df, start_frame=None, end_frame=None, save_path=None):
    """
    绘制指定帧范围内主导情绪占比的柱状图，支持多张人脸分图输出。
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

    df_range_all = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
    if df_range_all.empty:
        logging.warning(f"帧区间 [{start_frame}, {end_frame}] 内无数据。")
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
        colors = [emotion_colors.get(e, "black") for e in emotion_counts.index]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{int(height)}", ha='center', fontsize=11)

        plt.title(f"指定帧范围内主导情绪频数柱状图{title_suffix}", fontsize=14)
        plt.ylabel("出现次数")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if save_path:
            specific_path = save_path.replace(".png", f"_face{fid_str}.png") if fid is not None else save_path
            plt.savefig(specific_path, bbox_inches='tight')
            logging.info(f"✅ 情绪柱状图已保存至 {specific_path}")
            plt.close()
        else:
            plt.show()