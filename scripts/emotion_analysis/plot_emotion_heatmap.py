import logging
import numpy as np
import matplotlib.pyplot as plt

def plot_emotion_heatmap(df, fps, save_path=None):
    # 若传入的 fps 无效，则使用默认值30
    if not fps or fps <= 0:
        logging.warning("fps 参数，使用默认值 30。")
        fps = 30

    """
    绘制情绪强度热力图，展示不同情绪随时间帧的强度分布，支持多张人脸自动分图。

    参数：
    - df: DataFrame，包含 "frame" 和情绪列（可选含 "face_id"）
    - fps: 视频帧率
    - save_path: 图片保存路径，若为 None 则直接展示
    """

    emotion_colors = {
        "anger": "red",
        "happiness": "gold",
        "sadness": "blue",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "green",
        "neutral": "gray"
    }
    emotions = list(emotion_colors.keys())
    available_emotions = [e for e in emotions if e in df.columns]
    if not available_emotions:
        logging.warning("未找到情绪列，跳过热力图绘制。")
        return

    df = df.ffill().bfill()

    face_ids = df["face_id"].unique() if "face_id" in df.columns else [None]

    for fid in face_ids:
        if fid is not None:
            df_sub = df[df["face_id"] == fid].copy()
            fid_str = str(int(fid))
            title_suffix = f" - Face ID {fid_str}"
        else:
            df_sub = df.copy()
            fid_str = ""
            title_suffix = ""

        if df_sub.empty:
            continue

        heatmap_data = df_sub[available_emotions].to_numpy().T
        frame_min = df_sub["frame"].min()
        frame_max = df_sub["frame"].max()
        extent = [frame_min, frame_max, 0, len(available_emotions)]

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(15, 5))
        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest", extent=extent, origin='lower')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("情绪强度")

        ax.set_yticks(np.arange(0.5, len(available_emotions), 1))
        ax.set_yticklabels(available_emotions, fontsize=10)
        for tick_label in ax.get_yticklabels():
            emotion = tick_label.get_text()
            tick_label.set_color(emotion_colors.get(emotion, "black"))

        num_ticks = 10
        xticks = np.linspace(frame_min, frame_max, num_ticks, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xlabel("帧数", fontsize=12)
        ax.set_title(f"情绪强度热力图{title_suffix}", fontsize=14)

        secax = ax.secondary_xaxis('top', functions=(lambda x: x / fps, lambda x: x * fps))
        secax.set_xlabel("秒", fontsize=12)

        plt.tight_layout()
        if save_path:
            specific_path = save_path.replace(".png", f"_face{fid_str}.png") if fid is not None else save_path
            plt.savefig(specific_path, bbox_inches='tight')
            logging.info(f"✅ 热力图已保存至 {specific_path}")
            plt.close()
        else:
            plt.show()