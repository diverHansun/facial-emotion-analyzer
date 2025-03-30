import logging
import numpy as np
import matplotlib.pyplot as plt

def plot_emotion_line(df, fps, max_points=5400, save_path=None):
    # 若传入的 fps 无效，则使用默认值30
    if not fps or fps <= 0:
        logging.warning("fps 参数，使用默认值 30。")
        fps = 30

    """
    绘制情绪随时间变化的趋势图，支持长视频下采样，并自定义颜色。

    参数：
    - df: DataFrame，包含 "frame"、"face_id" 列和多个情绪列
    - max_points: 最大绘图点数，用于长视频数据下采样
    - save_path: 保存路径，若为 None 则不保存图像
    """

    emotions = [
        "anger", "happiness", "sadness", "surprise", "fear", "disgust", "neutral"
    ]
    emotion_colors = {
        "anger": "red",
        "happiness": "gold",
        "sadness": "blue",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "green",
        "neutral": "gray",
    }

    available_emotions = [e for e in emotions if e in df.columns]
    if not available_emotions:
        logging.warning("未在数据中找到情绪列，无法绘图。")
        return

    df = df.ffill().bfill()

    # 设置 Matplotlib 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 如果包含 face_id，则分别绘图
    if "face_id" in df.columns:
        face_ids = df["face_id"].unique()
    else:
        face_ids = [None]

    for fid in face_ids:
        if fid is not None:
            sub_df = df[df["face_id"] == fid].copy()
            fid_str = str(int(fid))
            title_suffix = f" - Face ID {fid_str}"
        else:
            sub_df = df.copy()
            fid_str = ""
            title_suffix = ""


        if len(sub_df) > max_points:
            ds_rate = len(sub_df) // max_points
            sub_df = sub_df.iloc[::ds_rate].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(15, 6))

        for emotion in available_emotions:
            color = emotion_colors.get(emotion, None)
            ax.plot(sub_df["frame"], sub_df[emotion], label=emotion, linewidth=1.5, color=color)

        x_max = sub_df["frame"].max()
        ax.set_xlim(0, x_max * 1.2)
        tick_step = max(1, x_max // 20)
        ax.set_xticks(np.arange(0, x_max, step=tick_step))
        plt.setp(ax.get_xticklabels(), rotation=45)

        secax = ax.secondary_xaxis('top', functions=(lambda x: x / fps, lambda x: x * fps))
        secax.set_xlabel("秒", fontsize=12)

        ax.set_title(f"面部表情情绪随帧数变化趋势{title_suffix}", fontsize=14)
        ax.set_xlabel("帧数", fontsize=12)
        ax.set_ylabel("情绪强度", fontsize=12)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, linestyle="--", alpha=0.7)

        try:
            if save_path:
                face_specific_path = save_path.replace(".png", f"_face{fid_str}.png") if fid is not None else save_path
                plt.savefig(face_specific_path, bbox_inches='tight')
                logging.info(f"✅ 情绪折线图已保存至：{face_specific_path}")
                plt.close()
        except Exception as e:
            logging.error(f"❌ 图像保存失败: {e}")
