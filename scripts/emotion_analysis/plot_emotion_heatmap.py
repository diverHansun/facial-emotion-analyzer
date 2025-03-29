import logging
import numpy as np
import matplotlib.pyplot as plt



def plot_emotion_heatmap(df,fps,save_path= None):



    # 若传入的 fps 无效，则使用默认值30
    if not fps or fps <= 0:
        logging.warning("fps 参数，使用默认值 30。")
        fps = 30
    """
    绘制情绪强度热力图，展示不同情绪随时间帧的强度分布，
    同时横轴显示帧数及对应的秒数（帧/30），
    其中每种情绪的标签采用预定义颜色。
    """
    # 定义情绪及对应颜色（参照原代码）
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

    # 填充缺失值
    df = df.ffill().bfill()

    # 提取各情绪数据，并保持顺序（每行对应一种情绪）
    heatmap_data = df[available_emotions].to_numpy().T

    # 使用实际帧号作为横坐标
    frame_min = df["frame"].min()
    frame_max = df["frame"].max()
    extent = [frame_min, frame_max, 0, len(available_emotions)]  # [x_min, x_max, y_min, y_max]

    # 设置中文显示及负号正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", extent=extent, origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("情绪强度")

    # 设置 y 轴：每行在中间位置显示情绪名称，并用对应颜色标记
    ax.set_yticks(np.arange(0.5, len(available_emotions), 1))
    ax.set_yticklabels(available_emotions, fontsize=10)
    for tick_label in ax.get_yticklabels():
        emotion = tick_label.get_text()
        tick_label.set_color(emotion_colors.get(emotion, "black"))

    # 设置 x 轴刻度（帧数）
    num_ticks = 10
    xticks = np.linspace(frame_min, frame_max, num_ticks, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xlabel("帧数", fontsize=12)
    ax.set_title("情绪强度热力图", fontsize=14)

    # 添加次要 x 轴：秒数（转换公式：秒 = 帧数 / 30）
    secax = ax.secondary_xaxis('top', functions=(lambda x: x / fps, lambda x: x * fps))
    secax.set_xlabel("秒", fontsize=12)

    plt.tight_layout()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"✅ 热力图已保存至 {save_path}")
        plt.close()
    else:
        plt.show()
