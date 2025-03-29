import logging
import numpy as np
import matplotlib.pyplot as plt


def plot_emotion_line(df, fps,max_points=2000,save_path=None):

    # 若传入的 fps 无效，则使用默认值30
    if not fps or fps <= 0:
        logging.warning("fps 参数，使用默认值 30。")
        fps = 30
    """
    绘制情绪随时间变化的趋势图，支持长视频下采样，并自定义颜色。

    参数：
    - df: DataFrame，包含 "frame" 列和多个情绪列
    - max_points: 最大绘图点数（默认2000），用于长视频数据下采样
    """

    # 扩展的情绪类别
    emotions = [
        "anger", "happiness", "sadness", "surprise", "fear", "disgust", "neutral"
    ]

    # 颜色映射：每种情绪对应一个固定颜色
    emotion_colors = {
        "anger": "red",
        "happiness": "gold",
        "sadness": "blue",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "green",
        "neutral": "gray",

    }

    # 过滤实际存在的情绪列
    available_emotions = [e for e in emotions if e in df.columns]
    if not available_emotions:
        logging.warning("未在数据中找到情绪列，无法绘图。")
        return

    #ffill() 是向前填充：将前一个非空值填到当前空值处
    #bfill() 是向后填充：将后一个非空值填到当前空值处（如果前面都没值）
    df = df.ffill().bfill()


    # 长视频数据下采样
    if len(df) > max_points:
        ds_rate = len(df) // max_points# 计算采样步长
        df = df.iloc[::ds_rate].reset_index(drop=True)

    # 设置 Matplotlib 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建绘图对象
    fig, ax = plt.subplots(figsize=(15, 6))

    # 绘制不同情绪的曲线，并指定颜色
    for emotion in available_emotions:
        color = emotion_colors.get(emotion, None)  # 取预定义颜色
        ax.plot(df["frame"], df[emotion], label=emotion, linewidth=1.5, color=color)

    # 设置 x 轴范围和刻度
    x_max = df["frame"].max()
    ax.set_xlim(0, x_max * 1.2)

    # 设置主 x 轴（帧数）的刻度
    tick_step = max(1, x_max // 20)
    ax.set_xticks(np.arange(0, x_max, step=tick_step))
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 添加次要 x 轴显示秒数，转换公式：秒 = 帧数 / 30
    secax = ax.secondary_xaxis('top', functions=(lambda x: x / fps, lambda x: x * fps))
    secax.set_xlabel("秒", fontsize=12)

    # 设置标题和轴标签
    ax.set_title("面部表情情绪随帧数变化趋势", fontsize=14)
    ax.set_xlabel("帧数", fontsize=12)
    ax.set_ylabel("情绪强度", fontsize=12)



    # 添加图例
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle="--", alpha=0.7)
    # 显示图像或保存
    try:
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"✅ 情绪折线图已保存至：{save_path}")
            plt.close()
    except Exception as e:
        logging.error(f"❌ 图像保存失败: {e}")



