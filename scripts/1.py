#!/usr/bin/env python3
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
import sys

# 关键：确保安装/升级到最新 py-feat
# pip install --upgrade py-feat
from feat import Detector

def process_video(video_path, sampling_rate, output_csv):
    """
    读取视频，按照采样率处理每一帧，利用 Py-Feat 检测面部表情，
    将检测结果存储到 DataFrame 中，并保存为 CSV 文件。
    """

    logging.info("初始化检测器...")
    # 初始化 Py-Feat 检测器
    detector = Detector()

    logging.info(f"正在打开视频文件：{video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("无法打开视频，请检查文件路径或格式是否正确。")
        sys.exit(1)

    frame_count = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取结束

        frame_count += 1

        # 只在指定采样率的帧上进行分析
        if frame_count % sampling_rate == 0:
            try:
                # 将 OpenCV 的 BGR 格式转换为 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 关键：将单个 np.ndarray 放入列表中，传给 detect_image
                # 这样 py-feat 内部不会错误地把它当作文件对象去调用 read()
                features = detector.detect_image([frame_rgb])

                # 如果返回的是非空 DataFrame，说明检测到了人脸及情绪
                if isinstance(features, pd.DataFrame) and not features.empty:
                    # 记录当前帧号
                    features["frame"] = frame_count
                    results.append(features)
                    logging.info(f"成功处理帧：{frame_count}")
                else:
                    logging.warning(f"帧 {frame_count} 未检测到人脸。")

            except Exception as e:
                logging.error(f"处理帧 {frame_count} 时出错：{e}")

    cap.release()
    logging.info("视频处理完成。")

    if not results:
        logging.error("没有获得任何检测结果，请确认视频内容是否包含人脸。")
        sys.exit(1)

    # 合并所有帧的检测结果
    df = pd.concat(results, ignore_index=True)
    df.to_csv(output_csv, index=False)
    logging.info(f"检测结果已保存到：{output_csv}")
    return df

def plot_emotions(df):
    """
    根据 DataFrame 中的数据绘制情绪随时间变化的趋势图，
    识别可能存在的情绪列（anger, happiness, sadness, surprise, fear, disgust, neutral）。
    """
    possible_emotions = ["anger", "happiness", "sadness", "surprise", "fear", "disgust", "neutral"]
    available_emotions = [emo for emo in possible_emotions if emo in df.columns]

    if not available_emotions:
        logging.warning("未在检测结果中找到预期的情绪列，无法绘图。")
        return

    df.plot(x="frame", y=available_emotions, kind="line", figsize=(12, 6))
    plt.title("面部表情情绪随帧数变化趋势")
    plt.xlabel("帧数")
    plt.ylabel("情绪强度")
    plt.legend(available_emotions)
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="基于 Py-Feat 的视频面部表情分析工具"
    )
    parser.add_argument("video_path", help="待分析视频文件的路径")
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=10,
        help="每隔多少帧进行一次情绪检测（默认 10 帧）"
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/facial_expression_analysis.csv",
        help="检测结果保存的 CSV 文件名称（默认 facial_expression_analysis.csv）"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    df = process_video(args.video_path, args.sampling_rate, args.output_csv)
    logging.info("检测结果预览：")
    print(df.head())

    plot_emotions(df)

if __name__ == "__main__":
    main()
