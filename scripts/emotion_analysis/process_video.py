import cv2
import os
import sys
import logging
import tempfile
import pandas as pd
from feat import Detector

def process_video(video_path, process_sampling_rate, output_csv, multi_face=False):
    """
    读取视频，按照采样率处理每一帧，利用 Py-Feat 检测面部表情，
    将检测结果存储到 DataFrame 中，并保存为 CSV 文件。
    支持多张人脸分析（可选）。
    """
    logging.info("初始化检测器...")
    detector = Detector()
    
    # 输出当前使用的设备信息
    device = detector.device
    logging.info(f"当前使用的设备: {device}")

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
        if frame_count % process_sampling_rate == 0:
            try:
                # 创建一个临时文件来保存当前帧
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    temp_path = tmp_file.name

                # 将当前帧写入临时文件
                cv2.imwrite(temp_path, frame)

                if multi_face:
                    # 多人脸处理：返回多个人脸特征
                    features = detector.detect_image(temp_path, return_multiple=True)
                    if isinstance(features, pd.DataFrame) and not features.empty:
                        for i in range(len(features)):
                            features.at[i, "frame"] = frame_count
                            features.at[i, "face_id"] = i+1  # 同一帧内的人脸编号，从1开始
                        results.append(features)
                        logging.info(f"帧 {frame_count}：检测到 {len(features)} 张人脸")
                    else:
                        logging.warning(f"帧 {frame_count} 未检测到人脸。")
                else:
                    # 单人脸处理
                    features = detector.detect_image(temp_path)
                    if isinstance(features, pd.DataFrame) and not features.empty:
                        features["frame"] = frame_count
                        features["face_id"] = 1  # 默认人脸编号
                        results.append(features)
                        logging.info(f"成功处理帧：{frame_count}")
                    else:
                        logging.warning(f"帧 {frame_count} 未检测到人脸。")

                # 检测完毕后，删除临时文件
                os.remove(temp_path)

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
