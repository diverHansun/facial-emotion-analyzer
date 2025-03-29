import cv2
import os
import time
import tempfile

from deepface import DeepFace
from feat import Detector
from PIL import Image



class RealTimeEmotionDetector:
    """
    实时情绪检测器：
    1. 打开摄像头实时捕捉画面
    2. 使用 DeepFace 检测主情绪
    3. 使用 Py-Feat 分析表情和 AU
    4. 在窗口中实时显示结果，按 'q' 键退出
    """

    def __init__(self, camera_index=0, width=640, height=480, skip_frames=5):
        """
        :param camera_index: 要打开的摄像头序号（默认0）
        :param width: 处理图像的宽度
        :param height: 处理图像的高度
        :param skip_frames: 每多少帧检测一次，减轻 CPU 负载
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.skip_frames = skip_frames

        # 初始化 Py-Feat 检测器（设为CPU模式）
        self.feat_detector = Detector(
            face_model="retinaface",
            landmark_model="mobilenet",
            au_model="xgb",
            emotion_model="resmasknet",
            device='cpu'
        )

    def analyze_frame(self, frame):
        """对单帧进行 DeepFace + Py-Feat 分析，并返回相关信息。"""
        # 1) DeepFace：主情绪
        try:
            deepface_res = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )
            dominant_emotion = deepface_res[0]['dominant_emotion']
        except:
            dominant_emotion = 'Unknown'

        # 2) Py-Feat：保存到临时文件后检测
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            pil_image.save(temp_path)

        facebox = None
        au_values = {}
        pyfeat_emotion = 'None'

        try:
            feat_res = self.feat_detector.detect_image([temp_path])
            if not feat_res.empty:
                # 提取 Py-Feat 表情
                if 'emotion' in feat_res.columns:
                    pyfeat_emotion = str(feat_res['emotion'].values[0])
                else:
                    pyfeat_emotion = 'None'

                # 提取 Py-Feat 的人脸框（如果有）
                if 'facebox' in feat_res.columns:
                    facebox = feat_res['facebox'].values[0]  # 形如 [x_min, y_min, w, h]

                # 提取所有 AU 列（如 AU01, AU02, AU12 等）
                au_cols = [col for col in feat_res.columns if col.startswith('AU')]
                # 生成 { "AU01": 0.2, "AU02": 0.0, ... }
                au_values = {col: float(feat_res[col].values[0]) for col in au_cols}
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return dominant_emotion, pyfeat_emotion, facebox, au_values

    def run(self):
        """
        启动摄像头并实时显示检测结果。
        按 'q' 键退出。
        """
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            return

        frame_count = 0
        # 用于计算 FPS
        prev_time = time.time()
        fps = 0.0

        # 先给一些默认值，防止前几帧还没分析时出错
        dominant_emotion = 'Unknown'
        pyfeat_emotion = 'None'
        facebox = None
        au_values = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 调整图像大小（可选，加快检测速度）
            frame_resized = cv2.resize(frame, (self.width, self.height))

            # 计算帧率
            curr_time = time.time()
            # 避免除零
            if (curr_time - prev_time) > 0:
                fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # 并不是每帧都分析，减少CPU负载
            if frame_count % self.skip_frames == 0:
                (dominant_emotion,
                 pyfeat_emotion,
                 facebox,
                 au_values) = self.analyze_frame(frame_resized)

            # 如果 Py-Feat 返回了人脸框，就在图像上画出
            if facebox is not None:
                # facebox = [x_min, y_min, w, h]
                x_min, y_min, w, h = facebox
                x_max = x_min + w
                y_max = y_min + h
                cv2.rectangle(
                    frame_resized,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0),
                    2
                )

            # 显示 DeepFace & Py-Feat 主情绪
            cv2.putText(
                frame_resized,
                f"DeepFace: {dominant_emotion}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_resized,
                f"Py-Feat: {pyfeat_emotion}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            # 显示 FPS
            cv2.putText(
                frame_resized,
                f"FPS: {fps:.2f}",
                (self.width - 120, 30),  # 右上角
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            # 显示 AU 数值
            # 这里简单示例一下，按行往下显示
            # 如果 AU 太多，可以筛选或只显示最高激活的几个
            start_y = 110
            for i, (au_name, au_val) in enumerate(au_values.items()):
                text = f"{au_name}: {au_val:.2f}"
                cv2.putText(
                    frame_resized,
                    text,
                    (20, start_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

            # 显示结果
            cv2.imshow("Real-Time Emotion Detection", frame_resized)

            frame_count += 1

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 运行示例
    detector = RealTimeEmotionDetector(
        camera_index=0,  # 默认为电脑自带摄像头
        width=640,
        height=480,
        skip_frames=5  # 每 5 帧分析一次
    )
    detector.run()

#cd "D:\basic software\pycharm\code\pythonProject1\facial-analysis\deepface"
#python realtime.py
