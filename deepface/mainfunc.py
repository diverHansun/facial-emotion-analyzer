from deepface import DeepFace
import os


# ================================
# 处理模块
# ================================

def process_reference_image(ref_img_path):
    """
    处理参考图片：检查文件是否存在，并输出提示信息。
    返回参考图片的路径。
    """
    if not os.path.exists(ref_img_path):
        print(f"参考图片 {ref_img_path} 不存在！")
    else:
        print(f"处理参考图片：{ref_img_path}")
    return ref_img_path


def process_detected_image(img_path):
    """
    处理待检测图片：检查文件是否存在，并输出提示信息。
    返回待检测图片的路径。
    """
    if not os.path.exists(img_path):
        print(f"待检测图片 {img_path} 不存在！")
    else:
        print(f"处理待检测图片：{img_path}")
    return img_path


# ================================
# 分析模块
# ================================

def analyze_image(img_path):
    """
    对单张图像进行属性分析（年龄、性别、种族、情绪），
    返回分析结果字典，若失败返回 None。
    """
    try:
        analysis = DeepFace.analyze(img_path=img_path,
                                    actions=['age', 'gender', 'race', 'emotion'],
                                    enforce_detection=False)
        # 若返回结果为列表，则取第一个元素
        if isinstance(analysis, list):
            analysis = analysis[0]
        return analysis
    except Exception as e:
        print(f"图像分析失败 ({img_path}):", e)
        return None


def verify_identity(ref_img_path, img_path):
    """
    对比参考图片和待检测图片，判断是否为同一人，
    返回 True 表示验证通过，否则返回 False。
    """
    try:
        verification = DeepFace.verify(ref_img_path, img_path, enforce_detection=False)
        return verification.get("verified", False)
    except Exception as e:
        print(f"身份验证失败 ({img_path}):", e)
        return False

def rel_path(*paths):
    """
    以 mainfunc.py 为基准，拼出相对路径的绝对路径
    使用: rel_path("pic", "amma.jpg")
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, *paths)
# ================================
# 主流程
# ================================

def main():
    # 参考图片（已知身份）和待检测图片列表
    ref_img_path = rel_path("pic", "amma.jpg")
    image_paths = [
        rel_path("pic", "musk.jpg"),
        rel_path("pic", "tatum.jpg"),
        rel_path("pic", "brown.jpg")
    ]

    # 处理参考图片
    ref_img = process_reference_image(ref_img_path)
    # 分析参考图片属性并打印
    print("正在分析参考图片的图像属性...")
    ref_analysis = analyze_image(ref_img)
    if ref_analysis:
        age = ref_analysis.get("age", "未知")
        gender = ref_analysis.get("gender", "未知")
        dominant_emotion = ref_analysis.get("dominant_emotion", "未知")
        dominant_race = ref_analysis.get("dominant_race", "未知")
        print("参考图片属性：")
        print(f"👤 年龄: {age}")
        print(f"👩‍🦰 性别: {gender}")
        print(f"😀 情绪: {dominant_emotion}")
        print(f"🌍 种族: {dominant_race}")
    else:
        print("参考图片的图像属性分析失败。")
    print("-"*40)

    # 遍历每张待检测图片
    for img_path in image_paths:
        # 处理待检测图片
        img = process_detected_image(img_path)

        # 分析图像属性
        print("正在分析待检测图像属性...")
        analysis = analyze_image(img)
        if analysis:
            age = analysis.get("age", "未知")
            gender = analysis.get("gender", "未知")
            dominant_emotion = analysis.get("dominant_emotion", "未知")
            dominant_race = analysis.get("dominant_race", "未知")

            print(f"👤 年龄: {age}")
            print(f"👩‍🦰 性别: {gender}")
            print(f"😀 情绪: {dominant_emotion}")
            print(f"🌍 种族: {dominant_race}")
        else:
            continue

        # 身份验证（参考图片 vs 当前待检测图片）
        print("🔍 正在验证身份...")
        if verify_identity(ref_img, img):
            print("✅ 身份验证结果：验证通过！")
        else:
            print("❌ 身份验证结果：验证未通过。")
        print("-" * 40)

    print(f"\n🎉 分析完成，已展示。程序退出。\n")


if __name__ == "__main__":
    main()

#analyse
#verification