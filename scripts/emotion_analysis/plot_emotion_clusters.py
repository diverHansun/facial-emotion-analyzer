import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .parse_arguments import parse_arguments
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def plot_emotion_clusters(df, fps, start_frame=None, end_frame=None, method=None, perplexity=None, n_neighbors=None,cluster_sampling_rate=None,save_path=None):

    # 检查 fps 参数
    if fps is None or not isinstance(fps, (int, float)) or fps <= 0:
        logging.warning("无效的 fps 参数，使用默认值 30")
        fps = 30

    """
    聚类图：将情绪多维数据降维至2D并着色显示主导情绪

    参数:
    - df: DataFrame，包含 "frame" 和情绪列
    - fps: 帧率
    - start_frame/end_frame: 可选帧范围
    - save_path: 图像保存路径
    - method: "tsne" 或 "umap"
    - perplexity: tsne 参数，自动校正以避免报错
    """
    try:
        from umap import UMAP
    except ImportError:
        UMAP = None

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
        logging.warning("数据中未包含情绪列，无法绘制聚类图。")
        return

    valid_frames = df["frame"].sort_values().unique()
    if start_frame is None:
        start_frame = valid_frames[0]
    else:
        valid_start = valid_frames[valid_frames >= start_frame]
        if valid_start.size == 0:
            logging.warning(f"起始帧 {start_frame} 后无有效数据，跳过绘图。")
            return
        start_frame = valid_start[0]

    if end_frame is None:
        end_frame = valid_frames[-1]
    else:
        valid_end = valid_frames[valid_frames <= end_frame]
        if valid_end.size == 0:
            logging.warning(f"终止帧 {end_frame} 前无有效数据，跳过绘图。")
            return
        end_frame = valid_end[-1]

    df_range = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
    if df_range.empty:
        logging.warning(f"帧区间 [{start_frame}, {end_frame}] 内无检测数据。")
        return

    # 自动或手动采样逻辑
    n_total = len(df_range)

    if cluster_sampling_rate is None:
        # 用户没传，就自动判断要不要采样（适配低配电脑）
        if n_total <= 450:
            logging.info(f"数据量较小（{n_total}条），直接绘制所有点。")
            df_sampled = df_range
        else:
            max_points = 540  # 自定义一个“最多点数”限制，适配你的机器
            interval = max(1, n_total // max_points)
            logging.info(f"数据量较大（{n_total}条），每隔 {interval} 行采样（最多绘制约 {max_points} 个点）")
            df_sampled = df_range.iloc[::interval]

    else:
        # 用户指定了采样率，按帧号筛选
        df_sampled = df_range[df_range["frame"] % cluster_sampling_rate == 0]

    if df_sampled.empty:
        logging.warning("采样后无数据，请检查采样率或帧率设置。")
        return


    X = df_sampled[available_emotions].values
    y = df_sampled[available_emotions].idxmax(axis=1)
    n_samples = X.shape[0]


    #进行聚类方法选择
    if not method or not isinstance(method, str):
        method = "umap"
    else:
        method = method.strip().lower()

    if method == "tsne":
        # 自动推导 perplexity（若未手动指定）
        logging.info(f"使用tsne进行降维")
        if perplexity is None:
            perplexity = min(30, max(5, n_samples // 3))
            logging.info(f"🔧 未传入 perplexity，自动计算为 {perplexity}")
        elif perplexity >= n_samples:
            recommended = min(30, max(5, n_samples // 3))
            logging.warning(f"❗ perplexity={perplexity} ≥ 样本数={n_samples}，已自动调整为 {recommended}")
            perplexity = recommended

         # 最终检查（极端情况：样本数过小）
        if n_samples <= 5:
            logging.warning(f"⚠️ 样本数过小（n={n_samples}），跳过降维可视化")
            return

        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

    elif method == "umap" and UMAP is not None:
        logging.info(f"使用umap进行降维")

        # 自动推导 n_neighbors（如果未指定）
        if n_neighbors is None:
            n_neighbors = min(15, max(2, n_samples - 1))  # 合理范围
            logging.info(f"🔧 未传入 n_neighbors，自动计算为 {n_neighbors}")
        elif n_neighbors >= n_samples:
            recommended = min(15, max(2, n_samples - 1))
            logging.warning(f"❗ n_neighbors={n_neighbors} ≥ 样本数={n_samples}，已自动调整为 {recommended}")
            n_neighbors = recommended
        # 最终检查（极端情况：样本数过小）
        if n_samples <= 5:
            logging.warning(f"⚠️ 样本数过小（n={n_samples}），跳过降维可视化")
            return

        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)

    else:
        logging.warning("无效的 method 参数或未安装 UMAP，默认使用 t-SNE")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)

    X_reduced = reducer.fit_transform(X)

    # 绘图
    plt.figure(figsize=(8, 6))
    for emotion in available_emotions:
        idx = y == emotion
        plt.scatter(
            X_reduced[idx, 0], X_reduced[idx, 1],
            label=emotion,
            color=emotion_colors.get(emotion, 'black'),
            alpha=0.6,
            s=40
        )

    title = f"帧范围 [{start_frame}, {end_frame}]"
    if fps:
        start_time = round(start_frame / fps, 2)
        end_time = round(end_frame / fps, 2)
        title += f"（{start_time}s - {end_time}s）"

    plt.title(f"情绪聚类分布图\n{title}", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc="best", fontsize=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"✅ 聚类图已保存至 {save_path}")
        plt.close()
    else:
        plt.show()


