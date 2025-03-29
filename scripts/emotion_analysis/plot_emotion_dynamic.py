import logging
import numpy as np
import plotly.graph_objects as go



def plot_emotion_dynamic(df, fps,save_path=None):
    """
    绘制情绪随时间变化图，秒数为主要横轴，帧数为次要横轴显示。

    参数：
      df  : 包含视频帧和情绪数据的 DataFrame，必须包含 'frame' 列；
            若存在 'second' 列则直接使用，否则按 fps 计算秒数。
      fps : 帧率；若无效（None、非数字或非正数）则使用默认值 30。
    """
    # 检查 fps 参数
    if fps is None or not isinstance(fps, (int, float)) or fps <= 0:
        logging.warning("无效的 fps 参数，使用默认值 30")
        fps = 30

    # 检查是否包含 "frame" 列
    if 'frame' not in df.columns:
        raise ValueError("输入的 DataFrame 必须包含 'frame' 列")

    # 如果不存在 "second" 列，则根据 fps 计算秒数（避免修改原始 df，先复制一份）
    if 'second' not in df.columns:
        df = df.copy()
        df['second'] = df['frame'] / fps

    # 定义情绪与对应颜色映射
    emotion_colors = {
        "happiness": "gold",
        "anger": "red",
        "sadness": "blue",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "green",
        "neutral": "gray"
    }

    # 创建图形对象
    fig = go.Figure()

    # 遍历所有预定义情绪，若数据中存在该情绪，则添加对应曲线
    for emotion, color in emotion_colors.items():
        if emotion in df.columns:
            # 将帧数作为自定义数据，便于在悬停时显示
            customdata = np.stack((df["frame"],), axis=-1)
            fig.add_trace(go.Scatter(
                x=df["second"],   # 使用秒数作为主要横轴
                y=df[emotion],
                mode='lines',
                name=emotion,
                line=dict(color=color, width=2),
                customdata=customdata,
                hovertemplate=(
                    f"情绪: {emotion}<br>" +
                    "秒: %{x:.2f}<br>" +
                    "帧: %{customdata[0]}<br>" +
                    "强度: %{y:.2f}<extra></extra>"
                )
            ))

    # 计算主 x 轴（秒数）的刻度值及对应的帧数刻度
    second_min = df["second"].min()
    second_max = df["second"].max()
    num_ticks = 10
    sec_ticks = np.linspace(second_min, second_max, num_ticks)
    # 根据 fps 将秒转换为帧
    frame_ticks = sec_ticks * fps

    # 更新图形布局：主 x 轴显示秒数，次级 x 轴显示帧数
    fig.update_layout(
        title="情绪随时间（秒 & 帧）变化",
        xaxis=dict(
            title='秒',
            tickvals=sec_ticks,
            ticktext=[f"{s:.2f}" for s in sec_ticks],
            rangeslider=dict(visible=True),
            type='linear',
            showgrid=True,
            zeroline=True,
            zerolinecolor='LightPink'
        ),
        xaxis2=dict(
            title='帧',
            tickvals=sec_ticks,
            ticktext=[str(int(f)) for f in frame_ticks],
            overlaying='x',
            side='top',
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor='black',
            ticks='outside'
        ),
        yaxis=dict(
            title='情绪强度',
            showgrid=True,
            zeroline=True,
            zerolinecolor='LightPink'
        ),
        legend=dict(
            title="情绪",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2
        ),
        hovermode="x unified",
        template='plotly_white'
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs='cdn')
        logging.info(f"动态情绪折线图已保存为 HTML：{save_path}")
    else:
        fig.show()






