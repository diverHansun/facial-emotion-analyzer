import logging
import numpy as np
import plotly.graph_objects as go

def plot_emotion_dynamic(df, fps, save_path=None):
    """
    绘制情绪随时间变化图，支持多张人脸数据，每张人脸生成一个 HTML 图表。

    参数：
      df  : 包含视频帧和情绪数据的 DataFrame，必须包含 'frame' 列；
            若存在 'second' 列则直接使用，否则按 fps 计算秒数。
      fps : 帧率；若无效（None、非数字或非正数）则使用默认值 30。
      save_path : 保存 HTML 路径；如包含多张人脸，将在文件名中追加 face_id。
    """
    if fps is None or not isinstance(fps, (int, float)) or fps <= 0:
        logging.warning("无效的 fps 参数，使用默认值 30")
        fps = 30

    if 'frame' not in df.columns:
        raise ValueError("输入的 DataFrame 必须包含 'frame' 列")

    if 'second' not in df.columns:
        df = df.copy()
        df['second'] = df['frame'] / fps

    emotion_colors = {
        "happiness": "gold",
        "anger": "red",
        "sadness": "blue",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "green",
        "neutral": "gray"
    }

    face_ids = df["face_id"].unique() if "face_id" in df.columns else [None]

    for fid in face_ids:
        if fid is not None:
            df_sub = df[df["face_id"] == fid].copy()
            fid_str = str(int(fid))
            suffix = f"_face{fid_str}"
        else:
            df_sub = df.copy()
            suffix = ""

        fig = go.Figure()

        for emotion, color in emotion_colors.items():
            if emotion in df_sub.columns:
                customdata = np.stack((df_sub["frame"],), axis=-1)
                fig.add_trace(go.Scatter(
                    x=df_sub["second"],
                    y=df_sub[emotion],
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

        second_min = df_sub["second"].min()
        second_max = df_sub["second"].max()
        num_ticks = 10
        sec_ticks = np.linspace(second_min, second_max, num_ticks)
        frame_ticks = sec_ticks * fps

        fig.update_layout(
            title=f"情绪随时间（秒 & 帧）变化{suffix}",
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
            final_path = save_path.replace(".html", f"{suffix}.html") if fid is not None else save_path
            fig.write_html(final_path, include_plotlyjs='cdn')
            logging.info(f"动态情绪折线图已保存为 HTML：{final_path}")
        else:
            fig.show()