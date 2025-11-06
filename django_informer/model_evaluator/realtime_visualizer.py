import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class RealTimeVisualizer:
    """实时轨迹可视化器"""

    def __init__(self):
        self.trajectory_data = {}
        self.prediction_data = {}
        self.update_callbacks = []

    def update_trajectory(self, flight_id: str, trajectory_points: list):
        """更新轨迹数据"""
        self.trajectory_data[flight_id] = {
            'points': trajectory_points,
            'last_update': datetime.now()
        }
        self._notify_update('trajectory', flight_id, trajectory_points)

    def update_prediction(self, flight_id: str, prediction_points: list, confidence: float):
        """更新预测数据"""
        self.prediction_data[flight_id] = {
            'points': prediction_points,
            'confidence': confidence,
            'last_update': datetime.now()
        }
        self._notify_update('prediction', flight_id, {
            'points': prediction_points,
            'confidence': confidence
        })

    def generate_3d_trajectory_plot(self, flight_id: str = None) -> dict:
        """生成3D轨迹图数据"""
        try:
            fig = go.Figure()

            # 如果指定了flight_id，只显示该飞机
            flights_to_show = [flight_id] if flight_id else list(self.trajectory_data.keys())

            colors = px.colors.qualitative.Set1
            color_index = 0

            for fid in flights_to_show:
                if fid not in self.trajectory_data:
                    continue

                trajectory = self.trajectory_data[fid]['points']
                if not trajectory:
                    continue

                # 提取坐标
                lats = [point['latitude'] for point in trajectory]
                lons = [point['longitude'] for point in trajectory]
                alts = [point['geo_altitude'] for point in trajectory]
                times = [point.get('timestamp', '') for point in trajectory]

                color = colors[color_index % len(colors)]
                color_index += 1

                # 绘制历史轨迹
                fig.add_trace(go.Scatter3d(
                    x=lons,
                    y=lats,
                    z=alts,
                    mode='lines+markers',
                    name=f'{fid} 历史轨迹',
                    line=dict(color=color, width=4),
                    marker=dict(size=3),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 '经度: %{x:.4f}<br>' +
                                 '纬度: %{y:.4f}<br>' +
                                 '高度: %{z:.0f}m<extra></extra>'
                ))

                # 绘制预测轨迹（如果有）
                if fid in self.prediction_data:
                    pred_data = self.prediction_data[fid]
                    pred_points = pred_data['points']
                    confidence = pred_data['confidence']

                    if pred_points:
                        # 将预测点转换为正确格式
                        pred_lons = [point[1] for point in pred_points]  # 经度
                        pred_lats = [point[0] for point in pred_points]  # 纬度
                        pred_alts = [point[2] for point in pred_points]  # 高度

                        # 从最后一个历史点开始
                        if lats:
                            pred_lons = [lons[-1]] + pred_lons
                            pred_lats = [lats[-1]] + pred_lats
                            pred_alts = [alts[-1]] + pred_alts

                        fig.add_trace(go.Scatter3d(
                            x=pred_lons,
                            y=pred_lats,
                            z=pred_alts,
                            mode='lines+markers',
                            name=f'{fid} 预测轨迹 (置信度: {confidence:.2f})',
                            line=dict(color=color, width=3, dash='dash'),
                            marker=dict(size=4, symbol='diamond'),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                         '经度: %{x:.4f}<br>' +
                                         '纬度: %{y:.4f}<br>' +
                                         '高度: %{z:.0f}m<br>' +
                                         '置信度: ' + f'{confidence:.2f}' + '<extra></extra>'
                        ))

            # 更新布局
            fig.update_layout(
                title='实时飞机轨迹预测 - 3D视图',
                scene=dict(
                    xaxis_title='经度',
                    yaxis_title='纬度',
                    zaxis_title='高度 (m)',
                    camera=dict(
                        eye=dict(x=1.2, y=1.2, z=0.8)
                    ),
                    aspectmode='manual'
                ),
                width=1000,
                height=700,
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"生成3D轨迹图时出错: {str(e)}")
            return {'error': str(e)}

    def generate_2d_projection_plot(self, flight_id: str = None) -> dict:
        """生成2D投影图数据"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    '水平轨迹 (经纬度)',
                    '高度剖面 (纬度-高度)',
                    '高度剖面 (经度-高度)',
                    '时间序列'
                ],
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )

            flights_to_show = [flight_id] if flight_id else list(self.trajectory_data.keys())
            colors = px.colors.qualitative.Set1

            for idx, fid in enumerate(flights_to_show):
                if fid not in self.trajectory_data:
                    continue

                trajectory = self.trajectory_data[fid]['points']
                if not trajectory:
                    continue

                color = colors[idx % len(colors)]

                # 提取数据
                lats = [point['latitude'] for point in trajectory]
                lons = [point['longitude'] for point in trajectory]
                alts = [point['geo_altitude'] for point in trajectory]
                times = list(range(len(trajectory)))

                # 1. 水平轨迹
                fig.add_trace(go.Scatter(
                    x=lons, y=lats,
                    mode='lines+markers',
                    name=f'{fid} 轨迹',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'{fid}<br>经度: %{{x:.4f}}<br>纬度: %{{y:.4f}}<extra></extra>'
                ), row=1, col=1)

                # 2. 纬度-高度剖面
                fig.add_trace(go.Scatter(
                    x=lats, y=alts,
                    mode='lines+markers',
                    name=f'{fid} 剖面',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate=f'{fid}<br>纬度: %{{x:.4f}}<br>高度: %{{y:.0f}}m<extra></extra>'
                ), row=1, col=2)

                # 3. 经度-高度剖面
                fig.add_trace(go.Scatter(
                    x=lons, y=alts,
                    mode='lines+markers',
                    name=f'{fid} 剖面',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate=f'{fid}<br>经度: %{{x:.4f}}<br>高度: %{{y:.0f}}m<extra></extra>'
                ), row=2, col=1)

                # 4. 时间-高度序列
                fig.add_trace(go.Scatter(
                    x=times, y=alts,
                    mode='lines+markers',
                    name=f'{fid} 高度变化',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    showlegend=False,
                    hovertemplate=f'{fid}<br>时间步: %{{x}}<br>高度: %{{y:.0f}}m<extra></extra>'
                ), row=2, col=2)

                # 添加预测轨迹（如果有）
                if fid in self.prediction_data:
                    pred_points = self.prediction_data[fid]['points']
                    confidence = self.prediction_data[fid]['confidence']

                    if pred_points:
                        pred_lats = [point[0] for point in pred_points]
                        pred_lons = [point[1] for point in pred_points]
                        pred_alts = [point[2] for point in pred_points]
                        pred_times = list(range(len(trajectory), len(trajectory) + len(pred_points)))

                        # 预测轨迹用虚线表示
                        fig.add_trace(go.Scatter(
                            x=pred_lons, y=pred_lats,
                            mode='lines+markers',
                            name=f'{fid} 预测',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=4, symbol='diamond'),
                            showlegend=False,
                            hovertemplate=f'{fid} 预测<br>经度: %{{x:.4f}}<br>纬度: %{{y:.4f}}<br>置信度: {confidence:.2f}<extra></extra>'
                        ), row=1, col=1)

                        fig.add_trace(go.Scatter(
                            x=pred_lats, y=pred_alts,
                            mode='lines+markers',
                            name=f'{fid} 预测',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=4, symbol='diamond'),
                            showlegend=False,
                            hovertemplate=f'{fid} 预测<br>纬度: %{{x:.4f}}<br>高度: %{{y:.0f}}m<br>置信度: {confidence:.2f}<extra></extra>'
                        ), row=1, col=2)

            # 更新布局
            fig.update_xaxes(title_text="经度", row=1, col=1)
            fig.update_yaxes(title_text="纬度", row=1, col=1)
            fig.update_xaxes(title_text="纬度", row=1, col=2)
            fig.update_yaxes(title_text="高度 (m)", row=1, col=2)
            fig.update_xaxes(title_text="经度", row=2, col=1)
            fig.update_yaxes(title_text="高度 (m)", row=2, col=1)
            fig.update_xaxes(title_text="时间步", row=2, col=2)
            fig.update_yaxes(title_text="高度 (m)", row=2, col=2)

            fig.update_layout(
                title='实时飞机轨迹预测 - 2D投影',
                height=800,
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )

            return fig.to_dict()

        except Exception as e:
            logger.error(f"生成2D投影图时出错: {str(e)}")
            return {'error': str(e)}

    def generate_confidence_gauge(self, flight_id: str) -> dict:
        """生成置信度仪表盘"""
        try:
            if flight_id not in self.prediction_data:
                confidence = 0.0
            else:
                confidence = self.prediction_data[flight_id]['confidence']

            # 创建仪表盘
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"预测置信度 - {flight_id}"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})

            return fig.to_dict()

        except Exception as e:
            logger.error(f"生成置信度仪表盘时出错: {str(e)}")
            return {'error': str(e)}

    def generate_real_time_update_data(self, flight_id: str) -> dict:
        """生成实时更新数据包"""
        try:
            update_data = {
                'timestamp': datetime.now().isoformat(),
                'flight_id': flight_id,
                'trajectory': self.trajectory_data.get(flight_id, {}),
                'prediction': self.prediction_data.get(flight_id, {}),
                'plots': {
                    '3d_trajectory': self.generate_3d_trajectory_plot(flight_id),
                    '2d_projection': self.generate_2d_projection_plot(flight_id),
                    'confidence_gauge': self.generate_confidence_gauge(flight_id)
                }
            }

            return update_data

        except Exception as e:
            logger.error(f"生成实时更新数据时出错: {str(e)}")
            return {'error': str(e)}

    def add_update_callback(self, callback):
        """添加更新回调函数"""
        self.update_callbacks.append(callback)

    def _notify_update(self, update_type: str, flight_id: str, data: dict):
        """通知更新"""
        for callback in self.update_callbacks:
            try:
                callback(update_type, flight_id, data)
            except Exception as e:
                logger.error(f"执行更新回调时出错: {str(e)}")

    def clear_flight_data(self, flight_id: str):
        """清除特定飞机的数据"""
        if flight_id in self.trajectory_data:
            del self.trajectory_data[flight_id]
        if flight_id in self.prediction_data:
            del self.prediction_data[flight_id]

    def get_all_flights_status(self) -> dict:
        """获取所有飞机的状态"""
        status = {}
        for flight_id in self.trajectory_data:
            status[flight_id] = {
                'has_trajectory': True,
                'has_prediction': flight_id in self.prediction_data,
                'last_trajectory_update': self.trajectory_data[flight_id]['last_update'].isoformat(),
                'confidence': self.prediction_data.get(flight_id, {}).get('confidence', 0.0)
            }
        return status


# 全局可视化器实例
realtime_visualizer = RealTimeVisualizer()