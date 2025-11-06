import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

from django.conf import settings
from django.utils import timezone
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# Add parent directory to Python path for importing Informer modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model import Informer
from data.aircraft.data_loader import AircraftTrajectoryDataset

logger = logging.getLogger(__name__)


class RealTimePredictionEngine:
    """实时轨迹预测引擎"""

    def __init__(self):
        self.model = None
        self.device = None
        self.args = None
        self.is_initialized = False

        # 轨迹缓冲区 - 每架飞机的轨迹历史
        self.trajectory_buffers: Dict[str, deque] = {}
        self.prediction_cache: Dict[str, Dict] = {}

        # 预测参数
        self.config = settings.REALTIME_PREDICTION
        self.buffer_size = self.config['TRAJECTORY_BUFFER_SIZE']
        self.update_interval = self.config['UPDATE_INTERVAL']
        self.confidence_threshold = self.config['CONFIDENCE_THRESHOLD']

        # 通道层用于WebSocket通信
        self.channel_layer = get_channel_layer()

        # 异步任务状态
        self.active_sessions: Dict[str, bool] = {}

    async def initialize(self):
        """异步初始化引擎"""
        if not self.is_initialized:
            await self._load_model()
            self.is_initialized = True
            logger.info("实时预测引擎初始化完成")

    async def _load_model(self):
        """加载Informer模型"""
        try:
            import yaml
            from easydict import EasyDict as edict

            # 加载配置
            config_path = '../config/aircraft.yaml'
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            args = edict()
            for key, value in config.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        setattr(args, sub_key, sub_value)
                else:
                    setattr(args, key, value)

            # 配置设备
            if args.use_gpu and torch.cuda.is_available():
                args.device = torch.device(f'cuda:{args.gpu}')
            else:
                args.device = torch.device('cpu')

            self.args = args
            self.device = args.device

            # 加载模型
            model = Informer(
                args.enc_in, args.dec_in, args.c_out,
                args.seq_len, args.label_len, args.pred_len,
                factor=args.factor, d_model=args.d_model, n_heads=args.n_heads,
                e_layers=args.e_layers, d_layers=args.d_layers, d_ff=args.d_ff,
                dropout=args.dropout, attn=args.attn, embed=args.embed,
                freq=args.freq, activation=args.activation,
                output_attention=args.output_attention, distil=args.distil,
                mix=args.mix, device=args.device
            ).float().to(args.device)

            model.load_state_dict(torch.load('../checkpoints/checkpoint.pth', map_location=args.device))
            model.eval()

            self.model = model
            logger.info(f"模型加载成功，设备: {args.device}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    async def process_flight_data(self, flight_data: Dict):
        """处理新的飞行数据点"""
        flight_id = flight_data['flight_id']

        # 初始化轨迹缓冲区
        if flight_id not in self.trajectory_buffers:
            self.trajectory_buffers[flight_id] = deque(maxlen=self.buffer_size)

        # 添加新数据点到缓冲区
        self.trajectory_buffers[flight_id].append({
            'timestamp': flight_data['timestamp'],
            'latitude': flight_data['latitude'],
            'longitude': flight_data['longitude'],
            'geo_altitude': flight_data['geo_altitude'],
            'baro_altitude': flight_data['baro_altitude'],
            'speed': flight_data.get('speed', 0),
            'heading': flight_data.get('heading', 0)
        })

        # 检查是否需要生成预测
        await self._check_and_predict(flight_id)

    async def _check_and_predict(self, flight_id: str):
        """检查是否需要生成预测"""
        if flight_id not in self.trajectory_buffers:
            return

        buffer = self.trajectory_buffers[flight_id]
        if len(buffer) < self.args.seq_len:
            return  # 数据不足，无法预测

        # 检查上次预测时间
        if flight_id in self.prediction_cache:
            last_pred_time = self.prediction_cache[flight_id].get('timestamp')
            if last_pred_time:
                time_diff = (datetime.now() - last_pred_time).total_seconds()
                if time_diff < self.update_interval:
                    return  # 还未到更新时间

        # 生成新的预测
        await self._generate_prediction(flight_id)

    async def _generate_prediction(self, flight_id: str):
        """生成轨迹预测"""
        try:
            if not self.is_initialized:
                await self.initialize()

            buffer = self.trajectory_buffers[flight_id]
            if len(buffer) < self.args.seq_len:
                logger.warning(f"飞机 {flight_id} 数据不足，无法预测")
                return

            start_time = datetime.now()

            # 准备输入数据
            input_data = self._prepare_input_data(buffer)
            if input_data is None:
                return

            # 执行预测
            prediction = await self._predict_trajectory(input_data)

            # 计算置信度
            confidence = await self._calculate_confidence(flight_id, prediction)

            # 保存预测结果
            prediction_time = datetime.now()
            self.prediction_cache[flight_id] = {
                'timestamp': prediction_time,
                'prediction': prediction,
                'confidence': confidence,
                'processing_time': (prediction_time - start_time).total_seconds() * 1000
            }

            # 通过WebSocket发送更新
            await self._broadcast_prediction_update(flight_id, prediction, confidence)

            # 保存到数据库
            await self._save_prediction_to_db(flight_id, prediction, confidence)

            logger.info(f"飞机 {flight_id} 预测完成，置信度: {confidence:.3f}")

        except Exception as e:
            logger.error(f"生成飞机 {flight_id} 预测时出错: {str(e)}")

    def _prepare_input_data(self, buffer: deque) -> Optional[np.ndarray]:
        """准备模型输入数据"""
        try:
            # 获取最近的seq_len个数据点
            recent_data = list(buffer)[-self.args.seq_len:]

            # 转换为numpy数组
            features = []
            for point in recent_data:
                features.append([
                    point['latitude'],
                    point['longitude'],
                    point['geo_altitude'],
                    point['baro_altitude']
                ])

            input_array = np.array(features, dtype=np.float32)

            # 标准化（这里需要使用训练时的标准化参数）
            # 暂时使用简单的z-score标准化
            mean = np.mean(input_array, axis=0)
            std = np.std(input_array, axis=0)
            std = np.where(std == 0, 1, std)
            normalized_input = (input_array - mean) / std

            return normalized_input

        except Exception as e:
            logger.error(f"准备输入数据时出错: {str(e)}")
            return None

    async def _predict_trajectory(self, input_data: np.ndarray) -> np.ndarray:
        """执行轨迹预测"""
        try:
            # 转换为tensor
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)

            # 准备解码器输入
            batch_size, seq_len, feature_dim = input_tensor.shape
            dec_input = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, feature_dim).to(self.device)

            # 使用输入序列的最后label_len个点作为解码器起始输入
            dec_input[:, :self.args.label_len, :] = input_tensor[:, -self.args.label_len:, :]

            # 准备时间特征（简化版本）
            time_features = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 4).to(self.device)

            # 执行预测
            with torch.no_grad():
                outputs = self.model(input_tensor, time_features, dec_input, time_features)
                prediction = outputs[:, -self.args.pred_len:, :].cpu().numpy()

            return prediction[0]  # 返回单个预测结果

        except Exception as e:
            logger.error(f"执行预测时出错: {str(e)}")
            return np.array([])

    async def _calculate_confidence(self, flight_id: str, prediction: np.ndarray) -> float:
        """计算预测置信度"""
        try:
            # 基于历史预测准确性计算置信度
            if len(prediction) == 0:
                return 0.0

            # 简单的置信度计算：基于预测的稳定性
            pred_std = np.std(prediction, axis=0)
            avg_std = np.mean(pred_std)

            # 将标准差转换为置信度（标准差越小，置信度越高）
            confidence = max(0.0, min(1.0, 1.0 - avg_std / 10.0))

            return confidence

        except Exception as e:
            logger.error(f"计算置信度时出错: {str(e)}")
            return 0.5  # 默认置信度

    async def _broadcast_prediction_update(self, flight_id: str, prediction: np.ndarray, confidence: float):
        """通过WebSocket广播预测更新"""
        try:
            # 发送给单机跟踪组
            await self.channel_layer.group_send(
                f'flight_{flight_id}',
                {
                    'type': 'prediction_update',
                    'flight_id': flight_id,
                    'prediction': prediction.tolist(),
                    'confidence': confidence
                }
            )

            # 发送给全局预测更新组
            await self.channel_layer.group_send(
                'prediction_updates',
                {
                    'type': 'new_prediction_available',
                    'flight_id': flight_id,
                    'prediction': prediction.tolist()
                }
            )

        except Exception as e:
            logger.error(f"广播预测更新时出错: {str(e)}")

    async def _save_prediction_to_db(self, flight_id: str, prediction: np.ndarray, confidence: float):
        """保存预测结果到数据库"""
        try:
            from .models import Aircraft, PredictionResult

            # 获取飞机对象
            aircraft = await Aircraft.objects.aget(flight_id=flight_id)

            # 创建预测结果记录
            prediction_result = PredictionResult.objects.create(
                aircraft=aircraft,
                base_time=timezone.now(),
                predicted_trajectory=prediction.tolist(),
                confidence_scores={'overall': confidence},
                input_sequence_length=self.args.seq_len,
                prediction_horizon=self.args.pred_len,
                processing_time_ms=int((datetime.now() - datetime.now()).total_seconds() * 1000)
            )

            logger.debug(f"预测结果已保存到数据库: {prediction_result.id}")

        except Exception as e:
            logger.error(f"保存预测结果到数据库时出错: {str(e)}")

    async def start_tracking_session(self, flight_id: str, session_config: Dict = None):
        """开始跟踪会话"""
        try:
            from .models import Aircraft, TrackingSession

            # 获取或创建飞机记录
            aircraft, created = await Aircraft.objects.aget_or_create(
                flight_id=flight_id,
                defaults={'is_active': True}
            )

            if not created:
                aircraft.is_active = True
                await aircraft.asave()

            # 创建跟踪会话
            session_config = session_config or {}
            session = TrackingSession.objects.create(
                session_id=f"{flight_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                aircraft=aircraft,
                update_interval=session_config.get('update_interval', self.update_interval),
                prediction_horizon=session_config.get('prediction_horizon', self.args.pred_len)
            )

            self.active_sessions[flight_id] = True

            logger.info(f"开始跟踪飞机 {flight_id}，会话ID: {session.session_id}")

            return session

        except Exception as e:
            logger.error(f"开始跟踪会话时出错: {str(e)}")
            return None

    async def stop_tracking_session(self, flight_id: str):
        """停止跟踪会话"""
        try:
            from .models import TrackingSession

            # 更新会话状态
            session = TrackingSession.objects.filter(
                aircraft__flight_id=flight_id,
                status='active'
            ).first()

            if session:
                session.status = 'completed'
                session.end_time = timezone.now()
                await session.asave()

            # 更新飞机状态
            aircraft = await Aircraft.objects.aget(flight_id=flight_id)
            aircraft.is_active = False
            aircraft.current_status = 'landed'
            await aircraft.asave()

            # 清理缓冲区
            if flight_id in self.trajectory_buffers:
                del self.trajectory_buffers[flight_id]
            if flight_id in self.prediction_cache:
                del self.prediction_cache[flight_id]

            self.active_sessions[flight_id] = False

            logger.info(f"停止跟踪飞机 {flight_id}")

        except Exception as e:
            logger.error(f"停止跟踪会话时出错: {str(e)}")

    def get_current_prediction(self, flight_id: str) -> Optional[Dict]:
        """获取当前预测结果"""
        return self.prediction_cache.get(flight_id)

    def get_trajectory_buffer(self, flight_id: str) -> Optional[List]:
        """获取轨迹缓冲区"""
        if flight_id in self.trajectory_buffers:
            return list(self.trajectory_buffers[flight_id])
        return None


# 全局预测引擎实例
prediction_engine = RealTimePredictionEngine()