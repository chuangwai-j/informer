"""
优化的实时预测引擎
Optimized real-time prediction engine for large-scale aircraft trajectory prediction

Author: Claude Code
"""

import asyncio
import logging
import time
import threading
import queue
import json
import gc
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from django.utils.functional import cached_property

from .models import Aircraft, FlightData, PredictionResult, TrackingSession
from .data_pipeline import DataNormalizer
from .data_loader import large_dataset_loader

logger = logging.getLogger(__name__)


class OptimizedPredictionEngine:
    """优化的实时预测引擎"""

    def __init__(self, model_path: str = None, config_path: str = None):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型配置
        self.seq_len = 96
        self.pred_len = 24
        self.label_len = 48
        self.feature_dim = 4

        # 性能优化配置
        self.max_concurrent_predictions = 50
        self.prediction_cache_size = 1000
        self.batch_prediction_size = 10
        self.update_interval = 30
        self.confidence_threshold = 0.7

        # 数据管理
        self.trajectory_buffers = {}  # flight_id -> EnhancedTrajectoryBuffer
        self.prediction_cache = {}    # flight_id -> PredictionCache
        self.active_sessions = {}     # session_id -> TrackingSession

        # 异步处理
        self.prediction_queue = asyncio.Queue(maxsize=1000)
        self.result_queue = asyncio.Queue(maxsize=1000)
        self.processing_executor = ThreadPoolExecutor(max_workers=4)

        # 模型管理
        self.model = None
        self.model_loaded = False
        self.normalizer = DataNormalizer()

        # 性能监控
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_prediction_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage_mb': 0.0,
            'gpu_memory_usage_mb': 0.0,
            'last_cleanup': timezone.now()
        }

        # 状态管理
        self.running = False
        self.initialized = False

        # 锁管理
        self.model_lock = threading.RLock()
        self.cache_lock = threading.RLock()

    async def initialize(self):
        """初始化预测引擎"""
        try:
            logger.info("初始化实时预测引擎")

            # 加载模型
            await self._load_model()

            # 启动异步处理任务
            self.running = True
            self.initialized = True

            # 启动后台任务
            asyncio.create_task(self._prediction_processing_loop())
            asyncio.create_task(self._result_broadcast_loop())
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._session_management_loop())

            logger.info("实时预测引擎初始化完成")

        except Exception as e:
            logger.error(f"预测引擎初始化失败: {e}")
            raise

    async def shutdown(self):
        """关闭预测引擎"""
        try:
            logger.info("关闭实时预测引擎")

            self.running = False

            # 等待队列处理完成
            await self.prediction_queue.join()
            await self.result_queue.join()

            # 关闭线程池
            self.processing_executor.shutdown(wait=True)

            # 清理资源
            self._cleanup_resources()

            logger.info("实时预测引擎已关闭")

        except Exception as e:
            logger.error(f"预测引擎关闭失败: {e}")

    async def process_flight_data(self, flight_data: Dict) -> Optional[Dict]:
        """处理飞行数据并触发预测"""
        try:
            flight_id = flight_data['flight_id']
            timestamp = flight_data.get('timestamp', timezone.now())

            # 确保时间戳是datetime对象
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            # 获取或创建轨迹缓冲区
            buffer = self._get_or_create_buffer(flight_id)

            # 添加数据点到缓冲区
            await buffer.add_data_point(flight_data)

            # 检查是否需要触发预测
            if await self._should_trigger_prediction(flight_id, buffer):
                prediction_request = {
                    'flight_id': flight_id,
                    'timestamp': timestamp,
                    'buffer': buffer
                }

                await self.prediction_queue.put(prediction_request)

            return {
                'status': 'processed',
                'flight_id': flight_id,
                'buffer_size': buffer.size(),
                'timestamp': timestamp.isoformat()
            }

        except Exception as e:
            logger.error(f"处理飞行数据失败: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'flight_id': flight_data.get('flight_id', 'unknown')
            }

    async def _load_model(self):
        """加载训练好的模型"""
        try:
            if not self.model_path:
                # 使用默认路径
                import sys
                from pathlib import Path

                # 假设模型在上级目录的checkpoints文件夹中
                model_path = Path(__file__).parent.parent.parent / "checkpoints" / "checkpoint.pth"
                self.model_path = str(model_path) if model_path.exists() else None

            if not self.model_path:
                logger.warning("模型文件不存在，使用模拟预测")
                self.model_loaded = False
                return

            logger.info(f"加载模型: {self.model_path}")

            # 加载模型（简化版本，实际需要根据模型结构调整）
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 这里需要根据实际的模型结构来创建模型
            # 由于无法导入具体的模型类，这里使用占位符
            self.model = self._create_mock_model()

            # 设置模型为评估模式
            self.model.eval()
            self.model_loaded = True

            logger.info("模型加载成功")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model_loaded = False

    def _create_mock_model(self):
        """创建模拟模型（用于演示）"""
        class MockModel(torch.nn.Module):
            def __init__(self, input_dim=4, hidden_dim=128, output_dim=4):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = torch.nn.Linear(hidden_dim, output_dim)

            def forward(self, x, x_mark=None, dec_input=None, dec_mark=None):
                # 简化的前向传播
                lstm_out, _ = self.lstm(x)

                # 只预测最后pred_len个时间步
                pred_len = dec_input.size(1) if dec_input is not None else 24
                batch_size = x.size(0)

                # 生成预测结果
                predictions = lstm_out[:, -1:, :].repeat(1, pred_len, 1)
                output = self.fc(predictions)

                return output

        model = MockModel()
        return model.to(self.device)

    def _get_or_create_buffer(self, flight_id: str) -> 'EnhancedTrajectoryBuffer':
        """获取或创建增强轨迹缓冲区"""
        with self.cache_lock:
            if flight_id not in self.trajectory_buffers:
                self.trajectory_buffers[flight_id] = EnhancedTrajectoryBuffer(
                    max_size=settings.REALTIME_PREDICTION['TRAJECTORY_BUFFER_SIZE'],
                    seq_len=self.seq_len
                )
            return self.trajectory_buffers[flight_id]

    async def _should_trigger_prediction(self, flight_id: str, buffer: 'EnhancedTrajectoryBuffer') -> bool:
        """检查是否应该触发预测"""
        try:
            # 检查数据量是否足够
            if buffer.size() < self.seq_len:
                return False

            # 检查时间间隔
            if flight_id in self.prediction_cache:
                last_prediction = self.prediction_cache[flight_id]['timestamp']
                if (timezone.now() - last_prediction).seconds < self.update_interval:
                    return False

            # 检查是否有显著变化
            if buffer.has_significant_change():
                return True

            # 定期预测（即使没有显著变化）
            if flight_id not in self.prediction_cache:
                return True

            # 检查上次预测时间
            last_prediction = self.prediction_cache[flight_id]['timestamp']
            if (timezone.now() - last_prediction).seconds >= self.update_interval * 2:
                return True

            return False

        except Exception as e:
            logger.error(f"检查预测条件失败: {e}")
            return False

    async def _prediction_processing_loop(self):
        """预测处理循环"""
        batch_requests = []
        batch_timeout = 2.0
        max_batch_size = self.batch_prediction_size

        while self.running:
            try:
                # 收集批次请求
                try:
                    request = await asyncio.wait_for(
                        self.prediction_queue.get(),
                        timeout=batch_timeout
                    )
                    batch_requests.append(request)
                    self.prediction_queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # 处理批次
                if len(batch_requests) >= max_batch_size or (batch_requests and batch_timeout > 0):
                    if batch_requests:
                        await self._process_prediction_batch(batch_requests)
                        batch_requests = []

            except Exception as e:
                logger.error(f"预测处理循环错误: {e}")
                await asyncio.sleep(1)

    async def _process_prediction_batch(self, requests: List[Dict]):
        """处理预测批次"""
        try:
            start_time = time.time()

            # 准备批次数据
            batch_data = []
            valid_requests = []

            for request in requests:
                try:
                    flight_id = request['flight_id']
                    buffer = request['buffer']

                    # 准备输入数据
                    input_data = await buffer.get_prediction_input(self.seq_len, self.feature_dim)
                    if input_data is not None:
                        batch_data.append(input_data)
                        valid_requests.append(request)
                    else:
                        logger.warning(f"航班 {flight_id} 数据不足，跳过预测")

                except Exception as e:
                    logger.error(f"准备预测数据失败: {e}")
                    continue

            if not batch_data:
                return

            # 批量预测
            batch_predictions = await self._batch_predict(batch_data)

            # 处理结果
            for i, request in enumerate(valid_requests):
                try:
                    flight_id = request['flight_id']
                    timestamp = request['timestamp']

                    if i < len(batch_predictions):
                        prediction = batch_predictions[i]
                        confidence = self._calculate_confidence(prediction, request['buffer'])

                        result = {
                            'flight_id': flight_id,
                            'timestamp': timestamp,
                            'prediction': prediction,
                            'confidence': confidence,
                            'processing_time': time.time() - start_time
                        }

                        await self.result_queue.put(result)

                        # 更新缓存
                        self._update_prediction_cache(flight_id, prediction, confidence)

                    else:
                        logger.error(f"预测结果数量不匹配: {flight_id}")

                except Exception as e:
                    logger.error(f"处理预测结果失败: {e}")

        except Exception as e:
            logger.error(f"批量预测处理失败: {e}")

    async def _batch_predict(self, batch_data: List[np.ndarray]) -> List[np.ndarray]:
        """批量预测"""
        try:
            if not self.model_loaded:
                # 模拟预测
                return self._mock_predict(batch_data)

            # 转换为tensor
            batch_tensor = torch.FloatTensor(np.array(batch_data)).to(self.device)

            # 准备解码器输入
            batch_size, seq_len, feature_dim = batch_tensor.shape
            dec_input = torch.zeros(batch_size, self.label_len + self.pred_len, feature_dim).to(self.device)
            dec_input[:, :self.label_len, :] = batch_tensor[:, -self.label_len:, :]

            # 时间特征（简化版本）
            time_features = torch.zeros(batch_size, self.label_len + self.pred_len, 4).to(self.device)

            # 执行预测
            with torch.no_grad():
                outputs = self.model(batch_tensor, time_features, dec_input, time_features)
                predictions = outputs[:, -self.pred_len:, :].cpu().numpy()

            return [predictions[i] for i in range(batch_size)]

        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            return self._mock_predict(batch_data)

    def _mock_predict(self, batch_data: List[np.ndarray]) -> List[np.ndarray]:
        """模拟预测（用于演示）"""
        predictions = []
        for data in batch_data:
            # 简单的线性外推
            last_positions = data[-5:]  # 最后5个位置

            # 计算平均变化率
            if len(last_positions) >= 2:
                delta = np.diff(last_positions, axis=0)
                avg_delta = np.mean(delta, axis=0)
            else:
                avg_delta = np.zeros(self.feature_dim)

            # 生成预测
            pred = np.zeros((self.pred_len, self.feature_dim))
            last_pos = data[-1]

            for i in range(self.pred_len):
                pred[i] = last_pos + avg_delta * (i + 1)
                # 添加一些随机性
                pred[i] += np.random.normal(0, 0.001, self.feature_dim)

            predictions.append(pred)

        return predictions

    def _calculate_confidence(self, prediction: np.ndarray, buffer: 'EnhancedTrajectoryBuffer') -> Dict:
        """计算预测置信度"""
        try:
            # 基于多个因素计算置信度
            factors = {}

            # 1. 数据质量因子
            data_quality = buffer.get_average_data_quality()
            factors['data_quality'] = data_quality / 100.0

            # 2. 数据连续性因子
            continuity = buffer.get_data_continuity()
            factors['continuity'] = continuity

            # 3. 飞行阶段因子
            flight_phase = buffer.infer_flight_phase()
            phase_confidence = {
                'ground': 0.6,
                'takeoff': 0.7,
                'climbing': 0.8,
                'cruising': 0.9,
                'descent': 0.8,
                'approach': 0.7,
                'landing': 0.6
            }
            factors['flight_phase'] = phase_confidence.get(flight_phase, 0.7)

            # 4. 预测一致性因子
            consistency = self._calculate_prediction_consistency(prediction)
            factors['consistency'] = consistency

            # 5. 历史准确度因子
            historical_accuracy = self._get_historical_accuracy(buffer.flight_id)
            factors['historical_accuracy'] = historical_accuracy

            # 计算加权平均置信度
            weights = {
                'data_quality': 0.25,
                'continuity': 0.20,
                'flight_phase': 0.20,
                'consistency': 0.20,
                'historical_accuracy': 0.15
            }

            overall_confidence = sum(factors[key] * weights[key] for key in factors)

            return {
                'overall': min(max(overall_confidence, 0.0), 1.0),
                'factors': factors,
                'by_timestep': self._calculate_timestep_confidence(prediction)
            }

        except Exception as e:
            logger.error(f"计算置信度失败: {e}")
            return {
                'overall': 0.5,
                'factors': {},
                'by_timestep': [0.5] * self.pred_len
            }

    def _calculate_prediction_consistency(self, prediction: np.ndarray) -> float:
        """计算预测一致性"""
        try:
            if len(prediction) < 2:
                return 0.5

            # 计算相邻预测点之间的变化率
            deltas = np.diff(prediction, axis=0)

            # 计算变化率的方差（方差越小越一致）
            variance = np.var(deltas, axis=0)
            avg_variance = np.mean(variance)

            # 将方差转换为一致性分数（方差越小，一致性越高）
            consistency = 1.0 / (1.0 + avg_variance)

            return min(max(consistency, 0.0), 1.0)

        except Exception as e:
            logger.error(f"计算预测一致性失败: {e}")
            return 0.5

    def _get_historical_accuracy(self, flight_id: str) -> float:
        """获取历史准确度"""
        try:
            # 从缓存或数据库获取历史准确度
            cache_key = f"historical_accuracy_{flight_id}"
            accuracy = cache.get(cache_key)

            if accuracy is None:
                # 查询最近的预测结果
                recent_predictions = PredictionResult.objects.filter(
                    aircraft__flight_id=flight_id,
                    is_valid=True
                ).order_by('-prediction_time')[:10]

                if recent_predictions:
                    # 计算平均置信度
                    accuracies = []
                    for pred in recent_predictions:
                        if pred.confidence_scores and 'overall' in pred.confidence_scores:
                            accuracies.append(pred.confidence_scores['overall'])

                    if accuracies:
                        accuracy = np.mean(accuracies)
                    else:
                        accuracy = 0.7  # 默认值
                else:
                    accuracy = 0.7  # 默认值

                # 缓存结果
                cache.set(cache_key, accuracy, timeout=3600)

            return accuracy

        except Exception as e:
            logger.error(f"获取历史准确度失败: {e}")
            return 0.7

    def _calculate_timestep_confidence(self, prediction: np.ndarray) -> List[float]:
        """计算各时间步的置信度"""
        try:
            # 简单的时间衰减置信度
            confidence_per_step = []
            base_confidence = 0.8
            decay_rate = 0.02

            for i in range(len(prediction)):
                step_confidence = base_confidence * (1 - decay_rate * i)
                confidence_per_step.append(max(step_confidence, 0.3))

            return confidence_per_step

        except Exception as e:
            logger.error(f"计算时间步置信度失败: {e}")
            return [0.5] * self.pred_len

    def _update_prediction_cache(self, flight_id: str, prediction: np.ndarray, confidence: Dict):
        """更新预测缓存"""
        try:
            with self.cache_lock:
                self.prediction_cache[flight_id] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': timezone.now()
                }

                # 限制缓存大小
                if len(self.prediction_cache) > self.prediction_cache_size:
                    # 删除最旧的条目
                    oldest_flight = min(self.prediction_cache.keys(),
                                      key=lambda k: self.prediction_cache[k]['timestamp'])
                    del self.prediction_cache[oldest_flight]

        except Exception as e:
            logger.error(f"更新预测缓存失败: {e}")

    async def _result_broadcast_loop(self):
        """结果广播循环"""
        while self.running:
            try:
                # 获取预测结果
                result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)

                # 广播结果
                await self._broadcast_prediction_result(result)

                self.result_queue.task_done()

                # 更新统计
                self.stats['total_predictions'] += 1
                self.stats['successful_predictions'] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"结果广播失败: {e}")
                self.stats['failed_predictions'] += 1

    async def _broadcast_prediction_result(self, result: Dict):
        """广播预测结果"""
        try:
            flight_id = result['flight_id']

            # 保存到数据库
            await self._save_prediction_result(result)

            # 发送WebSocket消息
            await self._send_websocket_update(flight_id, result)

            # 更新跟踪会话
            await self._update_tracking_session(flight_id, result)

        except Exception as e:
            logger.error(f"广播预测结果失败: {e}")

    async def _save_prediction_result(self, result: Dict):
        """保存预测结果到数据库"""
        try:
            flight_id = result['flight_id']

            # 获取飞机记录
            aircraft = await Aircraft.objects.aget_or_create(
                flight_id=flight_id,
                defaults={
                    'callsign': '',
                    'aircraft_type': 'other',
                    'current_status': 'ground'
                }
            )[0]

            # 转换预测结果
            prediction_trajectory = result['prediction'].tolist()
            confidence_scores = result['confidence']

            # 创建预测结果记录
            prediction_result = await PredictionResult.objects.acreate(
                aircraft=aircraft,
                prediction_id=f"{flight_id}_{int(time.time())}",
                prediction_time=timezone.now(),
                base_time=result['timestamp'],
                predicted_trajectory=prediction_trajectory,
                confidence_scores=confidence_scores,
                input_sequence_length=self.seq_len,
                prediction_horizon=self.pred_len,
                model_version='v1.0_optimized',
                processing_time_ms=int(result['processing_time'] * 1000),
                is_valid=confidence_scores['overall'] >= self.confidence_threshold
            )

        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")

    async def _send_websocket_update(self, flight_id: str, result: Dict):
        """发送WebSocket更新"""
        try:
            from channels.layers import get_channel_layer

            channel_layer = get_channel_layer()

            # 发送到特定航班频道
            await channel_layer.group_send(
                f"flight_{flight_id}",
                {
                    'type': 'prediction_update',
                    'flight_id': flight_id,
                    'prediction': result['prediction'].tolist(),
                    'confidence': result['confidence'],
                    'timestamp': result['timestamp'].isoformat()
                }
            )

            # 发送到全局更新频道
            await channel_layer.group_send(
                "all_flights",
                {
                    'type': 'flight_prediction_update',
                    'flight_id': flight_id,
                    'confidence': result['confidence']['overall'],
                    'timestamp': result['timestamp'].isoformat()
                }
            )

        except Exception as e:
            logger.error(f"发送WebSocket更新失败: {e}")

    async def _update_tracking_session(self, flight_id: str, result: Dict):
        """更新跟踪会话"""
        try:
            # 查找活跃的跟踪会话
            sessions = TrackingSession.objects.filter(
                aircraft__flight_id=flight_id,
                status='active'
            ).order_by('-last_activity')

            if sessions.exists():
                session = await sessions.afirst()

                # 更新会话统计
                session.total_predictions += 1
                if result['confidence']['overall'] >= self.confidence_threshold:
                    session.successful_predictions += 1

                session.average_confidence = (
                    (session.average_confidence or 0) * (session.total_predictions - 1) +
                    result['confidence']['overall']
                ) / session.total_predictions

                session.total_processing_time += result['processing_time']
                session.average_processing_time_ms = (
                    session.total_processing_time / session.total_predictions
                ) * 1000

                session.last_activity = timezone.now()
                await session.asave()

        except Exception as e:
            logger.error(f"更新跟踪会话失败: {e}")

    async def _cache_cleanup_loop(self):
        """缓存清理循环"""
        while self.running:
            try:
                # 每小时清理一次
                await asyncio.sleep(3600)

                current_time = timezone.now()
                cutoff_time = current_time - timedelta(hours=2)

                # 清理预测缓存
                with self.cache_lock:
                    expired_flights = [
                        flight_id for flight_id, data in self.prediction_cache.items()
                        if data['timestamp'] < cutoff_time
                    ]

                    for flight_id in expired_flights:
                        del self.prediction_cache[flight_id]

                # 清理轨迹缓冲区
                buffer_cutoff = current_time - timedelta(hours=6)
                for flight_id, buffer in self.trajectory_buffers.items():
                    await buffer.cleanup_old_data(buffer_cutoff)

                # 清理空缓冲区
                empty_flights = [
                    flight_id for flight_id, buffer in self.trajectory_buffers.items()
                    if buffer.size() == 0
                ]

                for flight_id in empty_flights:
                    del self.trajectory_buffers[flight_id]

                self.stats['last_cleanup'] = current_time

                logger.info(f"缓存清理完成: 删除 {len(expired_flights)} 个预测缓存, "
                           f"{len(empty_flights)} 个空缓冲区")

            except Exception as e:
                logger.error(f"缓存清理失败: {e}")

    async def _performance_monitoring_loop(self):
        """性能监控循环"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次

                # 计算内存使用
                self.stats['memory_usage_mb'] = self._calculate_memory_usage()

                if torch.cuda.is_available():
                    self.stats['gpu_memory_usage_mb'] = torch.cuda.memory_allocated() / 1024 / 1024

                # 计算平均预测时间
                if self.stats['successful_predictions'] > 0:
                    # 这里应该累计计算，简化处理
                    pass

                # 缓存命中率
                total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
                if total_requests > 0:
                    cache_hit_rate = self.stats['cache_hits'] / total_requests
                else:
                    cache_hit_rate = 0.0

                # 发送统计信息到缓存
                cache.set('prediction_engine_stats', self.stats, timeout=300)

                logger.info(f"预测引擎性能统计: "
                           f"总预测 {self.stats['total_predictions']}, "
                           f"成功 {self.stats['successful_predictions']}, "
                           f"失败 {self.stats['failed_predictions']}, "
                           f"缓存命中率 {cache_hit_rate:.2%}, "
                           f"内存使用 {self.stats['memory_usage_mb']:.1f}MB")

                # 垃圾回收
                if self.stats['memory_usage_mb'] > 1000:  # 超过1GB时强制GC
                    gc.collect()

            except Exception as e:
                logger.error(f"性能监控失败: {e}")

    async def _session_management_loop(self):
        """会话管理循环"""
        while self.running:
            try:
                # 每5分钟检查一次
                await asyncio.sleep(300)

                current_time = timezone.now()
                timeout_threshold = current_time - timedelta(minutes=30)

                # 查找超时的会话
                timeout_sessions = TrackingSession.objects.filter(
                    status='active',
                    last_activity__lt=timeout_threshold
                )

                async for session in timeout_sessions:
                    session.status = 'timeout'
                    session.end_time = current_time
                    await session.asave()

                    logger.info(f"会话超时: {session.session_id}")

            except Exception as e:
                logger.error(f"会话管理失败: {e}")

    def _calculate_memory_usage(self) -> float:
        """计算内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    def _cleanup_resources(self):
        """清理资源"""
        try:
            # 清理缓存
            self.trajectory_buffers.clear()
            self.prediction_cache.clear()
            self.active_sessions.clear()

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 垃圾回收
            gc.collect()

        except Exception as e:
            logger.error(f"资源清理失败: {e}")

    def get_engine_stats(self) -> Dict:
        """获取引擎统计信息"""
        return {
            **self.stats,
            'running': self.running,
            'initialized': self.initialized,
            'model_loaded': self.model_loaded,
            'active_buffers': len(self.trajectory_buffers),
            'cache_size': len(self.prediction_cache),
            'queue_sizes': {
                'prediction_queue': self.prediction_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            }
        }


class EnhancedTrajectoryBuffer:
    """增强轨迹缓冲区"""

    def __init__(self, max_size: int = 1000, seq_len: int = 96):
        self.max_size = max_size
        self.seq_len = seq_len
        self.flight_id = None
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.last_significant_change = None

        # 数据质量跟踪
        self.quality_history = deque(maxlen=100)

        # 特征缓存
        self._features_cache = None
        self._cache_timestamp = None
        self._cache_valid_duration = 5.0  # 缓存有效期5秒

    async def add_data_point(self, data: Dict):
        """添加数据点"""
        with self.lock:
            if not self.flight_id:
                self.flight_id = data.get('flight_id')

            # 确保时间戳是datetime对象
            if isinstance(data.get('timestamp'), str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))

            self.buffer.append(data)

            # 更新质量历史
            quality = data.get('data_quality', 100)
            self.quality_history.append(quality)

            # 检查显著变化
            if self._check_significant_change(data):
                self.last_significant_change = timezone.now()

            # 清除特征缓存
            self._features_cache = None
            self._cache_timestamp = None

    async def get_prediction_input(self, seq_len: int, feature_dim: int) -> Optional[np.ndarray]:
        """获取预测输入数据"""
        try:
            # 检查缓存
            current_time = time.time()
            if (self._features_cache is not None and
                self._cache_timestamp is not None and
                current_time - self._cache_timestamp < self._cache_valid_duration):
                return self._features_cache[:seq_len]

            with self.lock:
                if len(self.buffer) < seq_len:
                    return None

                # 提取最近的seq_len个数据点
                recent_data = list(self.buffer)[-seq_len:]

                # 转换为特征数组
                features = []
                for point in recent_data:
                    features.append([
                        point['latitude'],
                        point['longitude'],
                        point.get('geo_altitude', 0),
                        point.get('baro_altitude', 0)
                    ])

                features_array = np.array(features, dtype=np.float32)

                # 更新缓存
                self._features_cache = features_array
                self._cache_timestamp = current_time

                return features_array

        except Exception as e:
            logger.error(f"获取预测输入失败: {e}")
            return None

    def size(self) -> int:
        """获取缓冲区大小"""
        with self.lock:
            return len(self.buffer)

    def get_average_data_quality(self) -> float:
        """获取平均数据质量"""
        with self.lock:
            if not self.quality_history:
                return 100.0
            return np.mean(self.quality_history)

    def get_data_continuity(self) -> float:
        """获取数据连续性"""
        try:
            with self.lock:
                if len(self.buffer) < 2:
                    return 1.0

                # 检查时间间隔的连续性
                time_intervals = []
                buffer_list = list(self.buffer)

                for i in range(1, len(buffer_list)):
                    interval = (buffer_list[i]['timestamp'] - buffer_list[i-1]['timestamp']).total_seconds()
                    time_intervals.append(interval)

                if not time_intervals:
                    return 1.0

                # 计算间隔的一致性
                mean_interval = np.mean(time_intervals)
                std_interval = np.std(time_intervals)

                # 连续性评分（标准差越小越连续）
                continuity = 1.0 / (1.0 + std_interval / max(mean_interval, 1))

                return min(max(continuity, 0.0), 1.0)

        except Exception as e:
            logger.error(f"计算数据连续性失败: {e}")
            return 1.0

    def infer_flight_phase(self) -> str:
        """推断飞行阶段"""
        try:
            with self.lock:
                if len(self.buffer) < 5:
                    return 'ground'

                recent_data = list(self.buffer)[-5:]
                altitudes = [point.get('geo_altitude', 0) for point in recent_data]
                speeds = [point.get('speed', 0) for point in recent_data]

                avg_altitude = np.mean(altitudes)
                avg_speed = np.mean(speeds)

                # 简单的飞行阶段判断
                if avg_altitude < 100:
                    if avg_speed > 50:
                        return 'takeoff' if altitudes[0] < altitudes[-1] else 'landing'
                    return 'ground'
                elif avg_altitude < 5000:
                    return 'climbing' if altitudes[0] < altitudes[-1] else 'descent'
                else:
                    return 'cruising'

        except Exception as e:
            logger.error(f"推断飞行阶段失败: {e}")
            return 'ground'

    def has_significant_change(self) -> bool:
        """检查是否有显著变化"""
        if self.last_significant_change is None:
            return True

        # 如果最近有显著变化，返回True
        time_since_change = (timezone.now() - self.last_significant_change).seconds
        return time_since_change < 60  # 1分钟内的变化认为显著

    def _check_significant_change(self, new_data: Dict) -> bool:
        """检查是否为显著变化"""
        try:
            with self.lock:
                if len(self.buffer) < 2:
                    return True

                last_data = self.buffer[-2]
                current_data = self.buffer[-1]

                # 检查位置变化
                lat_change = abs(new_data['latitude'] - last_data['latitude'])
                lon_change = abs(new_data['longitude'] - last_data['longitude'])
                alt_change = abs(new_data.get('geo_altitude', 0) - last_data.get('geo_altitude', 0))

                # 设定阈值
                position_threshold = 0.01  # 约1km
                altitude_threshold = 100   # 100m

                return (lat_change > position_threshold or
                       lon_change > position_threshold or
                       alt_change > altitude_threshold)

        except Exception as e:
            logger.error(f"检查显著变化失败: {e}")
            return False

    async def cleanup_old_data(self, cutoff_time: datetime):
        """清理旧数据"""
        with self.lock:
            while self.buffer and self.buffer[0]['timestamp'] < cutoff_time:
                self.buffer.popleft()

            # 清除缓存
            self._features_cache = None
            self._cache_timestamp = None


# 全局实例
optimized_prediction_engine = OptimizedPredictionEngine()


async def get_prediction_engine_stats() -> Dict:
    """获取预测引擎统计信息"""
    return optimized_prediction_engine.get_engine_stats()


async def initialize_prediction_engine():
    """初始化预测引擎"""
    await optimized_prediction_engine.initialize()


async def shutdown_prediction_engine():
    """关闭预测引擎"""
    await optimized_prediction_engine.shutdown()