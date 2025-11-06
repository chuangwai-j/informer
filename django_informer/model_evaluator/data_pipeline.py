"""
高性能数据处理管道
Optimized data processing pipeline for large-scale flight data

Author: Claude Code
"""

import asyncio
import logging
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from concurrent.futures import ThreadPoolExecutor
import threading

from .models import Aircraft, FlightData, PredictionResult, TrackingSession

logger = logging.getLogger(__name__)


class DataBuffer:
    """高性能数据缓冲区"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.last_access = timezone.now()

    def append(self, data: Dict):
        """添加数据到缓冲区"""
        with self.lock:
            self.buffer.append(data)
            self.last_access = timezone.now()

    def get_latest(self, count: int) -> List[Dict]:
        """获取最新的N条数据"""
        with self.lock:
            return list(self.buffer)[-count:]

    def get_range(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """获取时间范围内的数据"""
        with self.lock:
            return [
                data for data in self.buffer
                if start_time <= data['timestamp'] <= end_time
            ]

    def clear_old(self, cutoff_time: datetime):
        """清理旧数据"""
        with self.lock:
            while self.buffer and self.buffer[0]['timestamp'] < cutoff_time:
                self.buffer.popleft()

    def size(self) -> int:
        """获取缓冲区大小"""
        with self.lock:
            return len(self.buffer)


class FlightDataProcessor:
    """高性能飞行数据处理器"""

    def __init__(self):
        self.buffers = {}  # flight_id -> DataBuffer
        self.processing_queue = asyncio.Queue()
        self.batch_processor = BatchDataProcessor()
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()
        self.aggregator = DataAggregator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.stats = {
            'processed_count': 0,
            'error_count': 0,
            'processing_time': 0.0,
            'last_cleanup': timezone.now()
        }

    async def start(self):
        """启动数据处理器"""
        self.running = True
        logger.info("启动飞行数据处理器")

        # 启动批处理任务
        asyncio.create_task(self._batch_processing_loop())

        # 启动清理任务
        asyncio.create_task(self._cleanup_loop())

        # 启动统计任务
        asyncio.create_task(self._stats_loop())

    async def stop(self):
        """停止数据处理器"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("飞行数据处理器已停止")

    async def process_flight_data(self, flight_data: Dict) -> bool:
        """处理单个飞行数据点"""
        try:
            start_time = time.time()

            # 数据验证
            if not self.validator.validate_flight_data(flight_data):
                self.stats['error_count'] += 1
                return False

            # 获取或创建缓冲区
            flight_id = flight_data['flight_id']
            if flight_id not in self.buffers:
                self.buffers[flight_id] = DataBuffer(
                    max_size=settings.REALTIME_PREDICTION['TRAJECTORY_BUFFER_SIZE']
                )

            # 数据标准化
            normalized_data = self.normalizer.normalize_flight_data(flight_data)

            # 添加到缓冲区
            self.buffers[flight_id].append(normalized_data)

            # 异步批处理
            await self.processing_queue.put({
                'flight_id': flight_id,
                'data': normalized_data,
                'timestamp': timezone.now()
            })

            # 更新统计
            self.stats['processed_count'] += 1
            self.stats['processing_time'] += time.time() - start_time

            return True

        except Exception as e:
            logger.error(f"处理飞行数据失败: {e}")
            self.stats['error_count'] += 1
            return False

    async def _batch_processing_loop(self):
        """批处理循环"""
        batch = []
        batch_size = 50
        batch_timeout = 2.0  # 秒

        while self.running:
            try:
                # 等待批处理数据
                try:
                    data = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=batch_timeout
                    )
                    batch.append(data)
                except asyncio.TimeoutError:
                    pass

                # 处理批次
                if len(batch) >= batch_size or (batch and batch_timeout > 0):
                    if batch:
                        await self.batch_processor.process_batch(batch)
                        batch = []

            except Exception as e:
                logger.error(f"批处理循环错误: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self):
        """清理循环"""
        while self.running:
            try:
                # 每小时清理一次
                await asyncio.sleep(3600)

                cutoff_time = timezone.now() - timedelta(hours=24)
                for buffer in self.buffers.values():
                    buffer.clear_old(cutoff_time)

                # 清理空缓冲区
                empty_flights = [
                    flight_id for flight_id, buffer in self.buffers.items()
                    if buffer.size() == 0
                ]
                for flight_id in empty_flights:
                    del self.buffers[flight_id]

                self.stats['last_cleanup'] = timezone.now()
                logger.info(f"清理完成，删除 {len(empty_flights)} 个空缓冲区")

            except Exception as e:
                logger.error(f"清理循环错误: {e}")

    async def _stats_loop(self):
        """统计循环"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟更新一次统计

                cache.set('flight_processor_stats', self.stats, timeout=300)

                if self.stats['processed_count'] > 0:
                    avg_time = self.stats['processing_time'] / self.stats['processed_count']
                    logger.info(f"数据处理器统计: 处理 {self.stats['processed_count']} 条, "
                              f"错误 {self.stats['error_count']} 条, "
                              f"平均处理时间 {avg_time:.3f}s")

            except Exception as e:
                logger.error(f"统计循环错误: {e}")

    def get_flight_buffer(self, flight_id: str) -> Optional[DataBuffer]:
        """获取飞机缓冲区"""
        return self.buffers.get(flight_id)

    def get_buffer_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        return {
            'total_flights': len(self.buffers),
            'total_buffer_size': sum(buf.size() for buf in self.buffers.values()),
            'stats': self.stats
        }


class BatchDataProcessor:
    """批数据处理器"""

    def __init__(self):
        self.batch_size = 100

    async def process_batch(self, batch: List[Dict]):
        """处理数据批次"""
        try:
            # 按航班ID分组
            flight_groups = defaultdict(list)
            for item in batch:
                flight_groups[item['flight_id']].append(item)

            # 并行处理各航班
            tasks = [
                self._process_flight_batch(flight_id, items)
                for flight_id, items in flight_groups.items()
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"批处理失败: {e}")

    async def _process_flight_batch(self, flight_id: str, items: List[Dict]):
        """处理单个航班的批次数据"""
        try:
            # 获取或创建飞机记录
            aircraft = await self._get_or_create_aircraft(flight_id)

            # 批量插入飞行数据
            flight_data_objects = []
            for item in items:
                data = item['data']
                flight_data_objects.append(FlightData(
                    aircraft=aircraft,
                    timestamp=data['timestamp'],
                    latitude=data['latitude'],
                    longitude=data['longitude'],
                    geo_altitude=data['geo_altitude'],
                    baro_altitude=data['baro_altitude'],
                    speed=data.get('speed'),
                    vertical_rate=data.get('vertical_rate'),
                    heading=data.get('heading'),
                    data_quality=data.get('data_quality', 100),
                    data_source=data.get('data_source', 'ADS-B')
                ))

            # 批量保存
            await self._bulk_create_flight_data(flight_data_objects)

            # 更新飞机当前位置
            if flight_data_objects:
                latest_data = flight_data_objects[-1]
                await self._update_aircraft_position(aircraft, latest_data)

        except Exception as e:
            logger.error(f"处理航班 {flight_id} 批次数据失败: {e}")

    async def _get_or_create_aircraft(self, flight_id: str) -> Aircraft:
        """获取或创建飞机记录"""
        # 使用缓存减少数据库查询
        cache_key = f"aircraft_{flight_id}"
        aircraft = cache.get(cache_key)

        if not aircraft:
            aircraft = await Aircraft.objects.aget_or_create(
                flight_id=flight_id,
                defaults={
                    'callsign': '',
                    'aircraft_type': 'other',
                    'current_status': 'ground'
                }
            )[0]
            cache.set(cache_key, aircraft, timeout=300)

        return aircraft

    async def _bulk_create_flight_data(self, flight_data_objects: List[FlightData]):
        """批量创建飞行数据"""
        try:
            # 分批处理，避免单次操作过大
            batch_size = 500
            for i in range(0, len(flight_data_objects), batch_size):
                batch = flight_data_objects[i:i + batch_size]
                await FlightData.objects.abulk_create(batch, batch_size=batch_size)

        except Exception as e:
            logger.error(f"批量创建飞行数据失败: {e}")

    async def _update_aircraft_position(self, aircraft: Aircraft, latest_data: FlightData):
        """更新飞机当前位置"""
        try:
            aircraft.current_latitude = latest_data.latitude
            aircraft.current_longitude = latest_data.longitude
            aircraft.current_geo_altitude = latest_data.geo_altitude
            aircraft.current_baro_altitude = latest_data.baro_altitude
            aircraft.current_speed = latest_data.speed
            aircraft.current_vertical_rate = latest_data.vertical_rate
            aircraft.current_heading = latest_data.heading
            aircraft.last_position_update = latest_data.timestamp
            aircraft.data_points_count += 1

            # 异步保存
            await aircraft.asave()

        except Exception as e:
            logger.error(f"更新飞机位置失败: {e}")


class DataValidator:
    """数据验证器"""

    def __init__(self):
        # 位置边界
        self.lat_range = (-90, 90)
        self.lon_range = (-180, 180)
        self.altitude_range = (0, 50000)
        self.speed_range = (0, 500)
        self.heading_range = (0, 360)
        self.vertical_rate_range = (-100, 100)

    def validate_flight_data(self, data: Dict) -> bool:
        """验证飞行数据"""
        try:
            # 必需字段检查
            required_fields = ['flight_id', 'timestamp', 'latitude', 'longitude']
            for field in required_fields:
                if field not in data or data[field] is None:
                    logger.warning(f"缺少必需字段: {field}")
                    return False

            # 数据类型检查
            if not isinstance(data['latitude'], (int, float)):
                return False
            if not isinstance(data['longitude'], (int, float)):
                return False

            # 范围检查
            if not (self.lat_range[0] <= data['latitude'] <= self.lat_range[1]):
                logger.warning(f"纬度超出范围: {data['latitude']}")
                return False

            if not (self.lon_range[0] <= data['longitude'] <= self.lon_range[1]):
                logger.warning(f"经度超出范围: {data['longitude']}")
                return False

            # 可选字段验证
            if 'geo_altitude' in data and data['geo_altitude'] is not None:
                if not (self.altitude_range[0] <= data['geo_altitude'] <= self.altitude_range[1]):
                    logger.warning(f"几何高度超出范围: {data['geo_altitude']}")
                    return False

            if 'speed' in data and data['speed'] is not None:
                if not (self.speed_range[0] <= data['speed'] <= self.speed_range[1]):
                    logger.warning(f"速度超出范围: {data['speed']}")
                    return False

            if 'heading' in data and data['heading'] is not None:
                if not (self.heading_range[0] <= data['heading'] <= self.heading_range[1]):
                    logger.warning(f"航向超出范围: {data['heading']}")
                    return False

            return True

        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return False


class DataNormalizer:
    """数据标准化器"""

    def __init__(self):
        # 从训练数据集获取的标准化参数
        self.normalization_params = {
            'latitude': {'mean': 47.916, 'std': 3.127},
            'longitude': {'mean': 8.103, 'std': 5.279},
            'geo_altitude': {'mean': 10205, 'std': 2281},
            'baro_altitude': {'mean': 10065, 'std': 2264},
            'speed': {'mean': 150, 'std': 50},
            'heading': {'mean': 180, 'std': 104}
        }

    def normalize_flight_data(self, data: Dict) -> Dict:
        """标准化飞行数据"""
        normalized = data.copy()

        # 确保时间戳是datetime对象
        if isinstance(normalized['timestamp'], str):
            normalized['timestamp'] = datetime.fromisoformat(normalized['timestamp'].replace('Z', '+00:00'))

        # 标准化数值字段
        for field, params in self.normalization_params.items():
            if field in normalized and normalized[field] is not None:
                normalized[f'{field}_normalized'] = (
                    normalized[field] - params['mean']
                ) / params['std']

        return normalized

    def denormalize_prediction(self, normalized_data: np.ndarray, features: List[str]) -> np.ndarray:
        """反标准化预测结果"""
        denormalized = np.zeros_like(normalized_data)

        for i, feature in enumerate(features):
            if feature in self.normalization_params:
                params = self.normalization_params[feature]
                denormalized[:, i] = (
                    normalized_data[:, i] * params['std'] + params['mean']
                )
            else:
                denormalized[:, i] = normalized_data[:, i]

        return denormalized


class DataAggregator:
    """数据聚合器"""

    def __init__(self):
        self.aggregation_cache = {}

    async def aggregate_flight_summary(self, flight_id: str, time_range: timedelta = timedelta(hours=1)) -> Dict:
        """聚合飞行摘要"""
        try:
            cache_key = f"flight_summary_{flight_id}_{int(time_range.total_seconds())}"
            cached_result = cache.get(cache_key)

            if cached_result:
                return cached_result

            # 计算时间范围
            end_time = timezone.now()
            start_time = end_time - time_range

            # 获取航班数据
            flight_data = await FlightData.objects.filter(
                aircraft__flight_id=flight_id,
                timestamp__gte=start_time,
                timestamp__lte=end_time
            ).order_by('timestamp').afetch()

            if not flight_data:
                return {}

            # 转换为DataFrame进行聚合
            data_list = []
            for fd in flight_data:
                data_list.append({
                    'timestamp': fd.timestamp,
                    'latitude': fd.latitude,
                    'longitude': fd.longitude,
                    'geo_altitude': fd.geo_altitude,
                    'speed': fd.speed or 0,
                    'heading': fd.heading or 0
                })

            df = pd.DataFrame(data_list)

            # 计算聚合指标
            summary = {
                'flight_id': flight_id,
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'data_points': len(df),
                'duration_minutes': int((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 60),
                'distance_km': self._calculate_total_distance(df),
                'altitude_stats': {
                    'max': float(df['geo_altitude'].max()),
                    'min': float(df['geo_altitude'].min()),
                    'avg': float(df['geo_altitude'].mean())
                },
                'speed_stats': {
                    'max': float(df['speed'].max()),
                    'min': float(df['speed'].min()),
                    'avg': float(df['speed'].mean())
                } if 'speed' in df.columns else None,
                'last_update': timezone.now().isoformat()
            }

            # 缓存结果
            cache.set(cache_key, summary, timeout=300)

            return summary

        except Exception as e:
            logger.error(f"聚合飞行摘要失败: {e}")
            return {}

    def _calculate_total_distance(self, df: pd.DataFrame) -> float:
        """计算总飞行距离"""
        try:
            if len(df) < 2:
                return 0.0

            total_distance = 0.0
            for i in range(1, len(df)):
                # 使用Haversine公式计算两点间距离
                lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
                lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']

                distance = self._haversine_distance(lat1, lon1, lat2, lon2)
                total_distance += distance

            return total_distance

        except Exception as e:
            logger.error(f"计算飞行距离失败: {e}")
            return 0.0

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间距离（公里）"""
        R = 6371  # 地球半径（公里）

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c


# 全局实例
flight_data_processor = FlightDataProcessor()


async def get_flight_processor_stats() -> Dict:
    """获取数据处理器统计信息"""
    return flight_data_processor.get_buffer_stats()


async def cleanup_old_data(days: int = 30):
    """清理旧数据"""
    try:
        cutoff_date = timezone.now() - timedelta(days=days)

        # 清理旧的飞行数据
        deleted_flight_data = await FlightData.objects.filter(
            timestamp__lt=cutoff_date
        ).adelete()

        # 清理旧的预测结果
        deleted_predictions = await PredictionResult.objects.filter(
            prediction_time__lt=cutoff_date
        ).adelete()

        # 清理旧的跟踪会话
        deleted_sessions = await TrackingSession.objects.filter(
            start_time__lt=cutoff_date
        ).adelete()

        logger.info(f"清理完成: 飞行数据 {deleted_flight_data[0]} 条, "
                   f"预测结果 {deleted_predictions[0]} 条, "
                   f"跟踪会话 {deleted_sessions[0]} 条")

    except Exception as e:
        logger.error(f"清理旧数据失败: {e}")