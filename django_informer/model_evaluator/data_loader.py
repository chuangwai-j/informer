"""
大规模数据加载器
Large-scale data loader optimized for high-performance aircraft data processing

Author: Claude Code
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import gzip
import aiofiles
from django.conf import settings
from django.core.cache import cache
from django.db import transaction, connection
from django.utils import timezone
from django.core.paginator import Paginator

from .models import Aircraft, FlightData, PredictionResult, TrackingSession
from .data_pipeline import DataValidator, DataNormalizer

logger = logging.getLogger(__name__)


class LargeDatasetLoader:
    """大规模数据集加载器"""

    def __init__(self, batch_size: int = 10000, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stats = {
            'total_loaded': 0,
            'batches_processed': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

    async def load_from_csv(self, file_path: str, date_column: str = 'timestamp',
                           date_format: str = '%Y-%m-%d %H:%M:%S') -> Dict:
        """从CSV文件大规模加载数据"""
        try:
            start_time = time.time()
            self.stats['start_time'] = timezone.now()

            logger.info(f"开始从CSV文件加载数据: {file_path}")

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 使用pandas分块读取
            total_rows = 0
            batch_count = 0

            for chunk in pd.read_csv(file_path, chunksize=self.batch_size):
                try:
                    # 数据预处理
                    processed_chunk = self._preprocess_dataframe(chunk, date_column, date_format)

                    # 转换为字典列表
                    records = processed_chunk.to_dict('records')

                    # 异步处理批次
                    await self._process_batch(records)

                    batch_count += 1
                    total_rows += len(records)

                    # 进度报告
                    if batch_count % 10 == 0:
                        logger.info(f"已处理 {total_rows} 行数据")

                    # 定期提交事务
                    if batch_count % 5 == 0:
                        transaction.commit()

                except Exception as e:
                    logger.error(f"处理数据块失败: {e}")
                    self.stats['errors'] += 1
                    continue

            # 最终提交
            transaction.commit()

            end_time = time.time()
            self.stats['end_time'] = timezone.now()
            self.stats['total_loaded'] = total_rows
            self.stats['batches_processed'] = batch_count

            logger.info(f"CSV加载完成: {total_rows} 行数据，耗时 {end_time - start_time:.2f}s")

            return {
                'status': 'success',
                'total_rows': total_rows,
                'batches': batch_count,
                'errors': self.stats['errors'],
                'processing_time': end_time - start_time
            }

        except Exception as e:
            logger.error(f"CSV文件加载失败: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'stats': self.stats
            }

    async def load_from_json(self, file_path: str) -> Dict:
        """从JSON文件大规模加载数据"""
        try:
            start_time = time.time()
            self.stats['start_time'] = timezone.now()

            logger.info(f"开始从JSON文件加载数据: {file_path}")

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            batch = []
            total_rows = 0
            batch_count = 0

            # 流式读取JSON文件
            async with aiofiles.open(file_path, 'r') as f:
                async for line in f:
                    try:
                        data = json.loads(line.strip())

                        # 数据验证
                        if self.validator.validate_flight_data(data):
                            batch.append(data)
                            total_rows += 1

                            # 处理批次
                            if len(batch) >= self.batch_size:
                                await self._process_batch(batch)
                                batch_count += 1
                                total_rows += len(batch)
                                batch = []

                                if batch_count % 10 == 0:
                                    logger.info(f"已处理 {total_rows} 行数据")

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析错误: {e}")
                        self.stats['errors'] += 1
                        continue
                    except Exception as e:
                        logger.error(f"处理数据行失败: {e}")
                        self.stats['errors'] += 1
                        continue

                # 处理最后一批
                if batch:
                    await self._process_batch(batch)
                    batch_count += 1

            end_time = time.time()
            self.stats['end_time'] = timezone.now()
            self.stats['total_loaded'] = total_rows
            self.stats['batches_processed'] = batch_count

            logger.info(f"JSON加载完成: {total_rows} 行数据，耗时 {end_time - start_time:.2f}s")

            return {
                'status': 'success',
                'total_rows': total_rows,
                'batches': batch_count,
                'errors': self.stats['errors'],
                'processing_time': end_time - start_time
            }

        except Exception as e:
            logger.error(f"JSON文件加载失败: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'stats': self.stats
            }

    async def load_from_multiple_sources(self, file_patterns: List[str]) -> Dict:
        """从多个文件源并行加载数据"""
        try:
            start_time = time.time()
            total_results = []

            # 创建并行任务
            tasks = []
            for pattern in file_patterns:
                files = list(Path('.').glob(pattern))
                for file_path in files:
                    if file_path.suffix.lower() == '.csv':
                        task = self.load_from_csv(str(file_path))
                    elif file_path.suffix.lower() == '.json':
                        task = self.load_from_json(str(file_path))
                    else:
                        logger.warning(f"不支持的文件格式: {file_path}")
                        continue

                    tasks.append(task)

            # 并行执行
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 汇总结果
            total_rows = 0
            total_errors = 0
            successful_files = 0

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"加载任务失败: {result}")
                    total_errors += 1
                elif result['status'] == 'success':
                    total_rows += result['total_rows']
                    total_errors += result['errors']
                    successful_files += 1
                    logger.info(f"文件加载成功: {result['total_rows']} 行")
                else:
                    logger.error(f"文件加载失败: {result['message']}")
                    total_errors += 1

            end_time = time.time()

            logger.info(f"多源加载完成: {successful_files} 个文件, "
                       f"{total_rows} 行数据, 耗时 {end_time - start_time:.2f}s")

            return {
                'status': 'success',
                'total_files': len(tasks),
                'successful_files': successful_files,
                'total_rows': total_rows,
                'total_errors': total_errors,
                'processing_time': end_time - start_time
            }

        except Exception as e:
            logger.error(f"多源加载失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _preprocess_dataframe(self, df: pd.DataFrame, date_column: str, date_format: str) -> pd.DataFrame:
        """预处理DataFrame"""
        try:
            # 复制DataFrame避免修改原数据
            processed_df = df.copy()

            # 转换时间戳
            if date_column in processed_df.columns:
                processed_df[date_column] = pd.to_datetime(processed_df[date_column], format=date_format)

            # 数据清洗
            processed_df = self._clean_dataframe(processed_df)

            # 添加默认值
            processed_df = self._add_default_values(processed_df)

            return processed_df

        except Exception as e:
            logger.error(f"DataFrame预处理失败: {e}")
            return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗DataFrame"""
        try:
            # 删除重复行
            df = df.drop_duplicates()

            # 删除必需字段为空的行
            required_fields = ['flight_id', 'timestamp', 'latitude', 'longitude']
            for field in required_fields:
                if field in df.columns:
                    df = df.dropna(subset=[field])

            # 处理异常值
            numeric_columns = ['latitude', 'longitude', 'geo_altitude', 'speed', 'heading']
            for col in numeric_columns:
                if col in df.columns:
                    # 使用IQR方法识别异常值
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # 限制异常值而不是删除
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            return df

        except Exception as e:
            logger.error(f"DataFrame清洗失败: {e}")
            return df

    def _add_default_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加默认值"""
        try:
            # 添加默认列值
            defaults = {
                'data_quality': 100,
                'data_source': 'ADS-B',
                'is_processed': False
            }

            for col, default_value in defaults.items():
                if col not in df.columns:
                    df[col] = default_value

            # 处理缺失值
            df['geo_altitude'] = df['geo_altitude'].fillna(df['baro_altitude'])
            df['baro_altitude'] = df['baro_altitude'].fillna(df['geo_altitude'])
            df['speed'] = df['speed'].fillna(0)
            df['heading'] = df['heading'].fillna(0)
            df['vertical_rate'] = df['vertical_rate'].fillna(0)

            return df

        except Exception as e:
            logger.error(f"添加默认值失败: {e}")
            return df

    async def _process_batch(self, records: List[Dict]) -> bool:
        """处理数据批次"""
        try:
            # 按航班ID分组
            flight_groups = {}
            for record in records:
                flight_id = record.get('flight_id')
                if flight_id:
                    if flight_id not in flight_groups:
                        flight_groups[flight_id] = []
                    flight_groups[flight_id].append(record)

            # 并行处理各航班
            tasks = [
                self._process_flight_group(flight_id, group_records)
                for flight_id, group_records in flight_groups.items()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 检查结果
            success_count = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"航班组处理失败: {result}")
                    self.stats['errors'] += 1
                else:
                    success_count += 1

            return success_count > 0

        except Exception as e:
            logger.error(f"批次处理失败: {e}")
            self.stats['errors'] += 1
            return False

    async def _process_flight_group(self, flight_id: str, records: List[Dict]) -> bool:
        """处理单个航班的数据组"""
        try:
            # 获取或创建飞机记录
            aircraft = await self._get_or_create_aircraft_async(flight_id)

            # 准备飞行数据对象
            flight_data_objects = []
            for record in records:
                # 标准化数据
                normalized_record = self.normalizer.normalize_flight_data(record)

                flight_data_objects.append(FlightData(
                    aircraft=aircraft,
                    timestamp=normalized_record['timestamp'],
                    latitude=normalized_record['latitude'],
                    longitude=normalized_record['longitude'],
                    geo_altitude=normalized_record.get('geo_altitude', 0),
                    baro_altitude=normalized_record.get('baro_altitude', 0),
                    speed=normalized_record.get('speed'),
                    vertical_rate=normalized_record.get('vertical_rate'),
                    heading=normalized_record.get('heading'),
                    data_quality=normalized_record.get('data_quality', 100),
                    data_source=normalized_record.get('data_source', 'ADS-B'),
                    is_processed=False
                ))

            # 批量创建
            await FlightData.objects.abulk_create(
                flight_data_objects,
                batch_size=min(1000, len(flight_data_objects))
            )

            # 更新飞机状态
            if flight_data_objects:
                latest_data = flight_data_objects[-1]
                await self._update_aircraft_status(aircraft, latest_data)

            return True

        except Exception as e:
            logger.error(f"处理航班组 {flight_id} 失败: {e}")
            return False

    async def _get_or_create_aircraft_async(self, flight_id: str) -> Aircraft:
        """异步获取或创建飞机记录"""
        try:
            # 先尝试从缓存获取
            cache_key = f"aircraft_bulk_{flight_id}"
            aircraft = cache.get(cache_key)

            if not aircraft:
                aircraft = await Aircraft.objects.aget_or_create(
                    flight_id=flight_id,
                    defaults={
                        'callsign': '',
                        'aircraft_type': 'other',
                        'current_status': 'ground',
                        'is_active': True
                    }
                )[0]

                # 缓存飞机记录
                cache.set(cache_key, aircraft, timeout=600)

            return aircraft

        except Exception as e:
            logger.error(f"获取飞机记录失败: {e}")
            raise

    async def _update_aircraft_status(self, aircraft: Aircraft, latest_data: FlightData):
        """更新飞机状态"""
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

            # 推断飞行状态
            if latest_data.geo_altitude > 1000:
                aircraft.current_status = 'cruising'
            elif latest_data.geo_altitude > 100:
                aircraft.current_status = 'climbing'
            else:
                aircraft.current_status = 'ground'

            await aircraft.asave()

        except Exception as e:
            logger.error(f"更新飞机状态失败: {e}")


class RealTimeDataStreamer:
    """实时数据流处理器"""

    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.data_buffer = asyncio.Queue(maxsize=buffer_size)
        self.processing_queue = asyncio.Queue()
        self.running = False
        self.stats = {
            'received_count': 0,
            'processed_count': 0,
            'error_count': 0,
            'buffer_size': 0,
            'last_update': timezone.now()
        }

    async def start_streaming(self):
        """启动数据流处理"""
        self.running = True
        logger.info("启动实时数据流处理")

        # 启动处理任务
        asyncio.create_task(self._stream_processing_loop())
        asyncio.create_task(self._stats_reporting_loop())

    async def stop_streaming(self):
        """停止数据流处理"""
        self.running = False
        logger.info("停止实时数据流处理")

    async def add_data_point(self, data: Dict) -> bool:
        """添加数据点到流中"""
        try:
            if self.data_buffer.full():
                logger.warning("数据缓冲区已满，丢弃旧数据")
                try:
                    self.data_buffer.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            await self.data_buffer.put(data)
            self.stats['received_count'] += 1
            return True

        except Exception as e:
            logger.error(f"添加数据点失败: {e}")
            self.stats['error_count'] += 1
            return False

    async def _stream_processing_loop(self):
        """流处理循环"""
        batch = []
        batch_timeout = 1.0
        max_batch_size = 100

        while self.running:
            try:
                # 收集批次数据
                try:
                    data = await asyncio.wait_for(
                        self.data_buffer.get(),
                        timeout=batch_timeout
                    )
                    batch.append(data)
                except asyncio.TimeoutError:
                    pass

                # 处理批次
                if len(batch) >= max_batch_size or (batch and batch_timeout > 0):
                    if batch:
                        await self._process_stream_batch(batch)
                        batch = []

            except Exception as e:
                logger.error(f"流处理循环错误: {e}")
                await asyncio.sleep(1)

    async def _process_stream_batch(self, batch: List[Dict]):
        """处理流数据批次"""
        try:
            # 这里可以调用数据处理管道
            from .data_pipeline import flight_data_processor

            for data in batch:
                await flight_data_processor.process_flight_data(data)
                self.stats['processed_count'] += 1

            self.stats['last_update'] = timezone.now()

        except Exception as e:
            logger.error(f"流批次处理失败: {e}")
            self.stats['error_count'] += len(batch)

    async def _stats_reporting_loop(self):
        """统计报告循环"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟报告一次

                self.stats['buffer_size'] = self.data_buffer.qsize()

                cache.set('realtime_streamer_stats', self.stats, timeout=300)

                logger.info(f"实时流统计: 接收 {self.stats['received_count']} 条, "
                           f"处理 {self.stats['processed_count']} 条, "
                           f"错误 {self.stats['error_count']} 条, "
                           f"缓冲区大小 {self.stats['buffer_size']}")

            except Exception as e:
                logger.error(f"统计报告循环错误: {e}")


class DataExporter:
    """数据导出器"""

    def __init__(self):
        self.export_formats = ['csv', 'json', 'parquet']

    async def export_flight_data(self, flight_id: str, start_time: datetime,
                                end_time: datetime, format_type: str = 'csv',
                                output_path: str = None) -> str:
        """导出飞行数据"""
        try:
            if format_type not in self.export_formats:
                raise ValueError(f"不支持的导出格式: {format_type}")

            # 查询数据
            flight_data = await FlightData.objects.filter(
                aircraft__flight_id=flight_id,
                timestamp__gte=start_time,
                timestamp__lte=end_time
            ).order_by('timestamp').afetch()

            if not flight_data:
                raise ValueError(f"没有找到航班 {flight_id} 在指定时间范围内的数据")

            # 转换为DataFrame
            data_list = []
            for fd in flight_data:
                data_list.append({
                    'timestamp': fd.timestamp.isoformat(),
                    'latitude': fd.latitude,
                    'longitude': fd.longitude,
                    'geo_altitude': fd.geo_altitude,
                    'baro_altitude': fd.baro_altitude,
                    'speed': fd.speed,
                    'vertical_rate': fd.vertical_rate,
                    'heading': fd.heading,
                    'data_quality': fd.data_quality,
                    'data_source': fd.data_source
                })

            df = pd.DataFrame(data_list)

            # 生成输出文件名
            if not output_path:
                timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"flight_{flight_id}_{timestamp}.{format_type}"

            # 根据格式导出
            if format_type == 'csv':
                df.to_csv(output_path, index=False)
            elif format_type == 'json':
                df.to_json(output_path, orient='records', date_format='iso')
            elif format_type == 'parquet':
                df.to_parquet(output_path, index=False)

            logger.info(f"数据导出完成: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"数据导出失败: {e}")
            raise


# 全局实例
large_dataset_loader = LargeDatasetLoader()
realtime_data_streamer = RealTimeDataStreamer()
data_exporter = DataExporter()


async def get_data_loader_stats() -> Dict:
    """获取数据加载器统计信息"""
    return {
        'large_dataset_loader': large_dataset_loader.stats,
        'realtime_streamer': realtime_data_streamer.stats
    }