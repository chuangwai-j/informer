import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from .models import Aircraft, FlightData, TrackingSession
from .prediction_engine import prediction_engine

logger = logging.getLogger(__name__)


class FlightDataReceiver:
    """飞行数据接收器"""

    def __init__(self):
        self.active_connections: Dict[str, AsyncWebsocketConsumer] = {}
        self.is_running = False

    async def start(self):
        """启动数据接收器"""
        if not self.is_running:
            self.is_running = True
            # 启动数据模拟器（在实际应用中，这里会连接真实的ADS-B数据源）
            asyncio.create_task(self._simulate_real_time_data())
            logger.info("飞行数据接收器已启动")

    async def stop(self):
        """停止数据接收器"""
        self.is_running = False
        logger.info("飞行数据接收器已停止")

    async def process_flight_data(self, flight_data: Dict):
        """处理接收到的飞行数据"""
        try:
            flight_id = flight_data['flight_id']

            # 更新飞机状态
            await self._update_aircraft_status(flight_data)

            # 保存飞行数据到数据库
            await self._save_flight_data(flight_data)

            # 发送到预测引擎进行处理
            await prediction_engine.process_flight_data(flight_data)

            # 通过WebSocket广播更新
            await self._broadcast_data_update(flight_id, flight_data)

        except Exception as e:
            logger.error(f"处理飞行数据时出错: {str(e)}")

    async def _update_aircraft_status(self, flight_data: Dict):
        """更新飞机状态"""
        try:
            aircraft, created = await Aircraft.objects.aget_or_create(
                flight_id=flight_data['flight_id'],
                defaults={
                    'callsign': flight_data.get('callsign', ''),
                    'current_latitude': flight_data['latitude'],
                    'current_longitude': flight_data['longitude'],
                    'current_altitude': flight_data['geo_altitude'],
                    'current_speed': flight_data.get('speed', 0),
                    'current_heading': flight_data.get('heading', 0),
                    'current_status': flight_data.get('status', 'cruising'),
                    'is_active': True
                }
            )

            if not created:
                aircraft.current_latitude = flight_data['latitude']
                aircraft.current_longitude = flight_data['longitude']
                aircraft.current_altitude = flight_data['geo_altitude']
                aircraft.current_speed = flight_data.get('speed', 0)
                aircraft.current_heading = flight_data.get('heading', 0)
                aircraft.current_status = flight_data.get('status', 'cruising')
                aircraft.is_active = True
                await aircraft.asave()

        except Exception as e:
            logger.error(f"更新飞机状态时出错: {str(e)}")

    async def _save_flight_data(self, flight_data: Dict):
        """保存飞行数据到数据库"""
        try:
            aircraft = await Aircraft.objects.aget(flight_id=flight_data['flight_id'])

            FlightData.objects.create(
                aircraft=aircraft,
                timestamp=flight_data['timestamp'],
                latitude=flight_data['latitude'],
                longitude=flight_data['longitude'],
                geo_altitude=flight_data['geo_altitude'],
                baro_altitude=flight_data['baro_altitude'],
                speed=flight_data.get('speed', 0),
                vertical_rate=flight_data.get('vertical_rate', 0),
                heading=flight_data.get('heading', 0),
                data_quality=flight_data.get('data_quality', 100)
            )

        except Exception as e:
            logger.error(f"保存飞行数据时出错: {str(e)}")

    async def _broadcast_data_update(self, flight_id: str, flight_data: Dict):
        """通过WebSocket广播数据更新"""
        try:
            from channels.layers import get_channel_layer
            channel_layer = get_channel_layer()

            # 发送给单机跟踪组
            await channel_layer.group_send(
                f'flight_{flight_id}',
                {
                    'type': 'flight_data_update',
                    'flight_id': flight_id,
                    'data': flight_data
                }
            )

            # 发送给全局跟踪组
            await channel_layer.group_send(
                'all_flights',
                {
                    'type': 'flight_status_update',
                    'flight_id': flight_id,
                    'status': flight_data.get('status', 'cruising')
                }
            )

        except Exception as e:
            logger.error(f"广播数据更新时出错: {str(e)}")

    async def _simulate_real_time_data(self):
        """模拟实时数据（用于演示）"""
        # 示例飞机数据
        sample_flights = [
            {
                'flight_id': 'CA1234',
                'callsign': 'CCA1234',
                'aircraft_type': 'A320',
                'initial_position': [39.9042, 116.4074, 1000],  # 北京
                'destination': [31.2304, 121.4737, 1000],     # 上海
                'status': 'cruising'
            },
            {
                'flight_id': 'MU5678',
                'callsign': 'CES5678',
                'aircraft_type': 'B737',
                'initial_position': [31.2304, 121.4737, 1000],  # 上海
                'destination': [23.1291, 113.2644, 1000],     # 广州
                'status': 'cruising'
            },
            {
                'flight_id': 'CZ9012',
                'callsign': 'CSN9012',
                'aircraft_type': 'A330',
                'initial_position': [23.1291, 113.2644, 1000],  # 广州
                'destination': [39.9042, 116.4074, 1000],     # 北京
                'status': 'cruising'
            }
        ]

        flight_positions = {}

        # 初始化飞机位置
        for flight in sample_flights:
            flight_id = flight['flight_id']
            flight_positions[flight_id] = {
                'current_pos': flight['initial_position'].copy(),
                'target_pos': flight['destination'].copy(),
                'progress': 0.0,
                'speed': 250,  # m/s
                'altitude': flight['initial_position'][2],
                'heading': 0
            }

            # 开始跟踪会话
            await prediction_engine.start_tracking_session(flight_id)

        # 模拟数据生成循环
        while self.is_running:
            try:
                for flight in sample_flights:
                    flight_id = flight['flight_id']
                    pos_data = flight_positions[flight_id]

                    # 更新位置
                    new_pos = self._update_position(pos_data)
                    flight_positions[flight_id]['current_pos'] = new_pos
                    flight_positions[flight_id]['progress'] += 0.01

                    # 生成飞行数据
                    flight_data = {
                        'flight_id': flight_id,
                        'callsign': flight['callsign'],
                        'timestamp': datetime.now(),
                        'latitude': new_pos[0],
                        'longitude': new_pos[1],
                        'geo_altitude': new_pos[2],
                        'baro_altitude': new_pos[2] - 50,  # 气压高度略低于几何高度
                        'speed': pos_data['speed'],
                        'heading': pos_data['heading'],
                        'vertical_rate': 0,
                        'status': flight['status'],
                        'data_quality': 95 + np.random.randint(0, 5)
                    }

                    # 处理飞行数据
                    await self.process_flight_data(flight_data)

                await asyncio.sleep(2)  # 每2秒更新一次

            except Exception as e:
                logger.error(f"模拟数据生成时出错: {str(e)}")
                await asyncio.sleep(5)

    def _update_position(self, pos_data: Dict) -> List[float]:
        """更新飞机位置"""
        import numpy as np

        current_pos = pos_data['current_pos']
        target_pos = pos_data['target_pos']
        progress = pos_data['progress']

        if progress >= 1.0:
            # 到达目标，重新开始
            progress = 0.0
            # 交换起点和终点
            pos_data['current_pos'], pos_data['target_pos'] = pos_data['target_pos'], pos_data['current_pos']

        # 线性插值计算新位置
        t = min(progress + 0.005, 1.0)  # 每次前进0.5%
        new_lat = current_pos[0] + (target_pos[0] - current_pos[0]) * t
        new_lon = current_pos[1] + (target_pos[1] - current_pos[1]) * t

        # 添加一些随机变化模拟真实飞行
        noise = np.random.normal(0, 0.0001, 2)  # 小的位置噪声
        new_lat += noise[0]
        new_lon += noise[1]

        # 计算航向
        lat_diff = target_pos[0] - current_pos[0]
        lon_diff = target_pos[1] - current_pos[1]
        heading = np.degrees(np.arctan2(lon_diff, lat_diff))
        pos_data['heading'] = heading % 360

        # 高度变化（模拟爬升和下降）
        altitude_variation = np.sin(progress * 2 * np.pi) * 500  # ±500米变化
        new_alt = pos_data['altitude'] + altitude_variation

        return [new_lat, new_lon, new_alt]

    def register_connection(self, flight_id: str, consumer: AsyncWebsocketConsumer):
        """注册WebSocket连接"""
        self.active_connections[flight_id] = consumer
        logger.info(f"注册连接: {flight_id}")

    def unregister_connection(self, flight_id: str):
        """注销WebSocket连接"""
        if flight_id in self.active_connections:
            del self.active_connections[flight_id]
            logger.info(f"注销连接: {flight_id}")


# 全局数据接收器实例
data_receiver = FlightDataReceiver()


# 需要导入numpy（在文件顶部添加）
import numpy as np