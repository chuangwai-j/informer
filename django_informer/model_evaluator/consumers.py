import json
import asyncio
from datetime import datetime
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)


class FlightTrackingConsumer(AsyncWebsocketConsumer):
    """单架飞机实时跟踪WebSocket消费者"""

    async def connect(self):
        self.flight_id = self.scope['url_route']['kwargs']['flight_id']
        self.flight_group_name = f'flight_{self.flight_id}'

        # 加入飞行组
        await self.channel_layer.group_add(
            self.flight_group_name,
            self.channel_name
        )

        await self.accept()
        logger.info(f"WebSocket连接已建立: 飞机 {self.flight_id}")

        # 发送连接确认
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'flight_id': self.flight_id,
            'timestamp': datetime.now().isoformat(),
            'message': f'已连接到飞机 {self.flight_id} 的实时跟踪'
        }))

    async def disconnect(self, close_code):
        # 离开飞行组
        await self.channel_layer.group_discard(
            self.flight_group_name,
            self.channel_name
        )
        logger.info(f"WebSocket连接已断开: 飞机 {self.flight_id}")

    async def receive(self, text_data):
        """接收来自客户端的消息"""
        try:
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')

            if message_type == 'get_current_position':
                # 客户端请求当前位置
                await self.send_current_position()
            elif message_type == 'start_tracking':
                # 开始跟踪
                await self.handle_start_tracking(text_data_json)
            elif message_type == 'stop_tracking':
                # 停止跟踪
                await self.handle_stop_tracking()
            else:
                logger.warning(f"未知消息类型: {message_type}")

        except json.JSONDecodeError:
            logger.error("无效的JSON格式")
            await self.send_error("无效的消息格式")
        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            await self.send_error(f"服务器错误: {str(e)}")

    async def send_current_position(self):
        """发送当前位置信息"""
        try:
            aircraft = await self.get_aircraft_data()
            if aircraft:
                position_data = {
                    'type': 'position_update',
                    'flight_id': self.flight_id,
                    'timestamp': datetime.now().isoformat(),
                    'position': {
                        'latitude': aircraft.current_latitude,
                        'longitude': aircraft.current_longitude,
                        'altitude': aircraft.current_altitude,
                        'speed': aircraft.current_speed,
                        'heading': aircraft.current_heading,
                        'status': aircraft.current_status
                    },
                    'last_update': aircraft.last_update.isoformat()
                }
                await self.send(text_data=json.dumps(position_data))
            else:
                await self.send_error(f"找不到飞机 {self.flight_id}")
        except Exception as e:
            logger.error(f"发送位置信息时出错: {str(e)}")
            await self.send_error(f"获取位置信息失败: {str(e)}")

    async def handle_start_tracking(self, data):
        """处理开始跟踪请求"""
        # 这里可以启动或激活跟踪会话
        await self.send(text_data=json.dumps({
            'type': 'tracking_started',
            'flight_id': self.flight_id,
            'timestamp': datetime.now().isoformat(),
            'message': f'开始跟踪飞机 {self.flight_id}'
        }))

    async def handle_stop_tracking(self):
        """处理停止跟踪请求"""
        # 这里可以暂停或停止跟踪会话
        await self.send(text_data=json.dumps({
            'type': 'tracking_stopped',
            'flight_id': self.flight_id,
            'timestamp': datetime.now().isoformat(),
            'message': f'停止跟踪飞机 {self.flight_id}'
        }))

    async def flight_data_update(self, event):
        """处理飞行数据更新"""
        await self.send(text_data=json.dumps({
            'type': 'flight_data_update',
            'flight_id': self.flight_id,
            'timestamp': datetime.now().isoformat(),
            'data': event['data']
        }))

    async def prediction_update(self, event):
        """处理预测更新"""
        await self.send(text_data=json.dumps({
            'type': 'prediction_update',
            'flight_id': self.flight_id,
            'timestamp': datetime.now().isoformat(),
            'prediction': event['prediction'],
            'confidence': event.get('confidence', 0.0)
        }))

    async def send_error(self, message):
        """发送错误消息"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }))

    @database_sync_to_async
    def get_aircraft_data(self):
        """从数据库获取飞机数据"""
        from .models import Aircraft
        try:
            return Aircraft.objects.get(flight_id=self.flight_id, is_active=True)
        except Aircraft.DoesNotExist:
            return None


class AllFlightsConsumer(AsyncWebsocketConsumer):
    """所有飞机全局跟踪WebSocket消费者"""

    async def connect(self):
        self.flights_group_name = 'all_flights'

        await self.channel_layer.group_add(
            self.flights_group_name,
            self.channel_name
        )

        await self.accept()
        logger.info("全局飞行跟踪WebSocket连接已建立")

        # 发送当前所有活跃飞机列表
        await self.send_active_flights_list()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.flights_group_name,
            self.channel_name
        )
        logger.info("全局飞行跟踪WebSocket连接已断开")

    async def receive(self, text_data):
        """接收客户端消息"""
        try:
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')

            if message_type == 'get_active_flights':
                await self.send_active_flights_list()
            elif message_type == 'subscribe_flight':
                flight_id = text_data_json.get('flight_id')
                await self.handle_subscribe_flight(flight_id)
            else:
                logger.warning(f"未知消息类型: {message_type}")

        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")
            await self.send_error(f"服务器错误: {str(e)}")

    async def send_active_flights_list(self):
        """发送活跃飞机列表"""
        try:
            active_flights = await self.get_active_flights()
            await self.send(text_data=json.dumps({
                'type': 'active_flights_list',
                'timestamp': datetime.now().isoformat(),
                'flights': active_flights
            }))
        except Exception as e:
            logger.error(f"发送活跃飞机列表时出错: {str(e)}")
            await self.send_error(f"获取飞机列表失败: {str(e)}")

    async def handle_subscribe_flight(self, flight_id):
        """处理订阅特定飞机"""
        await self.send(text_data=json.dumps({
            'type': 'flight_subscribed',
            'flight_id': flight_id,
            'timestamp': datetime.now().isoformat(),
            'message': f'已订阅飞机 {flight_id} 的更新'
        }))

    async def new_flight_detected(self, event):
        """新飞机检测通知"""
        await self.send(text_data=json.dumps({
            'type': 'new_flight_detected',
            'timestamp': datetime.now().isoformat(),
            'flight': event['flight']
        }))

    async def flight_status_update(self, event):
        """飞机状态更新通知"""
        await self.send(text_data=json.dumps({
            'type': 'flight_status_update',
            'timestamp': datetime.now().isoformat(),
            'flight_id': event['flight_id'],
            'status': event['status']
        }))

    async def send_error(self, message):
        """发送错误消息"""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }))

    @database_sync_to_async
    def get_active_flights(self):
        """获取活跃飞机列表"""
        from .models import Aircraft
        flights = Aircraft.objects.filter(is_active=True).values(
            'flight_id', 'callsign', 'aircraft_type', 'current_status',
            'current_latitude', 'current_longitude', 'current_altitude',
            'current_speed', 'current_heading', 'last_update'
        )
        return list(flights)


class PredictionUpdateConsumer(AsyncWebsocketConsumer):
    """预测更新专用WebSocket消费者"""

    async def connect(self):
        self.predictions_group_name = 'prediction_updates'

        await self.channel_layer.group_add(
            self.predictions_group_name,
            self.channel_name
        )

        await self.accept()
        logger.info("预测更新WebSocket连接已建立")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.predictions_group_name,
            self.channel_name
        )
        logger.info("预测更新WebSocket连接已断开")

    async def receive(self, text_data):
        """接收客户端消息"""
        try:
            text_data_json = json.loads(text_data)
            message_type = text_data_json.get('type')

            if message_type == 'get_latest_predictions':
                await self.send_latest_predictions()
            else:
                logger.warning(f"未知消息类型: {message_type}")

        except Exception as e:
            logger.error(f"处理消息时出错: {str(e)}")

    async def send_latest_predictions(self):
        """发送最新预测结果"""
        try:
            latest_predictions = await self.get_latest_predictions()
            await self.send(text_data=json.dumps({
                'type': 'latest_predictions',
                'timestamp': datetime.now().isoformat(),
                'predictions': latest_predictions
            }))
        except Exception as e:
            logger.error(f"发送最新预测时出错: {str(e)}")

    async def new_prediction_available(self, event):
        """新预测可用通知"""
        await self.send(text_data=json.dumps({
            'type': 'new_prediction',
            'timestamp': datetime.now().isoformat(),
            'flight_id': event['flight_id'],
            'prediction': event['prediction']
        }))

    @database_sync_to_async
    def get_latest_predictions(self):
        """获取最新预测结果"""
        from .models import PredictionResult
        predictions = PredictionResult.objects.filter(
            is_valid=True
        ).order_by('-prediction_time').values(
            'aircraft__flight_id', 'prediction_time', 'predicted_trajectory',
            'confidence_scores', 'rmse', 'processing_time_ms'
        )[:10]  # 最近10个预测
        return list(predictions)