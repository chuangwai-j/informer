"""
性能优化视图
Performance-optimized views for large-scale real-time aircraft tracking

Author: Claude Code
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.contrib import messages
from django.core.paginator import Paginator
from django.template.loader import render_to_string

from .models import Aircraft, FlightData, PredictionResult, TrackingSession
from .optimized_prediction_engine import optimized_prediction_engine
from .data_pipeline import flight_data_processor
from .data_loader import large_dataset_loader, realtime_data_streamer

logger = logging.getLogger(__name__)


class OptimizedRealTimeTrackingView(View):
    """优化的实时跟踪视图"""

    template_name = 'model_evaluator/optimized_realtime_tracking.html'

    def get(self, request):
        """获取实时跟踪页面"""
        try:
            context = {
                'title': '实时飞机轨迹预测系统',
                'page_description': '基于Informer模型的高性能实时飞机轨迹预测与跟踪系统',
                'performance_config': {
                    'max_concurrent_flights': settings.REALTIME_PREDICTION['MAX_CONCURRENT_TRACKING'],
                    'update_interval': settings.REALTIME_PREDICTION['UPDATE_INTERVAL'],
                    'buffer_size': settings.REALTIME_PREDICTION['TRAJECTORY_BUFFER_SIZE'],
                    'confidence_threshold': settings.REALTIME_PREDICTION['CONFIDENCE_THRESHOLD'],
                    'max_memory_usage': settings.REALTIME_PREDICTION['MAX_MEMORY_USAGE_MB'],
                },
                'system_status': self._get_system_status()
            }

            return render(request, self.template_name, context)

        except Exception as e:
            logger.error(f"获取实时跟踪页面失败: {e}")
            messages.error(request, f"加载页面失败: {e}")
            return redirect('model_evaluator:home')

    def _get_system_status(self) -> Dict:
        """获取系统状态"""
        try:
            # 获取预测引擎状态
            engine_stats = optimized_prediction_engine.get_engine_stats()

            # 获取数据处理器状态
            processor_stats = cache.get('flight_processor_stats', {})

            # 获取系统资源状态
            system_stats = {
                'database_size': self._get_database_size(),
                'active_aircraft': Aircraft.objects.filter(is_active=True).count(),
                'total_flights': Aircraft.objects.count(),
                'recent_predictions': PredictionResult.objects.filter(
                    prediction_time__gte=timezone.now() - timedelta(hours=1)
                ).count(),
                'active_sessions': TrackingSession.objects.filter(status='active').count()
            }

            return {
                'engine': engine_stats,
                'processor': processor_stats,
                'system': system_stats,
                'timestamp': timezone.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                'error': str(e),
                'timestamp': timezone.now().isoformat()
            }

    def _get_database_size(self) -> str:
        """获取数据库大小"""
        try:
            # 这里可以添加更具体的数据库大小查询
            return "N/A"
        except Exception:
            return "N/A"


@method_decorator(csrf_exempt, name='dispatch')
class OptimizedAPIDataReceiver(View):
    """优化的API数据接收器"""

    async def post(self, request):
        """接收飞行数据"""
        try:
            # 解析JSON数据
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                return JsonResponse({
                    'status': 'error',
                    'message': f'JSON解析错误: {e}'
                }, status=400)

            # 验证数据格式
            if not isinstance(data, list):
                data = [data]

            # 批量处理数据
            results = []
            for flight_data in data:
                try:
                    # 添加时间戳（如果不存在）
                    if 'timestamp' not in flight_data:
                        flight_data['timestamp'] = timezone.now().isoformat()

                    # 处理单个数据点
                    result = await optimized_prediction_engine.process_flight_data(flight_data)
                    results.append(result)

                except Exception as e:
                    logger.error(f"处理飞行数据失败: {e}")
                    results.append({
                        'status': 'error',
                        'message': str(e),
                        'flight_id': flight_data.get('flight_id', 'unknown')
                    })

            # 统计结果
            successful = sum(1 for r in results if r.get('status') == 'processed')
            failed = len(results) - successful

            return JsonResponse({
                'status': 'success',
                'processed': successful,
                'failed': failed,
                'total': len(results),
                'results': results
            })

        except Exception as e:
            logger.error(f"数据接收处理失败: {e}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=500)


@require_http_methods(["GET", "POST"])
def api_optimized_start_tracking(request):
    """启动优化跟踪"""
    if request.method == 'GET':
        return JsonResponse({
            'status': 'info',
            'message': '使用POST方法启动跟踪',
            'parameters': {
                'flight_id': 'string - 航班号',
                'config': 'object - 可选配置'
            }
        })

    try:
        data = json.loads(request.body)
        flight_id = data.get('flight_id')

        if not flight_id:
            return JsonResponse({
                'status': 'error',
                'message': '缺少flight_id参数'
            }, status=400)

        # 创建跟踪会话
        aircraft = Aircraft.objects.get_or_create(
            flight_id=flight_id,
            defaults={
                'callsign': '',
                'aircraft_type': 'other',
                'current_status': 'ground'
            }
        )[0]

        session = TrackingSession.objects.create(
            session_id=f"{flight_id}_{int(timezone.now().timestamp())}",
            aircraft=aircraft,
            status='active',
            start_time=timezone.now(),
            **data.get('config', {})
        )

        # 启动数据流处理
        asyncio.create_task(realtime_data_streamer.start_streaming())

        return JsonResponse({
            'status': 'success',
            'message': f'开始跟踪航班 {flight_id}',
            'session_id': session.session_id,
            'config': data.get('config', {})
        })

    except Exception as e:
        logger.error(f"启动跟踪失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["POST"])
def api_optimized_stop_tracking(request):
    """停止优化跟踪"""
    try:
        data = json.loads(request.body)
        flight_id = data.get('flight_id')

        if flight_id:
            # 停止特定航班跟踪
            sessions = TrackingSession.objects.filter(
                aircraft__flight_id=flight_id,
                status='active'
            )

            updated_count = sessions.update(
                status='completed',
                end_time=timezone.now()
            )

            return JsonResponse({
                'status': 'success',
                'message': f'已停止 {updated_count} 个跟踪会话'
            })
        else:
            # 停止所有跟踪
            updated_count = TrackingSession.objects.filter(
                status='active'
            ).update(
                status='completed',
                end_time=timezone.now()
            )

            # 停止数据流处理
            asyncio.create_task(realtime_data_streamer.stop_streaming())

            return JsonResponse({
                'status': 'success',
                'message': f'已停止所有跟踪 ({updated_count} 个会话)'
            })

    except Exception as e:
        logger.error(f"停止跟踪失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def api_optimized_flight_status(request):
    """获取优化航班状态"""
    try:
        # 获取查询参数
        flight_id = request.GET.get('flight_id')
        active_only = request.GET.get('active_only', 'true').lower() == 'true'

        # 查询航班
        aircraft_queryset = Aircraft.objects.all()

        if flight_id:
            aircraft_queryset = aircraft_queryset.filter(flight_id__icontains=flight_id)

        if active_only:
            aircraft_queryset = aircraft_queryset.filter(is_active=True)

        # 分页
        page_size = int(request.GET.get('page_size', 20))
        page = int(request.GET.get('page', 1))

        paginator = Paginator(aircraft_queryset, page_size)
        aircraft_page = paginator.get_page(page)

        # 构建响应数据
        flights_data = []
        for aircraft in aircraft_page:
            # 获取最新位置
            latest_data = aircraft.flight_data.order_by('-timestamp').first()

            flight_info = {
                'flight_id': aircraft.flight_id,
                'callsign': aircraft.callsign,
                'aircraft_type': aircraft.aircraft_type,
                'current_status': aircraft.current_status,
                'last_update': aircraft.last_update.isoformat() if aircraft.last_update else None,
                'data_points_count': aircraft.data_points_count
            }

            if latest_data:
                flight_info.update({
                    'current_latitude': latest_data.latitude,
                    'current_longitude': latest_data.longitude,
                    'current_altitude': latest_data.geo_altitude,
                    'current_speed': latest_data.speed,
                    'current_heading': latest_data.heading
                })

            flights_data.append(flight_info)

        return JsonResponse({
            'status': 'success',
            'flights': flights_data,
            'pagination': {
                'current_page': page,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'page_size': page_size,
                'has_next': aircraft_page.has_next(),
                'has_previous': aircraft_page.has_previous()
            }
        })

    except Exception as e:
        logger.error(f"获取航班状态失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def api_optimized_visualization_data(request):
    """获取优化可视化数据"""
    try:
        flight_id = request.GET.get('flight_id')
        view_type = request.GET.get('view', '3d')
        time_range = int(request.GET.get('time_range', 3600))  # 默认1小时

        if not flight_id:
            return JsonResponse({
                'status': 'error',
                'message': '缺少flight_id参数'
            }, status=400)

        # 计算时间范围
        end_time = timezone.now()
        start_time = end_time - timedelta(seconds=time_range)

        # 获取飞机
        try:
            aircraft = Aircraft.objects.get(flight_id=flight_id)
        except Aircraft.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': f'航班 {flight_id} 不存在'
            }, status=404)

        # 获取轨迹数据
        flight_data = FlightData.objects.filter(
            aircraft=aircraft,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')

        # 获取预测数据
        predictions = PredictionResult.objects.filter(
            aircraft=aircraft,
            prediction_time__gte=start_time,
            prediction_time__lte=end_time,
            is_valid=True
        ).order_by('-prediction_time')

        # 转换数据格式
        trajectory_data = []
        for fd in flight_data:
            trajectory_data.append({
                'timestamp': fd.timestamp.isoformat(),
                'latitude': fd.latitude,
                'longitude': fd.longitude,
                'altitude': fd.geo_altitude,
                'speed': fd.speed,
                'heading': fd.heading,
                'data_quality': fd.data_quality
            })

        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                'prediction_time': pred.prediction_time.isoformat(),
                'base_time': pred.base_time.isoformat(),
                'trajectory': pred.predicted_trajectory,
                'confidence': pred.confidence_scores,
                'trajectory_points': pred.trajectory_points
            })

        # 构建可视化配置
        viz_config = {
            'view_type': view_type,
            'flight_info': {
                'flight_id': flight_id,
                'callsign': aircraft.callsign,
                'aircraft_type': aircraft.aircraft_type,
                'current_status': aircraft.current_status
            },
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration': time_range
            }
        }

        return JsonResponse({
            'status': 'success',
            'config': viz_config,
            'trajectory_data': trajectory_data,
            'prediction_data': prediction_data,
            'data_points': len(trajectory_data),
            'predictions_count': len(prediction_data)
        })

    except Exception as e:
        logger.error(f"获取可视化数据失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def api_optimized_system_status(request):
    """获取优化系统状态"""
    try:
        # 获取各个组件状态
        engine_status = optimized_prediction_engine.get_engine_stats()
        processor_status = cache.get('flight_processor_stats', {})
        loader_status = cache.get('data_loader_stats', {})

        # 获取数据库统计
        db_stats = _get_database_statistics_sync()

        # 获取系统资源状态
        resource_stats = _get_system_resources_sync()

        # 获取最近活动
        recent_activity = _get_recent_activity_sync()

        return JsonResponse({
            'status': 'success',
            'timestamp': timezone.now().isoformat(),
            'components': {
                'prediction_engine': engine_status,
                'data_processor': processor_status,
                'data_loader': loader_status
            },
            'database': db_stats,
            'resources': resource_stats,
            'recent_activity': recent_activity
        })

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["POST"])
def api_optimized_bulk_data_load(request):
    """批量数据加载"""
    try:
        data = json.loads(request.body)
        file_patterns = data.get('file_patterns', [])
        load_config = data.get('config', {})

        if not file_patterns:
            return JsonResponse({
                'status': 'error',
                'message': '缺少file_patterns参数'
            }, status=400)

        # 启动异步批量加载
        task_id = f"bulk_load_{int(timezone.now().timestamp())}"

        asyncio.create_task(_perform_bulk_load(task_id, file_patterns, load_config))

        return JsonResponse({
            'status': 'success',
            'message': '批量加载任务已启动',
            'task_id': task_id,
            'file_patterns': file_patterns
        })

    except Exception as e:
        logger.error(f"批量数据加载失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@require_http_methods(["GET"])
def api_optimized_performance_metrics(request):
    """获取性能指标"""
    try:
        # 时间范围
        time_range = int(request.GET.get('time_range', 3600))  # 默认1小时
        end_time = timezone.now()
        start_time = end_time - timedelta(seconds=time_range)

        # 获取预测指标
        prediction_metrics = _get_prediction_metrics_sync(start_time, end_time)

        # 获取数据流指标
        stream_metrics = _get_stream_metrics_sync(start_time, end_time)

        # 获取系统性能指标
        system_metrics = _get_system_performance_metrics_sync(start_time, end_time)

        return JsonResponse({
            'status': 'success',
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration': time_range
            },
            'metrics': {
                'prediction': prediction_metrics,
                'stream': stream_metrics,
                'system': system_metrics
            }
        })

    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


# 辅助函数
def _get_database_statistics_sync() -> Dict:
    """获取数据库统计（同步版本）"""
    try:
        return {
            'aircraft_count': Aircraft.objects.count(),
            'flight_data_count': FlightData.objects.count(),
            'prediction_count': PredictionResult.objects.count(),
            'session_count': TrackingSession.objects.count(),
            'active_aircraft': Aircraft.objects.filter(is_active=True).count(),
            'active_sessions': TrackingSession.objects.filter(status='active').count(),
            'last_hour_predictions': PredictionResult.objects.filter(
                prediction_time__gte=timezone.now() - timedelta(hours=1)
            ).count()
        }
    except Exception as e:
        logger.error(f"获取数据库统计失败: {e}")
        return {'error': str(e)}


async def _get_database_statistics() -> Dict:
    """获取数据库统计"""
    try:
        return {
            'aircraft_count': await Aircraft.objects.acount(),
            'flight_data_count': await FlightData.objects.acount(),
            'prediction_count': await PredictionResult.objects.acount(),
            'session_count': await TrackingSession.objects.acount(),
            'active_aircraft': await Aircraft.objects.filter(is_active=True).acount(),
            'active_sessions': await TrackingSession.objects.filter(status='active').acount(),
            'last_hour_predictions': await PredictionResult.objects.filter(
                prediction_time__gte=timezone.now() - timedelta(hours=1)
            ).acount()
        }
    except Exception as e:
        logger.error(f"获取数据库统计失败: {e}")
        return {'error': str(e)}


def _get_system_resources_sync() -> Dict:
    """获取系统资源状态（同步版本）"""
    try:
        return {
            'cpu': {'usage_percent': 0, 'core_count': 0},
            'memory': {'total_gb': 0, 'available_gb': 0, 'usage_percent': 0},
            'gpu': {'available': False}
        }
    except Exception as e:
        logger.error(f"获取系统资源失败: {e}")
        return {'error': str(e)}


def _get_recent_activity_sync() -> List[Dict]:
    """获取最近活动（同步版本）"""
    return []


async def _get_system_resources() -> Dict:
    """获取系统资源状态"""
    try:
        import psutil
        import torch

        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # GPU信息
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                'memory_reserved': torch.cuda.memory_reserved() / 1024 / 1024,      # MB
            }
        else:
            gpu_info = {'available': False}

        return {
            'cpu': {
                'usage_percent': cpu_percent,
                'core_count': psutil.cpu_count()
            },
            'memory': {
                'total_gb': memory.total / 1024 / 1024 / 1024,
                'available_gb': memory.available / 1024 / 1024 / 1024,
                'usage_percent': memory.percent,
                'used_gb': memory.used / 1024 / 1024 / 1024
            },
            'gpu': gpu_info
        }

    except ImportError:
        # psutil未安装
        return {
            'cpu': {'usage_percent': 0, 'core_count': 0},
            'memory': {'total_gb': 0, 'available_gb': 0, 'usage_percent': 0},
            'gpu': {'available': False}
        }
    except Exception as e:
        logger.error(f"获取系统资源失败: {e}")
        return {'error': str(e)}


async def _get_recent_activity() -> List[Dict]:
    """获取最近活动"""
    try:
        activities = []

        # 最近的预测
        recent_predictions = PredictionResult.objects.filter(
            prediction_time__gte=timezone.now() - timedelta(minutes=30)
        ).order_by('-prediction_time')[:5]

        for pred in recent_predictions:
            activities.append({
                'type': 'prediction',
                'message': f"航班 {pred.aircraft.flight_id} 生成新预测",
                'timestamp': pred.prediction_time.isoformat(),
                'details': {
                    'confidence': pred.confidence_scores.get('overall', 0) if pred.confidence_scores else 0,
                    'trajectory_points': pred.trajectory_points
                }
            })

        # 最近的会话
        recent_sessions = TrackingSession.objects.filter(
            start_time__gte=timezone.now() - timedelta(minutes=30)
        ).order_by('-start_time')[:5]

        for session in recent_sessions:
            activities.append({
                'type': 'session',
                'message': f"航班 {session.aircraft.flight_id} 开始跟踪",
                'timestamp': session.start_time.isoformat(),
                'details': {
                    'session_id': session.session_id,
                    'status': session.status
                }
            })

        # 按时间排序
        activities.sort(key=lambda x: x['timestamp'], reverse=True)

        return activities[:10]  # 返回最近10条活动

    except Exception as e:
        logger.error(f"获取最近活动失败: {e}")
        return []


async def _perform_bulk_load(task_id: str, file_patterns: List[str], config: Dict):
    """执行批量加载"""
    try:
        logger.info(f"开始批量加载任务: {task_id}")

        # 更新任务状态
        cache.set(f"bulk_load_{task_id}", {
            'status': 'running',
            'progress': 0,
            'message': '正在加载文件...'
        }, timeout=3600)

        # 执行批量加载
        result = await large_dataset_loader.load_from_multiple_sources(file_patterns)

        # 更新任务状态
        cache.set(f"bulk_load_{task_id}", {
            'status': 'completed',
            'progress': 100,
            'message': f'加载完成: {result["total_rows"]} 行数据',
            'result': result
        }, timeout=3600)

        logger.info(f"批量加载任务完成: {task_id}")

    except Exception as e:
        logger.error(f"批量加载任务失败: {task_id}, 错误: {e}")

        # 更新任务状态
        cache.set(f"bulk_load_{task_id}", {
            'status': 'failed',
            'progress': 0,
            'message': f'加载失败: {e}'
        }, timeout=3600)


async def _get_prediction_metrics(start_time: datetime, end_time: datetime) -> Dict:
    """获取预测指标"""
    try:
        predictions = PredictionResult.objects.filter(
            prediction_time__gte=start_time,
            prediction_time__lte=end_time
        )

        total_count = await predictions.acount()
        successful_count = await predictions.filter(is_valid=True).acount()

        # 计算平均置信度
        successful_preds = predictions.filter(is_valid=True)
        confidences = []
        async for pred in successful_preds:
            if pred.confidence_scores and 'overall' in pred.confidence_scores:
                confidences.append(pred.confidence_scores['overall'])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # 计算平均处理时间
        processing_times = []
        async for pred in predictions:
            if pred.processing_time_ms:
                processing_times.append(pred.processing_time_ms)

        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            'total_predictions': total_count,
            'successful_predictions': successful_count,
            'success_rate': (successful_count / total_count * 100) if total_count > 0 else 0,
            'average_confidence': avg_confidence,
            'average_processing_time_ms': avg_processing_time
        }

    except Exception as e:
        logger.error(f"获取预测指标失败: {e}")
        return {'error': str(e)}


async def _get_stream_metrics(start_time: datetime, end_time: datetime) -> Dict:
    """获取数据流指标"""
    try:
        # 获取数据流统计
        stream_stats = cache.get('realtime_streamer_stats', {})

        # 获取时间段内的数据点
        data_points = await FlightData.objects.filter(
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).acount()

        return {
            'data_points_received': stream_stats.get('received_count', 0),
            'data_points_processed': stream_stats.get('processed_count', 0),
            'processing_errors': stream_stats.get('error_count', 0),
            'current_buffer_size': stream_stats.get('buffer_size', 0),
            'time_range_data_points': data_points
        }

    except Exception as e:
        logger.error(f"获取数据流指标失败: {e}")
        return {'error': str(e)}


async def _get_system_performance_metrics(start_time: datetime, end_time: datetime) -> Dict:
    """获取系统性能指标"""
    try:
        # 获取预测引擎统计
        engine_stats = cache.get('prediction_engine_stats', {})

        return {
            'total_predictions': engine_stats.get('total_predictions', 0),
            'successful_predictions': engine_stats.get('successful_predictions', 0),
            'failed_predictions': engine_stats.get('failed_predictions', 0),
            'average_prediction_time': engine_stats.get('average_prediction_time', 0),
            'memory_usage_mb': engine_stats.get('memory_usage_mb', 0),
            'gpu_memory_usage_mb': engine_stats.get('gpu_memory_usage_mb', 0),
            'cache_hits': engine_stats.get('cache_hits', 0),
            'cache_misses': engine_stats.get('cache_misses', 0)
        }

    except Exception as e:
        logger.error(f"获取系统性能指标失败: {e}")
        return {'error': str(e)}


# 同步包装函数
def _get_prediction_metrics_sync(start_time: datetime, end_time: datetime) -> Dict:
    """获取预测指标（同步版本）"""
    try:
        predictions = PredictionResult.objects.filter(
            prediction_time__gte=start_time,
            prediction_time__lte=end_time
        )

        total_count = predictions.count()
        successful_count = predictions.filter(is_valid=True).count()

        # 计算平均置信度
        successful_preds = predictions.filter(is_valid=True)
        confidences = []
        for pred in successful_preds:
            if pred.confidence_scores and 'overall' in pred.confidence_scores:
                confidences.append(pred.confidence_scores['overall'])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # 计算平均处理时间
        processing_times = []
        for pred in predictions:
            if pred.processing_time_ms:
                processing_times.append(pred.processing_time_ms)

        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            'total_predictions': total_count,
            'successful_predictions': successful_count,
            'success_rate': (successful_count / total_count * 100) if total_count > 0 else 0,
            'average_confidence': avg_confidence,
            'average_processing_time_ms': avg_processing_time
        }

    except Exception as e:
        logger.error(f"获取预测指标失败: {e}")
        return {'error': str(e)}


def _get_stream_metrics_sync(start_time: datetime, end_time: datetime) -> Dict:
    """获取数据流指标（同步版本）"""
    try:
        # 获取数据流统计
        stream_stats = cache.get('realtime_streamer_stats', {})

        # 获取时间段内的数据点
        data_points = FlightData.objects.filter(
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).count()

        return {
            'data_points_received': stream_stats.get('received_count', 0),
            'data_points_processed': stream_stats.get('processed_count', 0),
            'processing_errors': stream_stats.get('error_count', 0),
            'current_buffer_size': stream_stats.get('buffer_size', 0),
            'time_range_data_points': data_points
        }

    except Exception as e:
        logger.error(f"获取数据流指标失败: {e}")
        return {'error': str(e)}


def _get_system_performance_metrics_sync(start_time: datetime, end_time: datetime) -> Dict:
    """获取系统性能指标（同步版本）"""
    try:
        # 获取预测引擎统计
        engine_stats = cache.get('prediction_engine_stats', {})

        return {
            'total_predictions': engine_stats.get('total_predictions', 0),
            'successful_predictions': engine_stats.get('successful_predictions', 0),
            'failed_predictions': engine_stats.get('failed_predictions', 0),
            'average_prediction_time': engine_stats.get('average_prediction_time', 0),
            'memory_usage_mb': engine_stats.get('memory_usage_mb', 0),
            'gpu_memory_usage_mb': engine_stats.get('gpu_memory_usage_mb', 0),
            'cache_hits': engine_stats.get('cache_hits', 0),
            'cache_misses': engine_stats.get('cache_misses', 0)
        }

    except Exception as e:
        logger.error(f"获取系统性能指标失败: {e}")
        return {'error': str(e)}