from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.db import models
from django.conf import settings
from django.utils import timezone
from datetime import datetime
import json
import time
import os
import asyncio

from .models import EvaluationResult, ModelInfo, PredictionJob, Aircraft, FlightData, PredictionResult, TrackingSession
from .services import ModelEvaluator
from .visualizer import TrajectoryVisualizer
from .prediction_engine import prediction_engine
from .data_receiver import data_receiver
from .realtime_visualizer import realtime_visualizer


class HomeView(View):
    """首页视图"""

    def get(self, request):
        return render(request, 'model_evaluator/home.html')


class ModelInfoView(View):
    """系统信息视图"""

    def get(self, request):
        try:
            evaluator = ModelEvaluator()
            model_info = evaluator.get_model_info()

            # 保存到数据库
            ModelInfo.objects.create(
                total_params=model_info['model_params'],
                trainable_params=model_info['trainable_params'],
                model_size_mb=model_info['model_size_mb'],
                config_json=model_info
            )

            context = {
                'model_info': model_info,
                'page_title': '系统信息'
            }
            return render(request, 'model_evaluator/model_info.html', context)

        except Exception as e:
            messages.error(request, f'获取系统信息失败: {str(e)}')
            return render(request, 'model_evaluator/model_info.html', {
                'error': str(e),
                'page_title': '系统信息'
            })


class EvaluationView(View):
    """模型评估视图"""

    def get(self, request):
        # 获取���史评估结果
        results = EvaluationResult.objects.all()

        # 分页
        paginator = Paginator(results, 10)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        context = {
            'page_obj': page_obj,
            'page_title': '模型评估'
        }
        return render(request, 'model_evaluator/evaluation.html', context)

    def post(self, request):
        """启动模型评估"""
        try:
            # 创建预测任务
            job = PredictionJob.objects.create(
                config_path='../config/aircraft.yaml',
                checkpoint_path='../checkpoints/checkpoint.pth',
                status='running'
            )
            job.started_at = datetime.now()
            job.save()

            # 执行评估
            evaluator = ModelEvaluator()
            start_time = time.time()

            results = evaluator.evaluate_model()
            execution_time = time.time() - start_time

            if results:
                # 保存评估结果
                eval_result = EvaluationResult.objects.create(
                    seq_len=evaluator.args.seq_len,
                    label_len=evaluator.args.label_len,
                    pred_len=evaluator.args.pred_len,
                    d_model=evaluator.args.d_model,
                    n_heads=evaluator.args.n_heads,
                    e_layers=evaluator.args.e_layers,
                    d_layers=evaluator.args.d_layers,
                    mae=results['mae'],
                    mse=results['mse'],
                    rmse=results['rmse'],
                    num_batches=results['num_batches'],
                    device=str(evaluator.device),
                    execution_time=execution_time,
                    status='success'
                )

                # 更新任务状态
                job.evaluation_result = eval_result
                job.status = 'completed'
                job.completed_at = datetime.now()
                job.save()

                messages.success(request, f'模型评估完成！RMSE: {results["rmse"]:.4f}')
            else:
                job.status = 'failed'
                job.error_message = '评估失败'
                job.completed_at = datetime.now()
                job.save()

                messages.error(request, '模型评估失败')

        except Exception as e:
            # 更新任务状态为失败
            if 'job' in locals():
                job.status = 'failed'
                job.error_message = str(e)
                job.completed_at = datetime.now()
                job.save()

            messages.error(request, f'模型评估失败: {str(e)}')

        return redirect('evaluation')


@csrf_exempt
@require_http_methods(["POST"])
def api_evaluation(request):
    """API接口: 模型评估"""
    try:
        # 获取请求参数
        config_path = request.POST.get('config_path', '../config/aircraft.yaml')
        checkpoint_path = request.POST.get('checkpoint_path', '../checkpoints/checkpoint.pth')

        # 创建任务
        job = PredictionJob.objects.create(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            status='running'
        )
        job.started_at = datetime.now()
        job.save()

        # 执行评估
        evaluator = ModelEvaluator(config_path=config_path, checkpoint_path=checkpoint_path)
        start_time = time.time()

        results = evaluator.evaluate_model()
        execution_time = time.time() - start_time

        if results:
            # 保存结果
            eval_result = EvaluationResult.objects.create(
                seq_len=evaluator.args.seq_len,
                label_len=evaluator.args.label_len,
                pred_len=evaluator.args.pred_len,
                d_model=evaluator.args.d_model,
                n_heads=evaluator.args.n_heads,
                e_layers=evaluator.args.e_layers,
                d_layers=evaluator.args.d_layers,
                mae=results['mae'],
                mse=results['mse'],
                rmse=results['rmse'],
                num_batches=results['num_batches'],
                device=str(evaluator.device),
                execution_time=execution_time,
                status='success'
            )

            # 更新任务
            job.evaluation_result = eval_result
            job.status = 'completed'
            job.completed_at = datetime.now()
            job.save()

            return JsonResponse({
                'status': 'success',
                'results': results,
                'execution_time': execution_time,
                'job_id': job.id
            })
        else:
            job.status = 'failed'
            job.error_message = '评估失败'
            job.completed_at = datetime.now()
            job.save()

            return JsonResponse({
                'status': 'error',
                'message': '评估失败'
            })

    except Exception as e:
        # 记录错误
        if 'job' in locals():
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.now()
            job.save()

        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


def api_jobs(request):
    """API接口: 获取任务列表"""
    jobs = PredictionJob.objects.all().order_by('-created_at')

    job_list = []
    for job in jobs:
        job_data = {
            'id': job.id,
            'created_at': job.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'started_at': job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else None,
            'completed_at': job.completed_at.strftime('%Y-%m-%d %H:%M:%S') if job.completed_at else None,
            'status': job.status,
            'error_message': job.error_message
        }

        # 添加评估结果
        if job.evaluation_result:
            job_data.update({
                'mae': job.evaluation_result.mae,
                'mse': job.evaluation_result.mse,
                'rmse': job.evaluation_result.rmse,
                'execution_time': job.evaluation_result.execution_time
            })

        job_list.append(job_data)

    return JsonResponse({
        'status': 'success',
        'jobs': job_list
    })


def api_results(request):
    """API接口: 获取评估结果统计"""
    try:
        # 获取最近的评估结果
        recent_results = EvaluationResult.objects.all()[:10]

        results_data = []
        for result in recent_results:
            results_data.append({
                'created_at': result.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'mae': result.mae,
                'mse': result.mse,
                'rmse': result.rmse,
                'num_batches': result.num_batches,
                'execution_time': result.execution_time
            })

        # 计算统计信息
        if recent_results.exists():
            avg_rmse = recent_results.aggregate(models.Avg('rmse'))['rmse__avg']
            avg_mae = recent_results.aggregate(models.Avg('mae'))['mae__avg']
            avg_mse = recent_results.aggregate(models.Avg('mse'))['mse__avg']
        else:
            avg_rmse = avg_mae = avg_mse = 0

        return JsonResponse({
            'status': 'success',
            'recent_results': results_data,
            'statistics': {
                'total_evaluations': EvaluationResult.objects.count(),
                'avg_rmse': round(avg_rmse, 6) if avg_rmse else 0,
                'avg_mae': round(avg_mae, 6) if avg_mae else 0,
                'avg_mse': round(avg_mse, 6) if avg_mse else 0
            }
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        })


class VisualizationView(View):
    """轨迹可视化视图"""

    def get(self, request):
        return render(request, 'model_evaluator/visualization.html', {
            'page_title': '轨迹可视化'
        })

    def post(self, request):
        """生成可视化报告"""
        try:
            # 检查是否已有可视化文件
            static_dir = os.path.join(settings.BASE_DIR, 'static')
            os.makedirs(static_dir, exist_ok=True)

            # 创建可视化器
            visualizer = TrajectoryVisualizer()

            # 生成可视化报告
            report = visualizer.generate_visualization_report(num_samples=3)

            return JsonResponse({
                'status': 'success',
                'report': {
                    'matplotlib_3d': '/static/trajectory_3d.png',
                    'matplotlib_2d': '/static/trajectory_2d.png',
                    'plotly_3d': '/static/trajectory_3d_interactive.html',
                    'error_metrics': report['error_metrics'],
                    'num_samples': report['num_samples']
                }
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'可视化生成失败: {str(e)}'
            })


def serve_interactive_3d(request):
    """提供交互式3D可视化页面"""
    try:
        html_path = os.path.join(settings.BASE_DIR, 'static', 'trajectory_3d_interactive.html')
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HttpResponse(content, content_type='text/html')
        else:
            return HttpResponse('<h1>可视化文件不存在，请先生成可视化</h1>', status=404)
    except Exception as e:
        return HttpResponse(f'<h1>文件加载失败: {str(e)}</h1>', status=500)


class RealTimeTrackingView(View):
    """实时跟踪视图"""

    def get(self, request):
        return render(request, 'model_evaluator/realtime_tracking.html', {
            'page_title': '实时飞机跟踪'
        })


@csrf_exempt
@require_http_methods(["POST"])
def api_start_tracking(request):
    """API接口: 开始实时跟踪"""
    try:
        # 初始化预测引擎
        asyncio.create_task(prediction_engine.initialize())

        # 启动数据接收器
        asyncio.create_task(data_receiver.start())

        return JsonResponse({
            'status': 'success',
            'message': '实时跟踪已启动',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'启动跟踪失败: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["POST"])
def api_stop_tracking(request):
    """API接口: 停止实时跟踪"""
    try:
        # 停止数据接收器
        asyncio.create_task(data_receiver.stop())

        # 停止所有活跃的跟踪会话
        active_flights = Aircraft.objects.filter(is_active=True)
        for aircraft in active_flights:
            asyncio.create_task(prediction_engine.stop_tracking_session(aircraft.flight_id))

        return JsonResponse({
            'status': 'success',
            'message': '实时跟踪已停止',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'停止跟踪失败: {str(e)}'
        })


def api_get_visualization(request):
    """API接口: 获取可视化数据"""
    try:
        view_type = request.GET.get('view', '3d')
        flight_id = request.GET.get('flight_id', '')

        if view_type == '3d':
            plot_data = realtime_visualizer.generate_3d_trajectory_plot(flight_id)
        elif view_type == '2d':
            plot_data = realtime_visualizer.generate_2d_projection_plot(flight_id)
        elif view_type == 'confidence':
            if flight_id:
                plot_data = realtime_visualizer.generate_confidence_gauge(flight_id)
            else:
                plot_data = {'error': '请选择特定飞机查看置信度'}
        else:
            plot_data = {'error': '不支持的视图类型'}

        return JsonResponse({
            'status': 'success',
            'plot_data': plot_data,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'获取可视化数据失败: {str(e)}'
        })


def api_flight_status(request):
    """API接口: 获取飞机状态"""
    try:
        active_flights = Aircraft.objects.filter(is_active=True).values(
            'flight_id', 'callsign', 'current_status',
            'current_latitude', 'current_longitude', 'current_altitude',
            'current_speed', 'current_heading', 'last_update'
        )

        return JsonResponse({
            'status': 'success',
            'flights': list(active_flights),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'获取飞机状态失败: {str(e)}'
        })


def api_system_status(request):
    """API接口: 获取系统状态"""
    try:
        # 统计数据
        total_flights = Aircraft.objects.count()
        active_flights = Aircraft.objects.filter(is_active=True).count()
        total_predictions = PredictionResult.objects.count()
        active_sessions = TrackingSession.objects.filter(status='active').count()

        # 最近的预测结果
        recent_predictions = PredictionResult.objects.filter(
            prediction_time__gte=timezone.now() - timezone.timedelta(hours=1)
        ).count()

        # 系统指标
        avg_confidence = PredictionResult.objects.aggregate(
            avg_conf=models.Avg('confidence_scores__overall')
        )['avg_conf'] or 0

        return JsonResponse({
            'status': 'success',
            'metrics': {
                'total_flights': total_flights,
                'active_flights': active_flights,
                'total_predictions': total_predictions,
                'recent_predictions': recent_predictions,
                'active_sessions': active_sessions,
                'average_confidence': round(avg_confidence, 3)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'获取系统状态失败: {str(e)}'
        })