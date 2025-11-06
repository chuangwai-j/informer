from django.urls import path
from . import views
from . import optimized_views

app_name = 'model_evaluator'

urlpatterns = [
    # 主页面
    path('', views.HomeView.as_view(), name='home'),

    # 模型信息
    path('model-info/', views.ModelInfoView.as_view(), name='model_info'),

    # 模型评估
    path('evaluation/', views.EvaluationView.as_view(), name='evaluation'),

    # 轨迹可视化
    path('visualization/', views.VisualizationView.as_view(), name='visualization'),
    path('interactive-3d/', views.serve_interactive_3d, name='serve_interactive_3d'),

    # 实时跟踪 - 原版本
    path('realtime/', views.RealTimeTrackingView.as_view(), name='realtime_tracking'),

    # 高性能实时跟踪 - 新版本
    path('optimized-realtime/', optimized_views.OptimizedRealTimeTrackingView.as_view(), name='optimized_realtime_tracking'),

    # API接口 - 传统功能
    path('api/evaluation/', views.api_evaluation, name='api_evaluation'),
    path('api/jobs/', views.api_jobs, name='api_jobs'),
    path('api/results/', views.api_results, name='api_results'),

    # API接口 - 实时功能
    path('api/start_tracking/', views.api_start_tracking, name='api_start_tracking'),
    path('api/stop_tracking/', views.api_stop_tracking, name='api_stop_tracking'),
    path('api/get_visualization/', views.api_get_visualization, name='api_get_visualization'),
    path('api/flight_status/', views.api_flight_status, name='api_flight_status'),
    path('api/system_status/', views.api_system_status, name='api_system_status'),

    # API接口 - 优化功能
    path('api/optimized/start_tracking/', optimized_views.api_optimized_start_tracking, name='api_optimized_start_tracking'),
    path('api/optimized/stop_tracking/', optimized_views.api_optimized_stop_tracking, name='api_optimized_stop_tracking'),
    path('api/optimized/flight_status/', optimized_views.api_optimized_flight_status, name='api_optimized_flight_status'),
    path('api/optimized/visualization_data/', optimized_views.api_optimized_visualization_data, name='api_optimized_visualization_data'),
    path('api/optimized/system_status/', optimized_views.api_optimized_system_status, name='api_optimized_system_status'),
    path('api/optimized/bulk_load/', optimized_views.api_optimized_bulk_data_load, name='api_optimized_bulk_data_load'),
    path('api/optimized/performance_metrics/', optimized_views.api_optimized_performance_metrics, name='api_optimized_performance_metrics'),

    # API数据接收
    path('api/optimized/data_receiver/', optimized_views.OptimizedAPIDataReceiver.as_view(), name='api_optimized_data_receiver'),
]