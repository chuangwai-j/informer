from django.contrib import admin
from .models import EvaluationResult, ModelInfo, PredictionJob


@admin.register(EvaluationResult)
class EvaluationResultAdmin(admin.ModelAdmin):
    list_display = [
        'created_at', 'rmse', 'mae', 'mse', 'num_batches',
        'device', 'execution_time', 'status'
    ]
    list_filter = ['status', 'device', 'created_at']
    search_fields = ['created_at']
    readonly_fields = ['created_at']
    ordering = ['-created_at']


@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    list_display = [
        'created_at', 'total_params', 'trainable_params',
        'model_size_mb'
    ]
    readonly_fields = ['created_at', 'total_params', 'trainable_params', 'model_size_mb', 'config_json']
    ordering = ['-created_at']


@admin.register(PredictionJob)
class PredictionJobAdmin(admin.ModelAdmin):
    list_display = [
        'created_at', 'status', 'started_at', 'completed_at', 'error_message'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['error_message']
    readonly_fields = ['created_at', 'started_at', 'completed_at']
    ordering = ['-created_at']