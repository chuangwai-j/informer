from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import transaction
import json
import logging

logger = logging.getLogger(__name__)


class AircraftManager(models.Manager):
    """飞机模型管理器，提供高效查询方法"""

    def get_active_aircraft(self):
        """获取所有活跃飞机"""
        return self.filter(is_active=True).select_related()

    def get_by_status(self, status):
        """根据状态获取飞机"""
        return self.filter(current_status=status, is_active=True)

    def with_recent_data(self, minutes=30):
        """获取最近有数据更新的飞机"""
        cutoff_time = timezone.now() - timezone.timedelta(minutes=minutes)
        return self.filter(last_update__gte=cutoff_time, is_active=True)


class Aircraft(models.Model):
    """飞机信息模型 - 优化版本支持大数据集"""

    FLIGHT_STATUS_CHOICES = [
        ('ground', '地面'),
        ('pushback', '推出'),
        ('taxi', '滑行'),
        ('takeoff', '起飞'),
        ('initial_climb', '初始爬升'),
        ('climbing', '爬升'),
        ('cruising', '巡航'),
        ('descent', '下降'),
        ('approach', '进近'),
        ('final_approach', '最后进近'),
        ('landing', '降落'),
        ('landed', '已降落'),
        ('emergency', '紧急情况'),
    ]

    AIRCRAFT_TYPE_CHOICES = [
        ('A320', 'Airbus A320'),
        ('A330', 'Airbus A330'),
        ('A350', 'Airbus A350'),
        ('B737', 'Boeing 737'),
        ('B747', 'Boeing 747'),
        ('B777', 'Boeing 777'),
        ('B787', 'Boeing 787'),
        ('C919', 'COMAC C919'),
        ('other', '其他'),
    ]

    # 基本信息
    flight_id = models.CharField(max_length=20, unique=True, db_index=True, verbose_name='航班号')
    callsign = models.CharField(max_length=20, blank=True, db_index=True, verbose_name='呼号')
    registration = models.CharField(max_length=20, blank=True, db_index=True, verbose_name='注册号')
    aircraft_type = models.CharField(
        max_length=20,
        choices=AIRCRAFT_TYPE_CHOICES,
        default='other',
        db_index=True,
        verbose_name='机型'
    )

    # 航空公司信息
    airline = models.CharField(max_length=100, blank=True, verbose_name='航空公司')
    origin = models.CharField(max_length=4, blank=True, verbose_name='出发地IATA')
    destination = models.CharField(max_length=4, blank=True, verbose_name='目的地IATA')

    # 实时位置信息 (添加索引)
    current_latitude = models.FloatField(
        null=True, blank=True,
        db_index=True,
        verbose_name='当前纬度',
        validators=[MinValueValidator(-90), MaxValueValidator(90)]
    )
    current_longitude = models.FloatField(
        null=True, blank=True,
        db_index=True,
        verbose_name='当前经度',
        validators=[MinValueValidator(-180), MaxValueValidator(180)]
    )
    current_geo_altitude = models.FloatField(
        null=True, blank=True,
        verbose_name='当前几何高度(m)',
        validators=[MinValueValidator(0), MaxValueValidator(50000)]
    )
    current_baro_altitude = models.FloatField(
        null=True, blank=True,
        verbose_name='当前气压高度(m)',
        validators=[MinValueValidator(0), MaxValueValidator(50000)]
    )

    # 运动信息
    current_speed = models.FloatField(
        null=True, blank=True,
        verbose_name='当前速度(m/s)',
        validators=[MinValueValidator(0), MaxValueValidator(500)]
    )
    current_heading = models.FloatField(
        null=True, blank=True,
        verbose_name='当前航向(度)',
        validators=[MinValueValidator(0), MaxValueValidator(360)]
    )
    current_vertical_rate = models.FloatField(
        null=True, blank=True,
        verbose_name='当前垂直速度(m/s)',
        validators=[MinValueValidator(-50), MaxValueValidator(50)]
    )

    # 飞行状态
    current_status = models.CharField(
        max_length=20,
        choices=FLIGHT_STATUS_CHOICES,
        default='ground',
        db_index=True,
        verbose_name='飞行状态'
    )

    # 轨迹统计信息
    total_distance = models.FloatField(
        default=0.0,
        verbose_name='总飞行距离(km)'
    )
    total_flight_time = models.IntegerField(
        default=0,
        verbose_name='总飞行时间(分钟)'
    )
    data_points_count = models.IntegerField(
        default=0,
        verbose_name='数据点数量'
    )

    # 元数据
    first_seen = models.DateTimeField(auto_now_add=True, verbose_name='首次发现时间')
    last_update = models.DateTimeField(auto_now=True, verbose_name='最后更新时间')
    last_position_update = models.DateTimeField(null=True, blank=True, verbose_name='最后位置更新时间')
    is_active = models.BooleanField(default=True, db_index=True, verbose_name='是否活跃')

    # 性能优化字段
    created_at = models.DateTimeField(auto_now_add=True, db_index=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    objects = AircraftManager()

    class Meta:
        verbose_name = '飞机'
        verbose_name_plural = '飞机'
        ordering = ['-last_update']
        indexes = [
            models.Index(fields=['flight_id']),
            models.Index(fields=['callsign']),
            models.Index(fields=['registration']),
            models.Index(fields=['aircraft_type']),
            models.Index(fields=['current_status']),
            models.Index(fields=['is_active', 'last_update']),
            models.Index(fields=['current_latitude', 'current_longitude']),
        ]

    def __str__(self):
        return f"{self.flight_id} ({self.callsign or 'N/A'})"

    @property
    def current_position(self):
        """返回当前位置坐标"""
        if all([self.current_latitude, self.current_longitude, self.current_geo_altitude]):
            return {
                'latitude': self.current_latitude,
                'longitude': self.current_longitude,
                'geo_altitude': self.current_geo_altitude,
                'baro_altitude': self.current_baro_altitude,
                'speed': self.current_speed,
                'heading': self.current_heading,
                'timestamp': self.last_update.isoformat()
            }
        return None

    @property
    def flight_duration_minutes(self):
        """计算飞行时长（分钟）"""
        if self.first_seen and self.last_update:
            return int((self.last_update - self.first_seen).total_seconds() / 60)
        return 0


class FlightDataManager(models.Manager):
    """飞行数据管理器，提供高效查询方法"""

    def get_recent_data(self, flight_id, minutes=60):
        """获取最近的飞行数据"""
        cutoff_time = timezone.now() - timezone.timedelta(minutes=minutes)
        return self.filter(
            aircraft__flight_id=flight_id,
            timestamp__gte=cutoff_time
        ).order_by('-timestamp')

    def get_data_in_time_range(self, flight_id, start_time, end_time):
        """获取指定时间范围内的数据"""
        return self.filter(
            aircraft__flight_id=flight_id,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')


class FlightData(models.Model):
    """飞行数据点模型 - 优化版本支持大数据集"""

    aircraft = models.ForeignKey(
        Aircraft,
        on_delete=models.CASCADE,
        related_name='flight_data',
        db_index=True,
        verbose_name='飞机'
    )

    # 时间戳
    timestamp = models.DateTimeField(verbose_name='时间戳', db_index=True)
    received_at = models.DateTimeField(auto_now_add=True, verbose_name='接收时间')

    # 位置信息 (添加索引)
    latitude = models.FloatField(
        verbose_name='纬度',
        validators=[MinValueValidator(-90), MaxValueValidator(90)]
    )
    longitude = models.FloatField(
        verbose_name='经度',
        validators=[MinValueValidator(-180), MaxValueValidator(180)]
    )
    geo_altitude = models.FloatField(
        verbose_name='几何高度(m)',
        validators=[MinValueValidator(0), MaxValueValidator(50000)]
    )
    baro_altitude = models.FloatField(
        verbose_name='气压高度(m)',
        validators=[MinValueValidator(0), MaxValueValidator(50000)]
    )

    # 运动信息
    speed = models.FloatField(
        null=True, blank=True,
        verbose_name='速度(m/s)',
        validators=[MinValueValidator(0), MaxValueValidator(500)]
    )
    vertical_rate = models.FloatField(
        null=True, blank=True,
        verbose_name='垂直速度(m/s)',
        validators=[MinValueValidator(-50), MaxValueValidator(50)]
    )
    heading = models.FloatField(
        null=True, blank=True,
        verbose_name='航向(度)',
        validators=[MinValueValidator(0), MaxValueValidator(360)]
    )

    # 数据质量和处理信息
    data_quality = models.IntegerField(
        default=100,
        verbose_name='数据质量(%)',
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    signal_strength = models.IntegerField(
        null=True, blank=True,
        verbose_name='信号强度',
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )

    # 数据来源
    data_source = models.CharField(
        max_length=50,
        default='ADS-B',
        verbose_name='数据来源'
    )

    # 处理状态
    is_processed = models.BooleanField(default=False, db_index=True, verbose_name='是否已处理')
    processing_time_ms = models.IntegerField(
        null=True, blank=True,
        verbose_name='处理时间(毫秒)'
    )

    # 元数据
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')

    objects = FlightDataManager()

    class Meta:
        verbose_name = '飞行数据'
        verbose_name_plural = '飞行数据'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['aircraft', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['is_processed']),
            models.Index(fields=['data_source']),
            models.Index(fields=['latitude', 'longitude']),
            models.Index(fields=['aircraft', 'is_processed', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.aircraft.flight_id} - {self.timestamp}"

    @property
    def position_dict(self):
        """返回位置字典格式"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'geo_altitude': self.geo_altitude,
            'baro_altitude': self.baro_altitude,
            'speed': self.speed,
            'vertical_rate': self.vertical_rate,
            'heading': self.heading,
            'timestamp': self.timestamp.isoformat()
        }


class PredictionResultManager(models.Manager):
    """预测结果管理器"""

    def get_recent_predictions(self, flight_id, hours=2):
        """获取最近的预测结果"""
        cutoff_time = timezone.now() - timezone.timedelta(hours=hours)
        return self.filter(
            aircraft__flight_id=flight_id,
            prediction_time__gte=cutoff_time,
            is_valid=True
        ).order_by('-prediction_time')

    def get_high_confidence_predictions(self, min_confidence=0.8):
        """获取高置信度预测"""
        return self.filter(
            is_valid=True
        ).order_by('-prediction_time')


class PredictionResult(models.Model):
    """轨迹预测结果模型 - 优化版本"""

    aircraft = models.ForeignKey(
        Aircraft,
        on_delete=models.CASCADE,
        related_name='predictions',
        db_index=True,
        verbose_name='飞机'
    )

    # 预测信息
    prediction_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        verbose_name='预测ID'
    )
    prediction_time = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        verbose_name='预测时间'
    )
    base_time = models.DateTimeField(verbose_name='基准时间')

    # 预测数据 (使用JSONField存储复杂轨迹数据)
    predicted_trajectory = models.JSONField(
        verbose_name='预测轨迹',
        help_text='格式: [[lat, lon, geo_alt, baro_alt], ...]'
    )
    confidence_scores = models.JSONField(
        default=dict,
        verbose_name='置信度分数',
        help_text='格式: {"overall": 0.85, "by_feature": [0.9, 0.8, 0.7, 0.9], "by_timestep": [0.85, ...]}'
    )

    # 预测参数
    input_sequence_length = models.IntegerField(verbose_name='输入序列长度')
    prediction_horizon = models.IntegerField(verbose_name='预测时长(步数)')
    prediction_interval = models.IntegerField(
        default=1,
        verbose_name='预测间隔(秒)'
    )
    model_version = models.CharField(
        max_length=50,
        default='v1.0',
        verbose_name='模型版本'
    )

    # 评估指标
    mae = models.FloatField(
        null=True, blank=True,
        verbose_name='MAE'
    )
    mse = models.FloatField(
        null=True, blank=True,
        verbose_name='MSE'
    )
    rmse = models.FloatField(
        null=True, blank=True,
        verbose_name='RMSE'
    )

    # 元数据
    processing_time_ms = models.IntegerField(
        null=True, blank=True,
        verbose_name='处理时间(毫秒)'
    )
    memory_usage_mb = models.FloatField(
        null=True, blank=True,
        verbose_name='内存使用(MB)'
    )
    is_valid = models.BooleanField(
        default=True,
        db_index=True,
        verbose_name='是否有效'
    )

    # 创建和更新时间
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    objects = PredictionResultManager()

    class Meta:
        verbose_name = '预测结果'
        verbose_name_plural = '预测结果'
        ordering = ['-prediction_time']
        indexes = [
            models.Index(fields=['aircraft', 'prediction_time']),
            models.Index(fields=['prediction_time']),
            models.Index(fields=['prediction_id']),
            models.Index(fields=['is_valid']),
        ]

    def __str__(self):
        return f"{self.aircraft.flight_id} - {self.prediction_time.strftime('%Y-%m-%d %H:%M:%S')}"

    @property
    def overall_confidence(self):
        """获取整体置信度"""
        return self.confidence_scores.get('overall', 0.0)

    @property
    def trajectory_points(self):
        """获取轨迹点数量"""
        if isinstance(self.predicted_trajectory, str):
            trajectory = json.loads(self.predicted_trajectory)
        else:
            trajectory = self.predicted_trajectory
        return len(trajectory) if trajectory else 0


class TrackingSessionManager(models.Manager):
    """跟踪会话管理器"""

    def get_active_sessions(self):
        """获取活跃会话"""
        return self.filter(status='active').select_related('aircraft')

    def get_sessions_by_time_range(self, start_time, end_time):
        """获取时间范围内的会话"""
        return self.filter(
            start_time__gte=start_time,
            end_time__lte=end_time
        ).order_by('-start_time')


class TrackingSession(models.Model):
    """跟踪会话模型 - 优化版本"""

    STATUS_CHOICES = [
        ('pending', '等待中'),
        ('active', '活跃'),
        ('paused', '暂停'),
        ('completed', '完成'),
        ('error', '错误'),
        ('timeout', '超时'),
    ]

    session_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        verbose_name='会话ID'
    )
    aircraft = models.ForeignKey(
        Aircraft,
        on_delete=models.CASCADE,
        related_name='tracking_sessions',
        db_index=True,
        verbose_name='飞机'
    )

    # 会话状态
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending',
        db_index=True,
        verbose_name='状态'
    )

    # 会话时间
    start_time = models.DateTimeField(auto_now_add=True, verbose_name='开始时间')
    end_time = models.DateTimeField(null=True, blank=True, verbose_name='结束时间')
    last_activity = models.DateTimeField(auto_now=True, verbose_name='最后活动时间')

    # 会话统计
    total_data_points = models.IntegerField(default=0, verbose_name='总数据点数')
    total_predictions = models.IntegerField(default=0, verbose_name='总预测次数')
    successful_predictions = models.IntegerField(default=0, verbose_name='成功预测次数')
    average_confidence = models.FloatField(
        null=True, blank=True,
        verbose_name='平均置信度'
    )
    total_processing_time = models.FloatField(
        default=0.0,
        verbose_name='总处理时间(秒)'
    )

    # 配置
    update_interval = models.IntegerField(
        default=30,
        verbose_name='更新间隔(秒)',
        validators=[MinValueValidator(10), MaxValueValidator(300)]
    )
    prediction_horizon = models.IntegerField(
        default=24,
        verbose_name='预测时长(步数)',
        validators=[MinValueValidator(1), MaxValueValidator(100)]
    )
    buffer_size = models.IntegerField(
        default=200,
        verbose_name='缓冲区大小',
        validators=[MinValueValidator(50), MaxValueValidator(1000)]
    )

    # 性能监控
    peak_memory_usage_mb = models.FloatField(
        null=True, blank=True,
        verbose_name='峰值内存使用(MB)'
    )
    average_processing_time_ms = models.FloatField(
        null=True, blank=True,
        verbose_name='平均处理时间(毫秒)'
    )

    # 错误信息
    error_count = models.IntegerField(default=0, verbose_name='错误次数')
    last_error = models.TextField(null=True, blank=True, verbose_name='最后错误')

    # 创建和更新时间
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')

    objects = TrackingSessionManager()

    class Meta:
        verbose_name = '跟踪会话'
        verbose_name_plural = '跟踪会话'
        ordering = ['-last_activity']
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['aircraft']),
            models.Index(fields=['status']),
            models.Index(fields=['start_time']),
            models.Index(fields=['last_activity']),
            models.Index(fields=['aircraft', 'status', 'last_activity']),
        ]

    def __str__(self):
        return f"{self.aircraft.flight_id} - {self.session_id}"

    @property
    def duration_minutes(self):
        """计算会话持续时间（分钟）"""
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds() / 60)
        elif self.last_activity:
            return int((self.last_activity - self.start_time).total_seconds() / 60)
        return 0

    @property
    def success_rate(self):
        """计算预测成功率"""
        if self.total_predictions > 0:
            return (self.successful_predictions / self.total_predictions) * 100
        return 0.0


# 保留原有的评估结果模型（向后兼容）
class EvaluationResult(models.Model):
    """模型评估结果记录"""

    created_at = models.DateTimeField(auto_now_add=True, verbose_name='评估时间')

    # 模型配置
    seq_len = models.IntegerField(verbose_name='输入序列长度')
    label_len = models.IntegerField(verbose_name='标签长度')
    pred_len = models.IntegerField(verbose_name='预测长度')
    d_model = models.IntegerField(verbose_name='模型维度')
    n_heads = models.IntegerField(verbose_name='注意力头数')
    e_layers = models.IntegerField(verbose_name='编码器层数')
    d_layers = models.IntegerField(verbose_name='解码器层数')

    # 评估指标
    mae = models.FloatField(verbose_name='平均绝对误差')
    mse = models.FloatField(verbose_name='均方误差')
    rmse = models.FloatField(verbose_name='均方根误差')

    # 数据信息
    num_batches = models.IntegerField(verbose_name='评估批次数')
    device = models.CharField(max_length=20, verbose_name='计算设备')

    # 其他信息
    execution_time = models.FloatField(null=True, blank=True, verbose_name='执行时间(秒)')
    status = models.CharField(max_length=20, default='success', verbose_name='状态')
    error_message = models.TextField(null=True, blank=True, verbose_name='错误信息')

    class Meta:
        verbose_name = '评估结果'
        verbose_name_plural = '评估结果'
        ordering = ['-created_at']

    def __str__(self):
        return f"评估结果 - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"


class ModelInfo(models.Model):
    """模型信息记录"""

    created_at = models.DateTimeField(auto_now_add=True, verbose_name='记录时间')

    # 模型参数信息
    total_params = models.BigIntegerField(verbose_name='总参数量')
    trainable_params = models.BigIntegerField(verbose_name='可训练参数量')
    model_size_mb = models.FloatField(verbose_name='模型大小(MB)')

    # 配置信息 (JSON格式存储)
    config_json = models.JSONField(verbose_name='配置信息')

    class Meta:
        verbose_name = '模型信息'
        verbose_name_plural = '模型信息'
        ordering = ['-created_at']

    def __str__(self):
        return f"模型信息 - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"


class PredictionJob(models.Model):
    """预测任务"""

    STATUS_CHOICES = [
        ('pending', '等待中'),
        ('running', '运行中'),
        ('completed', '已完成'),
        ('failed', '失败'),
    ]

    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    started_at = models.DateTimeField(null=True, blank=True, verbose_name='开始时间')
    completed_at = models.DateTimeField(null=True, blank=True, verbose_name='完成时间')

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', verbose_name='状态')

    # 任务参数
    config_path = models.CharField(max_length=500, verbose_name='配置文件路径')
    checkpoint_path = models.CharField(max_length=500, verbose_name='模型检查点路径')

    # 结果
    evaluation_result = models.OneToOneField(
        EvaluationResult,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name='评估结果'
    )

    # 错误信息
    error_message = models.TextField(null=True, blank=True, verbose_name='错误信息')

    class Meta:
        verbose_name = '预测任务'
        verbose_name_plural = '预测任务'
        ordering = ['-created_at']

    def __str__(self):
        return f"预测任务 - {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"