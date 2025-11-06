/**
 * 高性能实时飞行跟踪客户端
 * High-performance real-time flight tracking client
 *
 * Author: Claude Code
 */

class OptimizedFlightTrackingClient {
    constructor(options = {}) {
        // 配置选项
        this.options = {
            flightId: null,
            wsUrl: null,
            updateInterval: 1000,  // 1秒更新间隔
            bufferSize: 1000,      // 缓冲区大小
            maxRetries: 5,         // 最大重连次数
            retryDelay: 2000,      // 重连延迟
            enableVisualization: true,
            enablePerformanceMonitoring: true,
            ...options
        };

        // WebSocket连接
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;

        // 数据管理
        this.flightData = [];
        this.predictions = [];
        this.performanceMetrics = {
            totalUpdates: 0,
            successfulUpdates: 0,
            failedUpdates: 0,
            averageLatency: 0,
            lastUpdate: null,
            connectionUptime: 0,
            memoryUsage: 0
        };

        // 缓存管理
        this.dataCache = new Map();
        this.cacheMaxSize = 1000;

        // 事件监听器
        this.eventListeners = new Map();

        // 性能优化
        this.updateThrottle = this.throttle(this._processUpdate.bind(this), 100);
        this.renderScheduled = false;
        this.animationFrameId = null;

        // 初始化
        this._initializeClient();
    }

    /**
     * 初始化客户端
     */
    _initializeClient() {
        this._setupWebSocket();
        this._setupEventListeners();
        this._startPerformanceMonitoring();

        console.log(`OptimizedFlightTrackingClient initialized for flight: ${this.options.flightId}`);
    }

    /**
     * 设置WebSocket连接
     */
    _setupWebSocket() {
        if (!this.options.wsUrl) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            this.options.wsUrl = `${protocol}//${host}/ws/flight/tracking/${this.options.flightId}/`;
        }

        this._connectWebSocket();
    }

    /**
     * 连接WebSocket
     */
    _connectWebSocket() {
        try {
            this.ws = new WebSocket(this.options.wsUrl);

            this.ws.onopen = () => {
                console.log(`WebSocket connected to flight ${this.options.flightId}`);
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.performanceMetrics.connectionUptime = Date.now();

                this._emit('connection:established');
                this._requestInitialData();
            };

            this.ws.onmessage = (event) => {
                this._handleMessage(event);
            };

            this.ws.onclose = (event) => {
                console.log(`WebSocket closed for flight ${this.options.flightId}`, event);
                this.isConnected = false;
                this._emit('connection:closed', { code: event.code, reason: event.reason });

                this._scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error(`WebSocket error for flight ${this.options.flightId}`, error);
                this._emit('connection:error', error);
                this.performanceMetrics.failedUpdates++;
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this._scheduleReconnect();
        }
    }

    /**
     * 处理WebSocket消息
     */
    _handleMessage(event) {
        try {
            const startTime = performance.now();
            const data = JSON.parse(event.data);

            this.performanceMetrics.totalUpdates++;

            // 节流处理更新
            this.updateThrottle(data);

            // 更新延迟指标
            const latency = performance.now() - startTime;
            this._updateLatencyMetric(latency);

        } catch (error) {
            console.error('Failed to handle WebSocket message:', error);
            this.performanceMetrics.failedUpdates++;
        }
    }

    /**
     * 处理数据更新
     */
    _processUpdate(data) {
        try {
            switch (data.type) {
                case 'position_update':
                    this._handlePositionUpdate(data);
                    break;
                case 'prediction_update':
                    this._handlePredictionUpdate(data);
                    break;
                case 'flight_status':
                    this._handleFlightStatus(data);
                    break;
                case 'error':
                    this._handleError(data);
                    break;
                default:
                    console.warn('Unknown message type:', data.type);
            }

            this.performanceMetrics.successfulUpdates++;
            this.performanceMetrics.lastUpdate = Date.now();

            this._emit('data:update', data);

        } catch (error) {
            console.error('Failed to process update:', error);
            this.performanceMetrics.failedUpdates++;
        }
    }

    /**
     * 处理位置更新
     */
    _handlePositionUpdate(data) {
        const positionData = {
            timestamp: data.timestamp || new Date().toISOString(),
            latitude: data.position.latitude,
            longitude: data.position.longitude,
            altitude: data.position.altitude,
            speed: data.position.speed,
            heading: data.position.heading,
            flight_id: data.flight_id
        };

        // 添加到数据数组
        this.flightData.push(positionData);

        // 限制数组大小
        if (this.flightData.length > this.options.bufferSize) {
            this.flightData = this.flightData.slice(-this.options.bufferSize);
        }

        // 更新缓存
        this._updateCache(`position_${data.flight_id}`, positionData);

        // 调度可视化更新
        if (this.options.enableVisualization) {
            this._scheduleVisualizationUpdate();
        }

        this._emit('position:update', positionData);
    }

    /**
     * 处理预测更新
     */
    _handlePredictionUpdate(data) {
        const predictionData = {
            flight_id: data.flight_id,
            timestamp: data.timestamp,
            prediction: data.prediction,
            confidence: data.confidence,
            trajectory_points: data.prediction ? data.prediction.length : 0
        };

        // 更新预测数据
        this.predictions = [predictionData]; // 只保留最新预测

        // 更新缓存
        this._updateCache(`prediction_${data.flight_id}`, predictionData);

        // 调度可视化更新
        if (this.options.enableVisualization) {
            this._scheduleVisualizationUpdate();
        }

        this._emit('prediction:update', predictionData);
    }

    /**
     * 处理飞行状态更新
     */
    _handleFlightStatus(data) {
        this._updateCache(`status_${data.flight_id}`, data);
        this._emit('flight:status', data);
    }

    /**
     * 处理错误
     */
    _handleError(data) {
        console.error('Received error from server:', data);
        this._emit('server:error', data);
    }

    /**
     * 请求初始数据
     */
    _requestInitialData() {
        this.sendRequest('get_current_position');
        this.sendRequest('get_recent_predictions');
    }

    /**
     * 发送WebSocket请求
     */
    sendRequest(type, data = {}) {
        if (!this.isConnected || !this.ws) {
            console.warn('WebSocket not connected, cannot send request:', type);
            return false;
        }

        const message = {
            type,
            flight_id: this.options.flightId,
            timestamp: new Date().toISOString(),
            ...data
        };

        try {
            this.ws.send(JSON.stringify(message));
            return true;
        } catch (error) {
            console.error('Failed to send WebSocket message:', error);
            return false;
        }
    }

    /**
     * 安排重连
     */
    _scheduleReconnect() {
        if (this.reconnectAttempts >= this.options.maxRetries) {
            console.error('Max reconnection attempts reached');
            this._emit('connection:failed');
            return;
        }

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }

        const delay = this.options.retryDelay * Math.pow(2, this.reconnectAttempts);

        this.reconnectTimer = setTimeout(() => {
            console.log(`Attempting to reconnect (${this.reconnectAttempts + 1}/${this.options.maxRetries})`);
            this.reconnectAttempts++;
            this._connectWebSocket();
        }, delay);
    }

    /**
     * 安排可视化更新
     */
    _scheduleVisualizationUpdate() {
        if (this.renderScheduled) {
            return;
        }

        this.renderScheduled = true;
        this.animationFrameId = requestAnimationFrame(() => {
            this._updateVisualization();
            this.renderScheduled = false;
        });
    }

    /**
     * 更新可视化
     */
    _updateVisualization() {
        // 这个方法应该被子类重写
        this._emit('visualization:update', {
            flightData: this.flightData,
            predictions: this.predictions
        });
    }

    /**
     * 缓存管理
     */
    _updateCache(key, value) {
        // 如果缓存已满，删除最旧的条目
        if (this.dataCache.size >= this.cacheMaxSize) {
            const firstKey = this.dataCache.keys().next().value;
            this.dataCache.delete(firstKey);
        }

        this.dataCache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    /**
     * 获取缓存数据
     */
    getCache(key) {
        const cached = this.dataCache.get(key);
        if (cached) {
            return cached.value;
        }
        return null;
    }

    /**
     * 更新延迟指标
     */
    _updateLatencyMetric(latency) {
        const alpha = 0.1; // 平滑因子
        this.performanceMetrics.averageLatency =
            this.performanceMetrics.averageLatency * (1 - alpha) + latency * alpha;
    }

    /**
     * 启动性能监控
     */
    _startPerformanceMonitoring() {
        if (!this.options.enablePerformanceMonitoring) {
            return;
        }

        setInterval(() => {
            this._collectPerformanceMetrics();
        }, 5000); // 每5秒收集一次指标
    }

    /**
     * 收集性能指标
     */
    _collectPerformanceMetrics() {
        if (performance.memory) {
            this.performanceMetrics.memoryUsage = performance.memory.usedJSHeapSize / 1024 / 1024; // MB
        }

        if (this.isConnected) {
            this.performanceMetrics.connectionUptime = Date.now() - this.performanceMetrics.connectionUptime;
        }

        this._emit('performance:metrics', this.performanceMetrics);
    }

    /**
     * 事件系统
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }

    off(event, callback) {
        if (this.eventListeners.has(event)) {
            const listeners = this.eventListeners.get(event);
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }

    _emit(event, data) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    /**
     * 工具函数
     */
    throttle(func, delay) {
        let lastCall = 0;
        return function(...args) {
            const now = Date.now();
            if (now - lastCall >= delay) {
                lastCall = now;
                return func.apply(this, args);
            }
        };
    }

    /**
     * 获取最新位置
     */
    getLatestPosition() {
        return this.flightData.length > 0 ? this.flightData[this.flightData.length - 1] : null;
    }

    /**
     * 获取最新预测
     */
    getLatestPrediction() {
        return this.predictions.length > 0 ? this.predictions[0] : null;
    }

    /**
     * 获取性能指标
     */
    getPerformanceMetrics() {
        return { ...this.performanceMetrics };
    }

    /**
     * 销毁客户端
     */
    destroy() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }

        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        this.eventListeners.clear();
        this.dataCache.clear();
        this.flightData = [];
        this.predictions = [];

        console.log(`OptimizedFlightTrackingClient for flight ${this.options.flightId} destroyed`);
    }
}

/**
 * 高性能3D轨迹可视化器
 * High-performance 3D trajectory visualizer
 */
class HighPerformanceTrajectoryVisualizer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container not found: ${containerId}`);
        }

        this.options = {
            enable3D: true,
            enablePerformanceMode: true,
            maxPoints: 1000,
            colors: {
                trajectory: '#00ff88',
                prediction: '#ff4444',
                aircraft: '#ffffff'
            },
            ...options
        };

        this.trajectoryData = [];
        this.predictionData = [];
        this.renderCache = new Map();
        this.needsUpdate = false;

        this._initializeVisualizer();
    }

    _initializeVisualizer() {
        // 如果有Plotly，使用Plotly
        if (window.Plotly) {
            this._initializePlotly();
        } else {
            this._initializeCanvas();
        }

        console.log('HighPerformanceTrajectoryVisualizer initialized');
    }

    _initializePlotly() {
        this.plotType = 'plotly';
        this.initialLayout = {
            scene: {
                xaxis: { title: 'Longitude' },
                yaxis: { title: 'Latitude' },
                zaxis: { title: 'Altitude (m)' },
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 0.5 }
                }
            },
            showlegend: true,
            margin: { l: 0, r: 0, t: 0, b: 0 }
        };

        this._createPlotlyPlot();
    }

    _initializeCanvas() {
        this.plotType = 'canvas';
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
        this.container.appendChild(this.canvas);

        this.ctx = this.canvas.getContext('2d');
        this._setupCanvasEvents();
    }

    _createPlotlyPlot() {
        const traces = [];

        if (this.options.enable3D) {
            traces.push({
                type: 'scatter3d',
                mode: 'lines+markers',
                x: [],
                y: [],
                z: [],
                name: '轨迹',
                line: { color: this.options.colors.trajectory, width: 4 },
                marker: { size: 2 }
            });

            traces.push({
                type: 'scatter3d',
                mode: 'lines+markers',
                x: [],
                y: [],
                z: [],
                name: '预测轨迹',
                line: { color: this.options.colors.prediction, width: 3, dash: 'dash' },
                marker: { size: 2 }
            });

            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: [],
                y: [],
                z: [],
                name: '当前位置',
                marker: {
                    color: this.options.colors.aircraft,
                    size: 8,
                    symbol: 'circle'
                }
            });
        } else {
            // 2D plot
            traces.push({
                type: 'scatter',
                mode: 'lines+markers',
                x: [],
                y: [],
                name: '轨迹',
                line: { color: this.options.colors.trajectory, width: 3 }
            });

            traces.push({
                type: 'scatter',
                mode: 'lines+markers',
                x: [],
                y: [],
                name: '预测轨迹',
                line: { color: this.options.colors.prediction, width: 2, dash: 'dash' }
            });
        }

        Plotly.newPlot(this.container, traces, this.initialLayout, {
            responsive: true,
            displayModeBar: true
        });
    }

    updateTrajectory(trajectoryData) {
        this.trajectoryData = this._processData(trajectoryData);
        this.needsUpdate = true;
        this._scheduleRender();
    }

    updatePrediction(predictionData) {
        this.predictionData = this._processPredictionData(predictionData);
        this.needsUpdate = true;
        this._scheduleRender();
    }

    _processData(data) {
        if (!Array.isArray(data)) {
            return [];
        }

        // 限制数据点数量
        if (data.length > this.options.maxPoints) {
            const step = Math.floor(data.length / this.options.maxPoints);
            return data.filter((_, index) => index % step === 0);
        }

        return data;
    }

    _processPredictionData(data) {
        if (!data || !data.prediction) {
            return { trajectory: [], confidence: 0 };
        }

        return {
            trajectory: data.prediction,
            confidence: data.confidence?.overall || 0
        };
    }

    _scheduleRender() {
        if (this.renderScheduled) {
            return;
        }

        this.renderScheduled = true;
        requestAnimationFrame(() => {
            this._render();
            this.renderScheduled = false;
        });
    }

    _render() {
        if (!this.needsUpdate) {
            return;
        }

        if (this.plotType === 'plotly') {
            this._renderPlotly();
        } else {
            this._renderCanvas();
        }

        this.needsUpdate = false;
    }

    _renderPlotly() {
        const traces = [];

        if (this.options.enable3D) {
            // 3D轨迹
            if (this.trajectoryData.length > 0) {
                const trajectory = this.trajectoryData;
                traces.push({
                    type: 'scatter3d',
                    mode: 'lines+markers',
                    x: trajectory.map(p => p.longitude),
                    y: trajectory.map(p => p.latitude),
                    z: trajectory.map(p => p.altitude || p.geo_altitude || 0),
                    name: '轨迹',
                    line: { color: this.options.colors.trajectory, width: 4 },
                    marker: { size: 2 }
                });

                // 当前位置
                const lastPoint = trajectory[trajectory.length - 1];
                if (lastPoint) {
                    traces.push({
                        type: 'scatter3d',
                        mode: 'markers',
                        x: [lastPoint.longitude],
                        y: [lastPoint.latitude],
                        z: [lastPoint.altitude || lastPoint.geo_altitude || 0],
                        name: '当前位置',
                        marker: {
                            color: this.options.colors.aircraft,
                            size: 8,
                            symbol: 'circle'
                        }
                    });
                }
            }

            // 预测轨迹
            if (this.predictionData.trajectory && this.predictionData.trajectory.length > 0) {
                const prediction = this.predictionData.trajectory;
                const lastTrajectoryPoint = this.trajectoryData[this.trajectoryData.length - 1];

                // 添加连接点
                let xData = [], yData = [], zData = [];

                if (lastTrajectoryPoint) {
                    xData.push(lastTrajectoryPoint.longitude);
                    yData.push(lastTrajectoryPoint.latitude);
                    zData.push(lastTrajectoryPoint.altitude || lastTrajectoryPoint.geo_altitude || 0);
                }

                xData.push(...prediction.map(p => p[1])); // longitude
                yData.push(...prediction.map(p => p[0])); // latitude
                zData.push(...prediction.map(p => p[2])); // geo_altitude

                traces.push({
                    type: 'scatter3d',
                    mode: 'lines+markers',
                    x: xData,
                    y: yData,
                    z: zData,
                    name: `预测轨迹 (置信度: ${(this.predictionData.confidence * 100).toFixed(1)}%)`,
                    line: { color: this.options.colors.prediction, width: 3, dash: 'dash' },
                    marker: { size: 2 }
                });
            }
        } else {
            // 2D轨迹
            if (this.trajectoryData.length > 0) {
                const trajectory = this.trajectoryData;
                traces.push({
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: trajectory.map(p => p.longitude),
                    y: trajectory.map(p => p.latitude),
                    name: '轨迹',
                    line: { color: this.options.colors.trajectory, width: 3 }
                });
            }

            // 预测轨迹
            if (this.predictionData.trajectory && this.predictionData.trajectory.length > 0) {
                const prediction = this.predictionData.trajectory;
                const lastTrajectoryPoint = this.trajectoryData[this.trajectoryData.length - 1];

                let xData = [], yData = [];

                if (lastTrajectoryPoint) {
                    xData.push(lastTrajectoryPoint.longitude);
                    yData.push(lastTrajectoryPoint.latitude);
                }

                xData.push(...prediction.map(p => p[1])); // longitude
                yData.push(...prediction.map(p => p[0])); // latitude

                traces.push({
                    type: 'scatter',
                    mode: 'lines+markers',
                    x: xData,
                    y: yData,
                    name: `预测轨迹 (置信度: ${(this.predictionData.confidence * 100).toFixed(1)}%)`,
                    line: { color: this.options.colors.prediction, width: 2, dash: 'dash' }
                });
            }
        }

        Plotly.react(this.container, traces, this.initialLayout);
    }

    _renderCanvas() {
        // Canvas渲染实现（简化版）
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (this.trajectoryData.length > 0) {
            this._drawTrajectory();
        }

        if (this.predictionData.trajectory && this.predictionData.trajectory.length > 0) {
            this._drawPrediction();
        }
    }

    _drawTrajectory() {
        this.ctx.strokeStyle = this.options.colors.trajectory;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        const trajectory = this.trajectoryData;
        trajectory.forEach((point, index) => {
            const x = this._longitudeToX(point.longitude);
            const y = this._latitudeToY(point.latitude);

            if (index === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });

        this.ctx.stroke();
    }

    _drawPrediction() {
        this.ctx.strokeStyle = this.options.colors.prediction;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();

        const prediction = this.predictionData.trajectory;
        prediction.forEach((point, index) => {
            const x = this._longitudeToX(point[1]); // longitude
            const y = this._latitudeToY(point[0]); // latitude

            if (index === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        });

        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    _longitudeToX(lon) {
        // 简单的投影转换
        const minLon = Math.min(...this.trajectoryData.map(p => p.longitude));
        const maxLon = Math.max(...this.trajectoryData.map(p => p.longitude));
        return ((lon - minLon) / (maxLon - minLon)) * this.canvas.width;
    }

    _latitudeToY(lat) {
        // 简单的投影转换
        const minLat = Math.min(...this.trajectoryData.map(p => p.latitude));
        const maxLat = Math.max(...this.trajectoryData.map(p => p.latitude));
        return this.canvas.height - ((lat - minLat) / (maxLat - minLat)) * this.canvas.height;
    }

    _setupCanvasEvents() {
        // Canvas事件处理
        this.canvas.addEventListener('resize', () => {
            this.canvas.width = this.container.clientWidth;
            this.canvas.height = this.container.clientHeight;
            this.needsUpdate = true;
            this._scheduleRender();
        });
    }

    destroy() {
        if (this.plotType === 'plotly' && window.Plotly) {
            Plotly.purge(this.container);
        } else if (this.canvas) {
            this.container.removeChild(this.canvas);
        }

        this.renderCache.clear();
        console.log('HighPerformanceTrajectoryVisualizer destroyed');
    }
}

// 导出到全局
window.OptimizedFlightTrackingClient = OptimizedFlightTrackingClient;
window.HighPerformanceTrajectoryVisualizer = HighPerformanceTrajectoryVisualizer;