# Django Informer模型评估系统

基于Django框架构建的Informer飞机轨迹预测模型评估系统。

## 功能特性

- 📊 **模型信息查看**: 查看Informer模型的详细配置和参数信息
- 🚀 **实时模型评估**: 对训练好的模型进行性能评估
- 📈 **结果可视化**: 展示评估指标和历史趋势
- 🗄️ **数据存储**: 将评估结果保存到数据库
- 🔄 **API接口**: 提供RESTful API支持
- 📱 **响应式设计**: 支持多设备访问

## 系统架构

```
django_informer/
├── manage.py                     # Django管理脚本
├── django_informer/              # 项目配置
│   ├── settings.py              # 项目设置
│   ├── urls.py                  # 主URL配置
│   └── wsgi.py                  # WSGI配置
├── model_evaluator/             # 核心应用
│   ├── models.py                # 数据模型
│   ├── views.py                 # 视图逻辑
│   ├── services.py              # 模型评估服务
│   ├── urls.py                  # 应用URL
│   └── admin.py                 # 管理界面
├── templates/                   # ��板文件
├── static/                      # 静态文件
└── requirements.txt             # 依赖包
```

## 安装和运行

### 1. 环境要求

- Python 3.8+
- PyTorch
- Django 4.2+
- 其他依赖见requirements.txt

### 2. 安装依赖

```bash
cd django_informer
pip install -r requirements.txt
```

### 3. 数据库迁移

```bash
python manage.py makemigrations
python manage.py migrate
```

### 4. 创建超级用户（可选）

```bash
python manage.py createsuperuser
```

### 5. 运行服务器

```bash
python manage.py runserver
```

访问 http://127.0.0.1:8000 即可使用系统。

## 使用说明

### 1. 模型信息
- 查看Informer模型的参数量和配置信息
- 了解模型架构和训练参数

### 2. 模型评估
- 点击"开始评估"按钮启动模型评估
- 系统会自动加载模型和测试数据
- 显示MAE、MSE、RMSE等评估指标

### 3. 历史结果
- 查看所有历史评估记录
- 支持分页浏览
- 显示性能趋势图表

### 4. API接口

- **POST** `/api/evaluation/` - 执行模型评估
- **GET** `/api/jobs/` - 获取任务列表
- **GET** `/api/results/` - 获取评估统计

## 核心功能

### 模型评估服务 (services.py)

基于原始test.py的逻辑封装的评估服务：

```python
class ModelEvaluator:
    def load_config()          # 加载YAML配置
    def load_model()          # 加载Informer模型
    def get_test_dataloader() # 获取测试数据
    def evaluate_model()      # 执行模型评估
    def get_model_info()      # 获取模型信息
```

### 数据模型 (models.py)

- **EvaluationResult**: 评估结果记录
- **ModelInfo**: 模型信息记录
- **PredictionJob**: 预测任务管理

### Web界面

- **首页**: 系统概览和快捷入口
- **模型信息**: 详细的模型配置展示
- **模型评估**: 评估操作和结果查看

## 注意事项

1. **路径配置**: 确保模型检查点和配置文件路径正确
2. **依赖环境**: 需要安装Informer项目的所有依赖
3. **GPU支持**: 系统自动检测GPU可用性
4. **数据安全**: 评估结果保存在SQLite数据库中

## 技术特点

- ✅ **模块化设计**: 清晰的代码结构和功能分离
- ✅ **异步处理**: 支持后台任务执行
- ✅ **错误处理**: 完善的异常处理机制
- ✅ **数据持久化**: 评估结果永久保存
- ✅ **用户友好**: 直观的Web界面和操作流程

## 扩展功能

- [ ] 支持多模型比较
- [ ] 添加可视化图表
- [ ] 支持自定义数据集
- [ ] 添加用户权限管理
- [ ] 支持分布式评估

## 开发团队

基于Informer深度学习模型开发的飞机轨迹预测评估系统。