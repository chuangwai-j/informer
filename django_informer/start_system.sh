#!/bin/bash
# Django高性能实时飞机轨迹预测系统启动脚本

echo "🚀 启动Django实时飞机轨迹预测系统..."

# 进入项目目录
cd /home/wai/Informer/django_informer

# 检查虚拟环境
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pct" ]; then
    echo "📦 激活虚拟环境..."
    source /home/wai/miniconda3/etc/profile.d/conda.sh
    conda activate pct
fi

# 清理缓存文件
echo "🧹 清理缓存文件..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 数据库迁移（如果需要）
echo "🗄️ 检查数据库迁移..."
python manage.py makemigrations model_evaluator --noinput
python manage.py migrate --noinput

# 启动服务器
echo "🌟 启动Django开发服务器..."
echo "📱 访问地址: http://127.0.0.1:8000/optimized-realtime/"
echo "📊 API文档: http://127.0.0.1:8000/api/"
echo "🛑 按 Ctrl+C 停止服务器"
echo ""

python manage.py runserver