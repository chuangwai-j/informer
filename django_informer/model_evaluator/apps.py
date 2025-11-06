"""
应用初始化模块
Application initialization module for the optimized real-time system

Author: Claude Code
"""

import os
from django.apps import AppConfig
from django.core.management import call_command
import asyncio
import logging

logger = logging.getLogger(__name__)


class ModelEvaluatorConfig(AppConfig):
    """模型评估器应用配置"""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'model_evaluator'
    verbose_name = '高性能实时飞行预测系统'

    def ready(self):
        """应用准备就绪时初始化"""
        # 避免在迁移和加载fixture时初始化
        if os.environ.get('RUN_MAIN'):
            try:
                self._initialize_async_components()
            except Exception as e:
                logger.error(f"应用初始化失败: {e}")

    def _initialize_async_components(self):
        """初始化异步组件"""
        try:
            logger.info("初始化高性能实时系统组件...")

            # 延迟初始化，确保Django完全加载
            from django.utils.module_loading import import_string
            from asgiref.sync import async_to_sync

            def async_init():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def init_components():
                    try:
                        # 初始化预测引擎
                        from .optimized_prediction_engine import initialize_prediction_engine
                        await initialize_prediction_engine()
                        logger.info("预测引擎初始化完成")

                        # 初始化数据处理器
                        from .data_pipeline import flight_data_processor
                        await flight_data_processor.start()
                        logger.info("数据处理器启动完成")

                        # 初始化数据流处理器
                        from .data_loader import realtime_data_streamer
                        await realtime_data_streamer.start_streaming()
                        logger.info("实时数据流处理器启动完成")

                    except Exception as e:
                        logger.error(f"异步组件初始化失败: {e}")

                loop.run_until_complete(init_components())
                loop.close()

            # 在新线程中启动异步初始化
            import threading
            init_thread = threading.Thread(target=async_init, daemon=True)
            init_thread.start()

        except Exception as e:
            logger.error(f"异步组件初始化启动失败: {e}")


# 为了避免循环导入，将初始化函数放在这里
async def initialize_system_components():
    """初始化系统组件"""
    try:
        logger.info("开始初始化系统组件...")

        # 初始化预测引擎
        from .optimized_prediction_engine import initialize_prediction_engine
        await initialize_prediction_engine()

        # 初始化数据处理器
        from .data_pipeline import flight_data_processor
        await flight_data_processor.start()

        # 初始化数据流处理器
        from .data_loader import realtime_data_streamer
        await realtime_data_streamer.start_streaming()

        logger.info("系统组件初始化完成")

    except Exception as e:
        logger.error(f"系统组件初始化失败: {e}")
        raise


async def shutdown_system_components():
    """关闭系统组件"""
    try:
        logger.info("开始关闭系统组件...")

        # 关闭预测引擎
        from .optimized_prediction_engine import shutdown_prediction_engine
        await shutdown_prediction_engine()

        # 关闭数据处理器
        from .data_pipeline import flight_data_processor
        await flight_data_processor.stop()

        # 关闭数据流处理器
        from .data_loader import realtime_data_streamer
        await realtime_data_streamer.stop_streaming()

        logger.info("系统组件关闭完成")

    except Exception as e:
        logger.error(f"系统组件关闭失败: {e}")