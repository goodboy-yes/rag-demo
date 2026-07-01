# 从 CLI 子模块导入 typer 应用实例
from .cli import app


# 允许通过 `python -m raglite` 直接运行此包作为 CLI 入口
# typer 应用对象本身是可调用的；传入 sys.argv 后会根据子命令名路由到对应函数（如 ingest / search / inspect）
if __name__ == "__main__":
    app()
