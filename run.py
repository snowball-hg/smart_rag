#!/usr/bin/env python3
"""启动脚本。

提供两种启动方式：
1. 直接运行本脚本: python run.py
2. 使用 uvicorn 命令: uvicorn app.main:app --reload
"""

from __future__ import annotations

import sys

import uvicorn

if __name__ == "__main__":
    # debugpy 的 runpy.run_path() 会在同进程内 exec 目标脚本，
    # 此时 debugpy 模块已在 sys.modules 中，以此判断是否在调试模式
    is_debugging = "debugpy" in sys.modules
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8100,
        reload=not is_debugging,
        log_level="info",
    )
