#!/bin/bash
# Finder 中双击即可启动；从任意工作目录运行也可。
LESSON_DIR="$(cd "$(dirname "$0")" && pwd)"
LESSON_PYTHON="$LESSON_DIR/../../.venv/bin/python"
if [ ! -x "$LESSON_PYTHON" ]; then
    LESSON_PYTHON="$(command -v python3)"
fi
if [ -z "$LESSON_PYTHON" ]; then
    echo "未找到 Python 3，请先安装 Python 3。"
    read -r -p "按回车结束…"
    exit 1
fi
"$LESSON_PYTHON" "$LESSON_DIR/local_server.py" start "$@"
LESSON_RESULT=$?
if [ "$LESSON_RESULT" -ne 0 ]; then
    read -r -p "按回车结束…"
fi
exit "$LESSON_RESULT"
