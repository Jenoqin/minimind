#!/usr/bin/env python3
"""本地课程的启动/关闭控制器；仅使用 Python 标准库。"""

import argparse
import fcntl
import functools
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
import webbrowser


APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR / '.runtime'
STATE_FILE = RUNTIME_DIR / 'server.json'
LOG_FILE = RUNTIME_DIR / 'server.log'
PORT_FILE = RUNTIME_DIR / 'preferred_port.json'
LOCAL_HTTP = build_opener(ProxyHandler({}))


def read_state():
    try:
        state = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        if (state['app_dir'] == str(APP_DIR)
                and type(state['port']) is int and 1024 <= state['port'] <= 65535
                and isinstance(state['instance'], str)
                and isinstance(state['shutdown_token'], str)):
            return state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def base_url(state):
    return 'http://127.0.0.1:{}'.format(state['port'])


def is_our_server(state):
    if not state:
        return False
    try:
        with LOCAL_HTTP.open(base_url(state) + '/__minimind__/health', timeout=0.6) as response:
            info = json.load(response)
        return info.get('instance') == state['instance'] and info.get('app_dir') == str(APP_DIR)
    except (OSError, URLError, ValueError):
        return False


def write_state(state):
    # 状态在 dist/ 外，不会作为网页资源被下载；权限仅允许当前用户读取。
    temporary = STATE_FILE.with_suffix('.tmp')
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump(state, handle, ensure_ascii=False)
    temporary.replace(STATE_FILE)


def serve():
    instance = secrets.token_hex(16)
    shutdown_token = secrets.token_hex(32)

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/__minimind__/health':
                content = json.dumps({'instance': instance, 'app_dir': str(APP_DIR)}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                super().do_GET()

        def do_POST(self):
            # 只接受本实例的关闭凭证，不按端口或进程名称杀其他服务。
            if self.path != '/__minimind__/shutdown':
                self.send_error(404)
                return
            provided = self.headers.get('X-MiniMind-Shutdown', '')
            if not secrets.compare_digest(provided, shutdown_token):
                self.send_error(403)
                return
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    handler = functools.partial(Handler, directory=str(APP_DIR / 'dist'))
    server = None
    preferred = 8765
    try:
        previous = json.loads(PORT_FILE.read_text(encoding='utf-8'))
        if type(previous) is int and 8765 <= previous < 8785:
            preferred = previous
    except (OSError, ValueError):
        pass
    # 尽量复用同一端口，让同一浏览器中的学习笔记仍属于同一网页来源。
    ports = [preferred] + [port for port in range(8765, 8785) if port != preferred]
    for port in ports:
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), handler)
            break
        except OSError as error:
            # macOS/Linux 的 EADDRINUSE：端口占用时尝试下一个端口。
            import errno
            if error.errno != errno.EADDRINUSE:
                raise
    if server is None:
        raise RuntimeError('8765–8784 端口均被占用，请关闭不需要的本地服务后重试。')
    state = {'app_dir': str(APP_DIR), 'port': server.server_address[1],
             'pid': os.getpid(), 'instance': instance, 'shutdown_token': shutdown_token}
    write_state(state)
    PORT_FILE.write_text(json.dumps(state['port']), encoding='utf-8')
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        current = read_state()
        if current and current['instance'] == instance:
            STATE_FILE.unlink(missing_ok=True)


def start(open_browser=True):
    if not (APP_DIR / 'dist/index.html').is_file():
        raise RuntimeError('缺少 dist/index.html，请保留整个应用目录后再启动。')
    state = read_state()
    if not is_our_server(state):
        STATE_FILE.unlink(missing_ok=True)
        with LOG_FILE.open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'serve'],
                                     cwd=str(APP_DIR), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True,
                                     close_fds=True)
        for _ in range(100):
            state = read_state()
            if is_our_server(state):
                break
            if child.poll() is not None:
                raise RuntimeError('服务器启动失败，请查看 {}'.format(LOG_FILE))
            time.sleep(0.1)
        else:
            child.terminate()
            raise RuntimeError('服务器未能及时就绪，请查看 {}'.format(LOG_FILE))
        print('课程应用已启动。')
    else:
        print('课程应用已在运行，使用现有服务器。')
    url = base_url(state)
    print('访问地址：' + url)
    if open_browser and not webbrowser.open(url):
        print('浏览器未自动打开，请复制上面的地址访问。')
    print('学习结束后，双击「关闭课程.command」即可关闭服务器。')


def stop():
    state = read_state()
    if not is_our_server(state):
        STATE_FILE.unlink(missing_ok=True)
        print('课程应用当前没有运行。')
        return
    request = Request(base_url(state) + '/__minimind__/shutdown', data=b'', method='POST',
                      headers={'X-MiniMind-Shutdown': state['shutdown_token']})
    with LOCAL_HTTP.open(request, timeout=2) as response:
        if response.status != 200:
            raise RuntimeError('课程服务器未确认关闭。')
    for _ in range(50):
        if not is_our_server(state):
            print('课程服务器已关闭。学习笔记和进度仍保留在浏览器中。')
            return
        time.sleep(0.1)
    raise RuntimeError('服务器仍在关闭中，请稍后再次运行关闭脚本。')


def main():
    parser = argparse.ArgumentParser(description='MiniMind 本地课程一键启动与关闭')
    parser.add_argument('action', choices=['start', 'stop', 'serve'])
    parser.add_argument('--no-browser', action='store_true', help='启动时不打开浏览器')
    args = parser.parse_args()
    RUNTIME_DIR.mkdir(mode=0o700, exist_ok=True)
    if args.action == 'serve':
        serve()
        return
    # 两次快速双击也按顺序处理；进程退出时系统会自动释放锁。
    with (RUNTIME_DIR / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'start':
            start(open_browser=not args.no_browser)
        else:
            stop()


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, URLError) as error:
        print('操作未完成：{}'.format(error), file=sys.stderr)
        sys.exit(1)
