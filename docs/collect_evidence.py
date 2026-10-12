#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预训练证据收集脚本：从 logs/pretrain.log 解析 loss/lr，绘制曲线 PNG，
并快照当前 GPU 状态与权重文件信息，全部输出到 docs/evidence/。
训练启动后任意时刻运行即可（可重复运行，文件自动覆盖）。
"""
import re
import os
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # minimind/
LOG = os.path.join(ROOT, 'logs', 'pretrain.log')
OUT = os.path.join(ROOT, 'docs', 'evidence')
os.makedirs(OUT, exist_ok=True)


def parse_log():
    """从训练日志解析 (step, loss, logits_loss, aux_loss, lr, epoch_time)"""
    pattern = re.compile(
        r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.]+), '
        r'logits_loss: ([\d.eE+-]+), aux_loss: ([\d.eE+-]+), lr: ([\deE.-]+), epoch_time: ([\d.]+)min'
    )
    steps, losses, lrs, etas = [], [], [], []
    if not os.path.exists(LOG):
        print(f'[!] 日志不存在: {LOG}')
        return steps, losses, lrs, etas
    with open(LOG, 'r', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch, iters, step = int(m.group(1)), int(m.group(4)), int(m.group(3))
                steps.append((epoch - 1) * iters + step)  # 换算为全局步数，曲线跨 epoch 连续
                losses.append(float(m.group(5)))
                lrs.append(float(m.group(8)))
                etas.append(float(m.group(9)))
    return steps, losses, lrs, etas


def plot_curves(steps, losses, lrs):
    if not steps:
        print('[!] 日志中还没有 loss 记录（每100步打一次），过一会儿再运行')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, losses, 'b-', linewidth=1)
    axes[0].axhline(y=8.46, color='r', linestyle='--', alpha=0.5, label='random init theory: ln(6400)~8.46')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('loss')
    axes[0].set_title('Pretrain Loss Curve'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(steps, lrs, 'g-', linewidth=1)
    axes[1].set_xlabel('step'); axes[1].set_ylabel('learning rate')
    axes[1].set_title('Cosine LR Schedule'); axes[1].grid(alpha=0.3)

    path = os.path.join(OUT, 'loss_lr_curve.png')
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f'[✓] 曲线图: {path}  (已记录 {len(steps)} 个日志点)')
    print(f'    最新: step={steps[-1]}, loss={losses[-1]:.4f}, lr={lrs[-1]:.2e}')


def snapshot_gpu():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw',
             '--format=csv'], capture_output=True, text=True, timeout=15)
        path = os.path.join(OUT, 'gpu_status.txt')
        with open(path, 'w') as f:
            f.write(f'# 采集时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n')
            f.write('# nvidia-smi --query-gpu=...\n')
            f.write(out.stdout)
        print(f'[✓] GPU 状态: {path}')
    except Exception as e:
        print(f'[!] nvidia-smi 采集失败: {e}')


def snapshot_weights():
    lines = [f'# 采集时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n']
    for sub in ('out', 'checkpoints'):
        d = os.path.join(ROOT, sub)
        lines.append(f'## {sub}/')
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                fp = os.path.join(d, fn)
                if os.path.isfile(fp):
                    lines.append(f'{fn}  {os.path.getsize(fp)/1e6:.1f} MB  {datetime.fromtimestamp(os.path.getmtime(fp)):%m-%d %H:%M}')
        else:
            lines.append('(目录不存在)')
        lines.append('')
    path = os.path.join(OUT, 'weight_files.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[✓] 权重文件清单: {path}')


if __name__ == '__main__':
    steps, losses, lrs, _ = parse_log()
    plot_curves(steps, losses, lrs)
    snapshot_gpu()
    snapshot_weights()
    print(f'\n全部证据输出目录: {OUT}')
