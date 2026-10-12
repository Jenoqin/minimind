# 第 1 课：先理解训练流程，再深入每个组件。
# 配套课程：docs/learning/01_预训练全流程地图.md
# 阅读顺序：先看下方 __main__（准备工作），再看 train_epoch（反复学习）。
# 核心链路：文本 → token/标签 → 模型预测 → loss → 梯度 → 参数更新 → 保存。
# 本脚本训练“预测下一个 token”；token 是分词器给文本划分的单位，不必等于一个字。
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 一轮 epoch：遍历这一轮安排的训练样本。
    # loader 每次给一小批数据；step 计数的是小批次，不是参数更新次数。
    # 定义这个函数时还不会训练；主程序准备好模型等对象后才会调用它。
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 【A 拿到练习材料】input_ids 是 token 编号，labels 是用来判分的目标。
        # 默认二者都是 [32, 340]：32 条样本，每条补齐/截断为 340 个位置。
        # labels 的 PAD 位置为 -100；预测与目标“错开一位”的操作在模型内部完成。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # 【B 设置学习步幅】按全局小批次位置计算学习率：前期较大，后期较小。
        # 学习率决定优化器更新参数时的步幅；这句本身还没有修改模型参数。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 【C 做预测并判分】model(...) 是一次前向计算。
        # 模型为每个位置输出对词表的分数 logits，再与下一个 token 的标签比较。
        # res.loss 是预测误差（交叉熵）；当前 Dense 模型的 res.aux_loss 为 0。
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            # 默认累计 8 个小批次的梯度：先把每次 loss 缩小，控制累加后的尺度。
            loss = loss / args.accumulation_steps

        # 【D 计算改进方向】backward 把误差传回参数，算出并累加梯度。
        # 梯度描述参数改变会怎样影响 loss；backward 本身不会更新参数。
        # scaler 处理启用时的数值缩放；默认 CUDA bf16 下它是禁用的。
        scaler.scale(loss).backward()

        # 【E 真正更新参数】默认每 8 个小批次更新一次；一轮最后的余数也会更新。
        # 通常一次更新处理 32 × 8 = 256 条样本，不是 256 个 token。
        if step % args.accumulation_steps == 0 or step == iters:
            # 如启用了梯度缩放，先恢复原尺度，再限制过大的全局梯度范数。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)  # 调用优化器：模型参数在这里改变。
            scaler.update()

            # 这次更新已经用完累积梯度，清零后开始下一组练习。
            optimizer.zero_grad(set_to_none=True)

        # 【F 记录学习情况】默认每 100 个小批次记一次，轮末也记一次。
        # current_loss 还原的是当前小批次的 loss，并非最近 100 步的平均值。
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 日志字段 epoch_time 表示本轮“预计还要多久”，不是已经运行多久。
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 【G 保存学习成果】默认每 1000 个小批次保存，轮末也保存。
        # out/ 保存模型权重供推理或 SFT 使用；checkpoints/ 还保存优化器和训练位置。
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # eval/train 切换模型模式；切换本身不是生成文本，也不是执行参数更新。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 文件中的权重转为 CPU 上的 fp16；这与训练时 autocast 的计算精度不同。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 【H 进入下一批】删除本批次的临时引用；已经更新的模型参数继续保留。
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # 直接运行这个文件时从这里开始。argparse 将命令行选项整理到 args 中。
    # 例如 --epochs 2 是看数据两轮，--batch_size 32 是一次取 32 条样本。
    # 默认相对路径按当前工作目录解释，通常要先进入 trainer/ 再启动。
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（DDP下每个rank为seed+rank，每轮为seed+epoch）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 单设备时无需建立分布式进程组；多 CUDA 卡时为各进程安排 GPU。
    # 随机种子控制初始化/打乱等随机过程，便于复查实验；不保证跨环境逐位一致。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # lm_config 是模型“结构说明书”，不是已经学到的参数。
    # --from_resume 1 会查找同名前缀的续训检查点；这里先读状态，稍后再恢复。
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # autocast 在前向计算时为部分算子选择低精度，兼顾速度、内存和数值表现。
    # CPU 分支不用 autocast；CUDA/MPS 使用各自的上下文。暂时先把它当计算设置。
    device_type = args.device.split(":")[0]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=dtype)
    
    # ========== 4. 配wandb ==========
    # --use_wandb 实际接入 SwanLab，负责保存和展示指标，不负责更新模型参数。
    # 续训时沿用检查点中的 run ID，让两次启动的记录归到同一个实验。
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # model：包含可学习参数的预测器；tokenizer：把文本换成 token 编号的分词器。
    # from_weight='none' 时随机初始化模型；如果后面恢复检查点，这些初值会被覆盖。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # Dataset 读取 JSONL，并在取样时生成 token、BOS/EOS、PAD 和 labels。
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 这里只在 CUDA + float16 时启用梯度缩放；默认 bfloat16 下不启用。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16' and device_type == 'cuda'))
    # AdamW 是更新参数的规则，维护自己的状态；“建好优化器”还不等于开始学习。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    # 模型权重回答“学到了什么”；优化器状态和 epoch/step 回答“训练走到哪里”。
    # 没有检查点就从第 0 轮开始；恢复时 epoch 从 0 计数，step 是本轮已处理批数。
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # 这两项服务于计算速度和多卡协作；第 1 课先知道作用，无需展开实现。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    # 外层循环按轮遍历数据，内层 train_epoch 按小批次反复预测、判分和更新。
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        # 单设备时生成本轮打乱后的样本序号；分布式时使用 train_sampler 安排样本。
        setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 恢复的第一轮跳过已完成的小批次；下一轮则从头遍历新排列。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # DataLoader 按批次取样并组成张量；num_workers 指数据加载子进程数量。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # 所有计划批次结束后释放多卡进程组；单设备时无需执行这部分。
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
