export const sections = [
  {id:'map', title:'训练地图', label:'整个过程发生了什么'},
  {id:'tokens', title:'一句话的练习题', label:'答案就藏在原文里'},
  {id:'cycle', title:'亲手更新参数', label:'从误差到一次改变'},
  {id:'evidence', title:'读懂真实记录', label:'用证据还原训练'},
  {id:'terms', title:'九个概念的原理', label:'先问为什么需要它'},
  {id:'quiz', title:'检查我的理解', label:'合上讲义，试着解释'},
];
export const tokens = [
  {text:'BOS',id:1,special:true},{text:'春',id:3102},{text:'天',id:993},{text:'到了',id:4520},
  {text:'，',id:294},{text:'花',id:1884},{text:'开',id:727},{text:'了',id:423},{text:'。',id:302},
  {text:'EOS',id:2,special:true},{text:'PAD',id:0,special:true},
];
export const stages = [
  {letter:'A',name:'取一批数据',en:'INPUT',icon:'data',title:'先准备一小批练习题',text:'分词器把文本换成编号，Dataset 准备输入与标签，DataLoader 把它们组成一批。默认一次取 32 条，每条补齐或截断到 340 个位置。',code:'input_ids = input_ids.to(args.device)\nlabels = labels.to(args.device)',why:'标签来自原文中的下一枚 token；PAD 的标签设为 -100，不参与判分。',ref:'dataset/lm_dataset.py · PretrainDataset'},
  {letter:'B',name:'设置步幅',en:'LEARNING RATE',icon:'dial',title:'决定这次更新的步幅',text:'学习率控制优化器更新参数时的步幅。本次从约 0.0005 逐渐下降到 0.00005。设置学习率这一步，还没有改变参数。',code:'lr = get_lr(global_step, total_steps, args.learning_rate)\n# 原脚本用 epoch * iters + step 计算全局位置\nparam_group["lr"] = lr',why:'前期步幅较大，后期较小。它按小批次位置调度，没有 warmup。',ref:'trainer_utils.py · get_lr'},
  {letter:'C',name:'预测与判分',en:'FORWARD',icon:'network',title:'预测下一枚 token，再计算误差',text:'模型为每个位置的 6400 个候选 token 输出分数 logits，再用下一个位置的原文计算交叉熵 loss。正确目标的概率越低，误差代价越大。',code:'res = model(input_ids, labels=labels)\nloss = res.loss + res.aux_loss\nloss = loss / args.accumulation_steps',why:'当前 Dense 模型 aux_loss 为 0。loss 衡量文本预测，不直接判断知识真假。',ref:'model_minimind.py · MiniMindForCausalLM.forward'},
  {letter:'D',name:'计算梯度',en:'BACKWARD',icon:'backward',title:'算出改进方向，参数还没变',text:'backward 把误差传回参数，计算并累加梯度。你可以先把它理解为：为了降低当前误差，各个参数应该怎样调整？',code:'scaler.scale(loss).backward()',why:'backward 本身不会修改参数。默认 bf16 下 scaler 禁用，没有梯度缩放。',ref:'train_pretrain.py · train_epoch'},
  {letter:'E',name:'更新参数',en:'OPTIMIZER',icon:'update',title:'模型在这里真正改变',text:'通常累积 8 个小批次后，先恢复梯度尺度、再裁剪梯度，最后由 AdamW 更新参数。用完的梯度清零，下一组重新积累。',code:'scaler.unscale_(optimizer)\nclip_grad_norm_(model.parameters(), args.grad_clip)\nscaler.step(optimizer)\noptimizer.zero_grad(set_to_none=True)',why:'通常一次更新处理 32 × 8 = 256 条样本。轮末不足 8 批也会更新。',ref:'train_pretrain.py · train_epoch'},
  {letter:'F',name:'记录指标',en:'LOG',icon:'chart',title:'把学习过程留下来',text:'默认每 100 个小批次记录一次 loss、学习率和本轮剩余时间，轮末也记录。SwanLab 将这些记录展示为曲线。',code:'current_loss = loss.item() * args.accumulation_steps\nLogger(f"loss: {current_loss:.4f}, ...")\n# 这里只展示核心语句，完整字段见源文件',why:'记录的是当前批次 loss，不是最近 100 步的平均值，也不是正确率。',ref:'logs/pretrain.log · 784 条指标'},
  {letter:'G',name:'保存成果',en:'CHECKPOINT',icon:'save',title:'保存学到的内容与训练位置',text:'out/ 保存用于推理或 SFT 的模型权重。checkpoints/ 还保存优化器和训练位置，让下次可以恢复。默认每 1000 个小批次保存，轮末也保存。',code:'torch.save(state_dict, ckp)\nlm_checkpoint(..., model=model, optimizer=optimizer,\n              epoch=epoch, step=step)\n# 以上为简化展示；实际权重转为 CPU fp16 保存',why:'模型权重回答“学到了什么”；续训检查点还回答“练习进行到了哪里”。',ref:'trainer_utils.py · lm_checkpoint'},
  {letter:'H',name:'继续下一批',en:'REPEAT',icon:'repeat',title:'用更新后的模型，做下一组练习',text:'这一批的临时数据引用被清理，已经更新的参数留下来。取下一批、预测、判分、计算梯度……直到走完计划中的所有轮次。',code:'del input_ids, labels, res, loss\n# for 循环继续取下一批',why:'本次遍历两轮，共处理 79,390 个小批次，更新参数 9,924 次。',ref:'最终检查点 · epoch=1, step=39695'},
];
export const terms = [
  ['TOKEN','token','分词器划出的文本单位，模型拿到它的整数编号。','input_ids','“到了”是一枚 token，不一定一个字就是一枚 token。'],
  ['PARAMETER','参数','模型通过训练调整的数值。','model.parameters()','结构和词表固定，本次主要学习的是约 6391 万个参数。'],
  ['LOGITS','预测分数','每个位置对词表中候选 token 给出的分数。','res.logits','logits 尚未归一化成概率；交叉熵直接接收这些分数。'],
  ['LOSS','预测误差','正确目标的概率不够高时，付出的误差代价。','res.loss','loss 小表示更贴近训练文本，不能直接证明知识可靠。'],
  ['GRADIENT','梯度','参数变化怎样影响当前 loss 的方向信息。','loss.backward()','backward 计算梯度，本身没有改变参数。'],
  ['OPTIMIZER','优化器','利用梯度和自身状态来更新参数的规则。','AdamW / optimizer.step()','参数在优化器 step 时改变；优化器也保存自己的状态。'],
  ['BATCH','小批次','一次取出的多条训练样本。','batch_size = 32','正式输入一般为 [32, 340]，最后一批可能不足 32 条。'],
  ['STEP','批次位置','本脚本已经处理了多少个小批次。','step / iters','8 个小批次通常才更新一次，所以 step 不等于更新次数。'],
  ['EPOCH','一轮','按本轮安排遍历一次数据集。','epochs = 2','每轮重新打乱样本顺序；检查点的 epoch 从 0 计数。'],
];
export const questions = [
  {q:'哪一步真正修改模型参数？',options:['model(...) 前向计算','loss.backward() 反向计算','optimizer.step() 参数更新','打印 loss 日志'],answer:2,explain:'前向产生预测与误差，backward 计算梯度，优化器 step 才修改参数。'},
  {q:'把 log_interval 从 100 改成 10，会发生什么？',options:['模型会多学习十倍','日志记录更加频繁','模型参数量变小','每次输入变成 10 条样本'],answer:1,explain:'log_interval 控制记录频率，不控制样本数或参数更新次数；更频繁记录也会带来开销。'},
  {q:'只有前向和 loss，没有 backward 与 step，模型学会新东西了吗？',options:['学会了，只要 loss 存在','学会了，前向会自动更新','还没有进行参数学习','取决于是否开启 SwanLab'],answer:2,explain:'这种情况下只计算了现有模型的表现，没有通过优化器改变模型参数。'},
  {q:'本次 79,390 个小批次，对应多少次参数更新？',options:['79,390 次','9,924 次','39,695 次','2 次'],answer:1,explain:'每轮 39,695 = 4,961 × 8 + 7。末尾 7 批也更新，所以每轮 4,962 次，两轮 9,924 次。最终优化器状态与此一致。'},
  {q:'只有 out/pretrain_768.pth，能够做什么？',options:['可以生成文本，但缺少恢复原优化轨迹所需状态','只能继续训练，不能生成文本','同时包含完整日志与优化器状态','可以证明所有回答都是正确的'],answer:0,explain:'out/ 权重可以加载到同一结构中推理；恢复原训练状态还需要优化器、训练位置等检查点内容。'},
];
export const tokenLoss = p => -Math.log(p);
export const probability = parameter => 1 / (1 + Math.exp(-parameter));
export function initialDemo(){return {parameter:0,gradient:0,batches:0,updates:0,phase:'ready',loss:null,message:'先做一次前向预测。观察误差出现后，参数有没有改变。'};}
export function demoTransition(state, action){
  const s={...state};
  if(action==='reset')return initialDemo();
  if(action==='forward' && s.phase==='ready' && s.batches<8){s.loss=tokenLoss(probability(s.parameter));s.phase='predicted';s.message='完成前向：得到了预测概率与 loss。参数和梯度都还没有改变。';}
  if(action==='backward' && s.phase==='predicted'){
    s.gradient+=(probability(s.parameter)-1)/8;s.batches++;s.phase=s.batches===8?'accumulated':'ready';
    s.message=`已计算 ${s.batches} / 8 批梯度。参数仍然是 ${s.parameter.toFixed(3)}${s.batches===8?'，现在可以执行参数更新。':'，下一批继续累积。'}`;
  }
  if(action==='fill' && s.phase==='ready' && s.batches>0 && s.batches<8){
    s.gradient+=(8-s.batches)*(probability(s.parameter)-1)/8;s.batches=8;s.phase='accumulated';s.message='把剩余批次的前向与反向补齐：累计 8 批梯度，参数仍未改变。';
  }
  if(action==='step' && s.phase==='accumulated'){
    s.parameter-=0.8*s.gradient;s.gradient=0;s.batches=0;s.updates++;s.phase='ready';
    s.message=`第 ${s.updates} 次更新完成：参数变为 ${s.parameter.toFixed(3)}，预测目标的概率提升到 ${(100*probability(s.parameter)).toFixed(1)}%。梯度已清零。`;
  }
  return s;
}
export function gradeQuiz(answers){return questions.reduce((total,q,i)=>total+(answers[i]===q.answer?1:0),0);}
export function updateCount(point){return (point.epoch-1)*Math.ceil(point.total/8)+(point.step===point.total?Math.ceil(point.step/8):Math.floor(point.step/8));}
