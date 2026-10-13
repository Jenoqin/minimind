import {sections,tokens,stages,terms,questions,tokenLoss,probability,initialDemo,demoTransition,gradeQuiz,updateCount} from './core.mjs';

const $=s=>document.querySelector(s);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const storageKey='minimind-lesson-1-v1';
let saved={completed:[],notes:''};
try{const loaded=JSON.parse(localStorage.getItem(storageKey)||'null');if(loaded&&typeof loaded==='object')saved={completed:Array.isArray(loaded.completed)?loaded.completed.filter(id=>sections.some(s=>s.id===id)):[],notes:typeof loaded.notes==='string'?loaded.notes:''};}catch{}
let current='map', stage=0, tokenPosition=4, demo=initialDemo(), records=null, recordIndex=783, chartMode='loss', answers={},submitted=false, selectedTerm=0, termPrinciples=null;
const save=()=>{try{localStorage.setItem(storageKey,JSON.stringify(saved));return true;}catch{return false;}};
function notify(message){$('#toast').textContent=message;$('#toast').classList.add('visible');clearTimeout(notify.timer);notify.timer=setTimeout(()=>$('#toast').classList.remove('visible'),2600);}
function icon(type){const paths={
  data:'<rect x="6" y="4" width="19" height="24" rx="2"/><path d="M11 10h9M11 15h9M11 20h6"/>',
  dial:'<path d="M5 24a13 13 0 1 1 22 0M16 17l8-8"/><circle cx="16" cy="17" r="2"/><path d="M5 24h22"/>',
  network:'<circle cx="6" cy="7" r="3"/><circle cx="6" cy="25" r="3"/><circle cx="16" cy="16" r="3"/><circle cx="27" cy="7" r="3"/><circle cx="27" cy="25" r="3"/><path d="m9 9 5 5m-5 9 5-5m5-4 5-5m-5 9 5 5"/>',
  backward:'<path d="M27 10H12a7 7 0 0 0 0 14h13M13 4l-6 6 6 6"/>',
  update:'<path d="m8 19 7-7 9 7M15 12v15"/><path d="M5 6h22"/>',
  chart:'<path d="M5 4v24h24M8 22l6-9 6 4 7-11"/>',
  save:'<path d="M5 4h19l5 5v19H5zM10 4v8h13V4M10 28V18h13v10"/>',
  repeat:'<path d="M26 11A11 11 0 0 0 7 7L3 11m0-7v7h7M6 21a11 11 0 0 0 19 4l4-4m0 7v-7h-7"/>',
};return `<svg viewBox="0 0 32 32" aria-hidden="true">${paths[type]}</svg>`;}
function header(id,title,description,kicker){return `<div class="page-head"><span class="eyebrow">LESSON 01 / ${kicker}</span><div class="head-line"><h1>${title}</h1><span class="time-label">60–90 MIN · 自由探索</span></div><p>${description}</p></div>`;}
function nextSection(){const i=sections.findIndex(s=>s.id===current),next=sections[i+1];return `<div class="section-next"><p>${saved.completed.includes(current)?'✓ 这一节已标记为理解':'完成探索后，用自己的话解释一次。'}</p><button class="primary-button" data-complete>${next?'我理解了，继续下一节 <span>→</span>':'标记本课已完成 <span>✓</span>'}</button></div>`;}
function renderNav(){
  $('#navigation').innerHTML=sections.map((s,i)=>`<a href="#${s.id}" class="nav-item ${s.id===current?'active':''}" ${s.id===current?'aria-current="page"':''}><span class="nav-num">0${i+1}</span><span>${s.title}</span><span class="nav-check">${saved.completed.includes(s.id)?'✓':''}</span></a>`).join('');
  $('#current-section').textContent=sections.find(s=>s.id===current).title;
  $('#progress-text').textContent=`已完成 ${saved.completed.length} / 6`;
  $('#progress-fill').style.width=`${saved.completed.length/6*100}%`;
  $('.progress-track').setAttribute('aria-valuenow',saved.completed.length);
}
function mapPage(){return header('map','一段文字，<br>如何变成<span class="text-accent">会预测的模型</span>？','把文本变成练习题，让模型预测下一枚 token，用真实文本判分，再根据误差调整参数。先看整张地图，再走进每一步。','THE BIG PICTURE')+
`<section class="surface" aria-label="预训练流程互动地图"><div class="panel-top"><div><div class="panel-title">一批数据的学习旅程</div><div class="panel-subtitle">点击任一步骤，查看它的作用与代码</div></div><span class="pill">A → H · 反复循环</span></div><div class="map-layout"><div class="map-board"><div class="map-intro"><span>准备完成</span><span>模型、数据、分词器、优化器已就位 ↓</span></div><div class="flow-grid">${stages.map((s,i)=>`<button class="flow-node ${i===stage?'selected':''}" data-stage="${i}" aria-pressed="${i===stage}"><span class="node-letter">${s.letter}</span><span class="node-icon">${icon(s.icon)}</span><strong>${s.name}</strong><small>${s.en}</small></button>`).join('')}</div><div class="map-legend"><span>当前探索的步骤</span><span>点击节点自由跳转</span></div><div class="map-controls"><span class="step-count">STEP ${String(stage+1).padStart(2,'0')} / 08</span><div><button class="soft-button" data-stage-prev ${stage===0?'disabled':''} aria-label="上一个训练步骤">←</button> <button class="soft-button" data-stage-next>${stage===7?'从头再走一遍':'下一步 →'}</button></div></div></div><div class="detail-panel" id="stage-detail" aria-live="polite">${stageDetail()}</div></div></section>
<div class="facts-strip" aria-label="本次真实训练配置">${[['63.91','M','可学习参数','Dense · 8 层'],['32','条','每个小批次','每条最多 340 个位置'],['8','批','通常更新一次','轮末余数也更新'],['2','轮','已完成预训练','RTX 4090 · 24 GB']].map(v=>`<div class="fact"><div class="fact-label">${v[2]}</div><div class="fact-value">${v[0]}<small>${v[1]}</small></div><div class="fact-note">${v[3]}</div></div>`).join('')}</div><details class="surface preparation"><summary>开始这个循环前，程序要准备什么？</summary><p class="muted" style="font-size:12px;margin-top:12px">先读脚本下方的 __main__，再读上方的 train_epoch。定义函数不会立即训练；主程序准备好对象后才调用它。</p><ol class="prep-steps">${[['环境与种子','确定计算设备，控制初始化与打乱等随机过程。跨设备不保证逐位一致。'],['结构与断点','lm_config 是结构说明书。--from_resume 1 检测已有续训记录。'],['混合精度','给部分计算选择数值精度。本次云端默认 bf16，不是把所有对象都变成低精度。'],['实验记录','SwanLab 记录指标，不负责更新模型。恢复时可沿用 run ID。'],['建立三个对象','模型负责预测，Dataset 供应样本，AdamW 提供参数更新规则；分词器负责文本编号。'],['恢复原来状态','恢复模型、优化器、epoch 和 step。没有已有权重与断点时才从随机参数开始。'],['可选加速与多卡','compile 优化执行，多卡包装负责协作。当前实验是单卡，默认未开启 compile。'],['开始逐轮训练','每轮安排样本顺序，按批次调用 train_epoch，恢复的第一轮跳过已完成批次。'],['结束后清理','所有计划批次结束，保留权重和记录；多卡场景释放进程组。']].map(v=>`<li><strong>${v[0]}</strong><p>${v[1]}</p></li>`).join('')}</ol></details>${nextSection()}`;}
function stageDetail(){const s=stages[stage];return `<div class="detail-number">${s.letter} / ${s.en}</div><h3>${s.title}</h3><p>${s.text}</p><pre class="code-block">${escape(s.code)}</pre><div class="detail-why"><strong>记住这一点</strong>${s.why}<div class="source-actions"><a href="assets/train_pretrain.py" download>查看完整注释脚本 ↗</a></div></div>`;}
function tokenPage(){return header('tokens','答案，就在<span class="text-accent">原文里</span>。','同一句话能提供很多道练习题。选择一个输入位置，看看模型允许看到的前文，以及它要预测的下一枚 token。','NEXT-TOKEN PREDICTION')+
`<section class="surface"><div class="panel-top"><div><div class="panel-title">春天到了，花开了。</div><div class="panel-subtitle">分词与编号来自本项目分词器 · 点击 token 或拖动位置</div></div><span class="pill">真实分词</span></div><div class="token-stage"><div class="tokens" id="token-sequence"></div><div class="token-legend"><span><i class="legend-key"></i>允许看到的前缀</span><span><i class="legend-key orange"></i>下一枚目标</span><span><i class="legend-key gray"></i>更后面的原文</span></div><div class="token-controls"><span class="range-label" id="position-label"></span><input id="token-position" type="range" min="0" max="9" value="${tokenPosition}" aria-label="选择输入 token 的位置"><button class="soft-button" id="token-next">下一位置 →</button></div><div class="token-result"><div class="result-box"><span class="result-label">模型可见的前文</span><div class="result-value" id="visible-prefix"></div></div><div class="result-box"><span class="result-label">原文给出的正确目标</span><div class="result-value target-value" id="target-token"></div></div></div><div class="explain-note" id="token-explanation"></div></div><div class="probability-lab" id="probability-lab"><div><span class="lesson-tag">教学示意 · 概率非真实模型输出</span><h3 style="margin-top:9px">给目标的概率更高，会怎样？</h3><p>拖动概率，观察一个目标位置的误差。分数经过归一化成为概率，这里直接从概率开始。</p><input id="probability" type="range" min="1" max="99" value="30" aria-label="正确目标的预测概率"></div><div class="loss-display" aria-live="polite"><div><small>目标概率</small><strong id="probability-value">30%</strong></div><div><small>该位置 loss = −ln(p)</small><strong class="loss-number" id="loss-value">1.204</strong></div></div></div></section>${nextSection()}`;}
function renderTokens(){
  $('#token-sequence').innerHTML=tokens.map((t,i)=>`<button class="token ${t.special?'special':''} ${i>tokenPosition+1?'future':''} ${i===tokenPosition+1?'target':''} ${i===tokenPosition?'selected':''}" ${i===10?'disabled':''} data-token="${i}" aria-label="${t.text}，编号 ${t.id}${i===tokenPosition?'，当前输入位置':i===tokenPosition+1?'，预测目标':''}">${t.text}<small>ID ${t.id}</small></button>`).join('');
  $('#position-label').textContent=`位置 ${tokenPosition+1} / 10`;
  $('#token-position').value=tokenPosition;
  $('#visible-prefix').textContent=tokenPosition===0?'BOS · 开始标记':`BOS · ${tokens.slice(1,tokenPosition+1).map(t=>t.text).join('')}`;
  const target=tokens[tokenPosition+1];$('#target-token').textContent=target.text==='PAD'?'PAD → 忽略（−100）':target.text==='EOS'?'EOS · 结束标记':target.text;
  $('#token-explanation').textContent=target.text==='PAD'?'文本已经结束。补齐用的 PAD 对应标签 −100，不参与 loss；模型不用学习输出这些占位符。':target.text==='EOS'?'“。”之后的目标是 EOS。保留结束标记，让模型学习一句文本何时结束。':'标签来自下一位置的原文。预测当前目标时，模型看不到目标和更后面的内容；因果注意力阻止偷看答案。';
  $('#probability-lab').hidden=target.text==='PAD';
}
function cyclePage(){return header('cycle','算出梯度，<br>参数就<span class="text-accent">改变了吗</span>？','亲手执行前向、反向和参数更新。留意参数数值：误差出现、梯度累积与参数改变，是三个不同的时刻。','ONE TRAINING CYCLE')+
`<div class="two-columns"><section class="surface"><div class="panel-top"><div><div class="panel-title">观察一次参数更新</div><div class="panel-subtitle">8 个小批次累积后，执行一次更新</div></div><span class="pill orange">教学演示</span></div><div class="training-demo"><div class="demo-stats"><div><small>可学习参数</small><strong id="demo-parameter"></strong></div><div><small>累积梯度</small><strong id="demo-gradient"></strong></div><div><small>已更新次数</small><strong id="demo-updates"></strong></div></div><div class="demo-state"><label id="batch-label"></label><div class="batch-slots">${Array.from({length:8},(_,i)=>`<span class="batch-slot" data-slot="${i}"></span>`).join('')}</div></div><div class="demo-steps"><button class="demo-action" data-demo="forward">① 前向预测<span>model(...)</span></button><button class="demo-action" data-demo="backward">② 计算梯度<span>backward()</span></button><button class="demo-action" data-demo="step">③ 更新参数<span>optimizer.step()</span></button></div><div class="demo-message" id="demo-message" role="status"></div><div class="demo-bottom"><button class="soft-button" data-demo="fill">补齐剩余批次 →</button><button class="text-button" data-demo="reset">重置演示</button></div></div></section><section class="surface frequency-list"><h3>这些动作，各有自己的节奏</h3>${[['每批','预测与反向','得到误差并累积梯度'],['每 8 批','通常更新一次','32 × 8 = 256 条样本'],['每 100 批','记录一次指标','当前批次的 loss'],['每 1000 批','保存一次成果','模型与续训检查点']].map(v=>`<div class="frequency-row"><span>${v[0]}</span><div>${v[1]}<small>${v[2]}</small></div></div>`).join('')}<div class="explain-note">轮末有余数时，也会更新、记录与保存。39,695 个小批次并不是 39,695 次更新。</div></section></div><div class="explain-note">为看清数值变化，这里用 1 个参数、二分类概率和简单梯度更新（步幅 0.8）示意，重复同一目标；正式训练使用 63.91M 参数与 AdamW。网页演示不会修改你的模型权重。</div>${nextSection()}`;}
function renderDemo(){
  $('#demo-parameter').textContent=demo.parameter.toFixed(3);$('#demo-gradient').textContent=demo.gradient.toFixed(3);$('#demo-updates').textContent=demo.updates;
  $('#batch-label').textContent=`累计 ${demo.batches} / 8 个小批次 · 目标概率 ${(probability(demo.parameter)*100).toFixed(1)}%${demo.loss!==null?' · 最近一次前向 loss '+demo.loss.toFixed(3):''}`;
  document.querySelectorAll('[data-slot]').forEach((e,i)=>e.classList.toggle('filled',i<demo.batches));
  $('#demo-message').textContent=demo.message;
  document.querySelectorAll('[data-demo]').forEach(e=>{const action=e.dataset.demo;e.disabled=action==='forward'?demo.phase!=='ready':action==='backward'?demo.phase!=='predicted':action==='step'?demo.phase!=='accumulated':action==='fill'?!(demo.phase==='ready'&&demo.batches>0&&demo.batches<8):false;e.classList.toggle('current',!e.disabled&&action!== 'fill'&&action!=='reset');});
}
function evidencePage(){return header('evidence','你完成的训练，<br>留下了<span class="text-accent">哪些证据</span>？','拖动位置，查看云端日志中的真实记录。曲线展示整体变化，每个记录点仍然只代表当时的一个小批次。','YOUR REAL EXPERIMENT')+
`<section class="surface"><div class="panel-top"><div><div class="panel-title">RTX 4090 · 两轮预训练记录</div><div class="panel-subtitle">784 个真实记录点 · 从续训后 step 1100 开始</div></div><div class="mode-tabs" role="group" aria-label="选择曲线指标"><button class="active" data-chart="loss" aria-pressed="true">Loss</button><button data-chart="lr" aria-pressed="false">学习率</button></div></div><div id="evidence-body" class="empty-state" role="status">正在读取训练记录…</div></section><div class="callout-grid"><div class="callout"><h3>看到波动，不急着判断退化</h3><p>第一轮末 loss 2.0311，第二轮首个记录为 2.1673。它们来自不同批次；整体下降，并不要求每个批次都更低。</p></div><div class="callout"><h3>训练完成 ≠ 所有回答都正确</h3><p>最后一条 loss 1.6360 是当前批次误差，不是正确率，也不是验证集成绩。本次完整训练没有独立验证集。</p></div></div>${nextSection()}`;}
async function loadEvidence(){
  try{if(!records){const response=await fetch('assets/training-data.json');if(!response.ok)throw new Error('load');records=await response.json();}if(current!=='evidence')return;
    recordIndex=Math.min(recordIndex,records.points.length-1);$('#evidence-body').className='';$('#evidence-body').innerHTML=`<div class="chart-wrap"><svg id="training-chart" class="chart" viewBox="0 0 900 260" role="img" aria-label="真实训练指标曲线"></svg><div class="chart-selector"><span class="range-label">日志位置</span><input type="range" id="record-position" min="0" max="${records.points.length-1}" value="${recordIndex}" aria-label="选择训练日志记录"><button class="soft-button" id="record-start">起点</button><button class="soft-button" id="record-end">终点</button></div></div><div class="record-detail"><div class="record-stats" aria-live="polite"><div><div class="fact-label">轮次 / 本轮批次</div><div class="fact-value" id="record-epoch"></div></div><div><div class="fact-label">全局小批次</div><div class="fact-value" id="record-global"></div></div><div><div class="fact-label">当前批次 loss</div><div class="fact-value" id="record-loss"></div></div><div><div class="fact-label">学习率</div><div class="fact-value" id="record-lr"></div></div></div><div class="raw-log" id="raw-log"></div><div class="source-actions"><span id="update-count"></span><a href="assets/pretrain.log" download>下载原始日志 ↓</a><a href="https://github.com/Jenoqin/minimind/blob/b51c2d3/trainer/train_pretrain.py" target="_blank" rel="noopener noreferrer">对照训练代码 ↗</a></div></div>`;renderChart();
  }catch{if(current==='evidence')$('#evidence-body').innerHTML='<div class="error-state">训练记录读取失败。请刷新页面重试，或直接<a href="assets/pretrain.log" download>下载原始日志</a>。</div>';}
}
function renderChart(){
  if(!records||current!=='evidence')return;const points=records.points,p=points[recordIndex],x=s=>55+s/79390*815,max=chartMode==='loss'?6:.0005,y=v=>219-v/max*194;
  const path=points.map((p,i)=>`${i?'L':'M'}${x(p.global).toFixed(2)},${y(p[chartMode]).toFixed(2)}`).join(' ');
  let svg='';for(let i=0;i<=4;i++){const v=max*i/4,yy=y(v);svg+=`<line x1="55" y1="${yy}" x2="870" y2="${yy}"/><text x="43" y="${yy+4}" text-anchor="end">${chartMode==='loss'?v.toFixed(1):v===0?'0':v.toExponential(1)}</text>`;}
  for(const n of [0,20000,40000,60000,79390])svg+=`<text x="${x(n)}" y="245" text-anchor="middle">${n===0?'0':n===79390?'79,390':n/1000+'k'}</text>`;
  svg+=`<text x="870" y="258" text-anchor="end">全局小批次</text><path d="${path}" fill="none" stroke="${chartMode==='loss'?'#6b8c59':'#d18a58'}" stroke-width="1.6"/><line class="selected-line" x1="${x(p.global)}" y1="16" x2="${x(p.global)}" y2="219"/><circle cx="${x(p.global)}" cy="${y(p[chartMode])}" r="5" fill="#d4663f" stroke="white" stroke-width="2"/>`;
  $('#training-chart').innerHTML=svg;$('#training-chart').setAttribute('aria-label',`${chartMode==='loss'?'Loss':'学习率'}真实训练曲线，当前全局批次 ${p.global}，指标 ${p[chartMode]}`);
  $('#record-position').value=recordIndex;$('#record-epoch').textContent=`${p.epoch}/2 · ${p.step.toLocaleString()}`;$('#record-epoch').style.fontSize='16px';$('#record-global').textContent=p.global.toLocaleString();$('#record-loss').textContent=p.loss.toFixed(4);$('#record-lr').textContent=p.lr.toFixed(8);$('#record-lr').style.fontSize='16px';$('#raw-log').textContent=`源文件第 ${p.line} 行 · ${p.raw}`;$('#update-count').textContent=`至此约 ${updateCount(p).toLocaleString()} 次参数更新`;
}
function termsPage(){return header('terms','从问题出发，<br>理解<span class="text-accent">九个概念</span>。','给一段前文，预测下一枚 token；看到真实目标后，调整模型。从这个最小任务出发，看看每个概念为什么必须出现。','FIRST PRINCIPLES')+
`<div class="concept-roadmap"><p><strong>表示与学习</strong> token → 参数 → logits → loss → 梯度 → 优化器</p><p><strong>组织过程</strong> batch：一次取哪些样本 · step：处理到第几批 · epoch：遍历了几轮</p></div><div class="concept-tabs" role="group" aria-label="选择要理解的概念">${terms.map((t,i)=>`<button data-term="${i}" aria-pressed="${selectedTerm===i}" aria-controls="principle-detail" class="concept-tab ${selectedTerm===i?'selected':''}"><span>0${i+1}</span>${t[1]}<small>${t[0]}</small></button>`).join('')}</div><section class="surface principle-detail" id="principle-detail" aria-label="概念第一性原理解释"><p class="empty-state" role="status">正在读取概念讲解…</p></section><details class="surface preparation"><summary>开始训练前，三个对象有什么不同？</summary><div class="prep-grid"><p><strong>lm_config · 结构说明书</strong><br>规定层数、维度等。它不是学到的知识。</p><p><strong>model · 预测器</strong><br>内部有可学习参数，前向时根据前文预测。</p><p><strong>optimizer · 更新规则</strong><br>使用梯度修改参数，并维护自身状态。创建它不等于已经开始学习。</p></div></details><div class="callout-grid"><div class="callout"><h3>训练</h3><p>目标来自原文。预测 → 计算 loss → backward → 参数更新。</p></div><div class="callout"><h3>推理</h3><p>用已学到的参数选出下一枚 token，追加后继续预测。普通生成不更新参数。</p></div></div>${nextSection()}`;}
const formatTerm=s=>escape(s).replace(/`([^`]+)`/g,'<code>$1</code>');
async function loadTermPrinciples(){
  try{if(!termPrinciples){const response=await fetch('assets/term-principles.json');if(!response.ok)throw new Error('load');termPrinciples=await response.json();}if(current==='terms')renderTermPrinciple();}
  catch{if(current==='terms')$('#principle-detail').innerHTML='<div class="error-state">概念讲解读取失败，请刷新重试，或<a href="assets/lesson-1.md" download>下载完整讲义</a>。</div>';}
}
function renderTermPrinciple(){
  if(current!=='terms'||!termPrinciples)return;
  const t=terms[selectedTerm],p=termPrinciples[String(selectedTerm+1)];
  $('#principle-detail').innerHTML=`<div class="principle-heading"><span class="eyebrow">0${selectedTerm+1} / ${t[0]}</span><h2>${escape(t[1])}：${escape(p.title.split('：').slice(1).join('：'))}</h2><p>${t[2]}</p></div><div class="principle-body"><div class="principle-section"><span class="principle-label">01 · 为什么需要它？</span><p>${formatTerm(p.why)}</p></div><div class="principle-section"><span class="principle-label">02 · 本质是什么？</span><p>${formatTerm(p.essence)}</p></div><div class="principle-example"><span class="principle-label">03 · 用一个小例子理解</span><p>${formatTerm(p.example)}</p></div><div class="principle-section"><span class="principle-label">04 · 回到这次训练</span><p>${formatTerm(p.connection)}</p><code class="principle-code">${escape(t[3])}</code></div><details class="principle-question"><summary><span>检验理解</span>${escape(p.question)}</summary><p>${formatTerm(p.answer)}</p></details></div>`;
  document.querySelectorAll('[data-term]').forEach(e=>{const active=Number(e.dataset.term)===selectedTerm;e.classList.toggle('selected',active);e.setAttribute('aria-pressed',active);});
}
function quizPage(){return header('quiz','你能自己<span class="text-accent">解释一遍</span>吗？','这五道题检查你是否把整个过程连起来了。答完后查看解释，再回到对应的互动环节验证。','CHECK YOUR UNDERSTANDING')+
`<form id="quiz-form"><div class="quiz-list">${questions.map((q,i)=>`<fieldset class="surface question ${submitted?(answers[i]===q.answer?'correct':'incorrect'):''}" id="question-${i}"><legend><span class="question-number">0${i+1}</span>${q.q}</legend><div class="question-content"><div class="answers">${q.options.map((a,j)=>`<label class="answer"><input type="radio" name="question-${i}" value="${j}" ${answers[i]===j?'checked':''} ${submitted?'disabled':''} required><span>${a}</span></label>`).join('')}</div>${submitted?`<p class="question-feedback">${answers[i]===q.answer?'✓ 理解正确。':'↗ 再想一想。'}${q.explain}</p>`:''}</div></fieldset>`).join('')}</div>${submitted?`<div class="quiz-result" role="status"><div><strong>${gradeQuiz(answers)} / 5</strong><p>${gradeQuiz(answers)===5?'你已把关键环节连起来了。再用自己的话讲一次。':'沿着题目解释，再去地图或演示中验证。'}</p></div><button type="button" class="soft-button" id="quiz-retry">重新练习 ↻</button></div>`:`<div class="quiz-actions"><span id="answered-count">已作答 ${Object.keys(answers).length} / 5</span><button type="submit" class="primary-button">检查我的答案 →</button></div>`}</form><div class="finish-panel"><h3>最后，用 3 分钟讲清这条链路</h3><p class="chain">文本 → 分词 → 输入与标签 → 预测 → loss → 梯度 → 参数更新 → 保存 → 生成</p><p>能讲清原因，并指到代码位置，就完成了第一课。第二课再深入 token 与标签移位。</p><div class="source-actions"><button class="text-button" data-open-notes>把自己的解释写进笔记 ↗</button><a href="assets/lesson-1.md" download>下载完整讲义 ↓</a></div></div>${nextSection()}`;}
function render(){
  current=sections.some(s=>s.id===location.hash.slice(1))?location.hash.slice(1):'map';renderNav();
  const views={map:mapPage,tokens:tokenPage,cycle:cyclePage,evidence:evidencePage,terms:termsPage,quiz:quizPage};$('#main').innerHTML=`<div class="content-enter">${views[current]()}</div>`;
  if(current==='tokens')renderTokens();if(current==='cycle')renderDemo();if(current==='evidence')loadEvidence();if(current==='terms')loadTermPrinciples();
  $('#sidebar').classList.remove('open');$('#menu-button').setAttribute('aria-expanded','false');
}
document.addEventListener('click',event=>{
  if(event.target.closest('.skip-link')){event.preventDefault();$('#main').focus();$('#main').scrollIntoView({block:'start'});return;}
  const b=event.target.closest('button');if(!b)return;
  if(b.dataset.stage!==undefined){stage=Number(b.dataset.stage);refreshMap();}
  if(b.hasAttribute('data-stage-prev')){stage=Math.max(0,stage-1);refreshMap();}
  if(b.hasAttribute('data-stage-next')){stage=(stage+1)%8;refreshMap();}
  if(b.dataset.token!==undefined){tokenPosition=Number(b.dataset.token);renderTokens();}
  if(b.id==='token-next'){tokenPosition=(tokenPosition+1)%10;renderTokens();}
  if(b.dataset.demo){demo=demoTransition(demo,b.dataset.demo);renderDemo();}
  if(b.dataset.chart){chartMode=b.dataset.chart;document.querySelectorAll('[data-chart]').forEach(e=>{e.classList.toggle('active',e===b);e.setAttribute('aria-pressed',e===b);});renderChart();}
  if(b.id==='record-start'){recordIndex=0;renderChart();}if(b.id==='record-end'){recordIndex=records.points.length-1;renderChart();}
  if(b.dataset.term!==undefined){selectedTerm=Number(b.dataset.term);renderTermPrinciple();}
  if(b.id==='quiz-retry'){answers={};submitted=false;render();}
  if(b.hasAttribute('data-complete')){
    if(current==='quiz'&&(!submitted||gradeQuiz(answers)<5)){notify('先完成五道题，并根据解释修正答案。');return;}
    if(!saved.completed.includes(current))saved.completed.push(current);save();renderNav();
    const i=sections.findIndex(s=>s.id===current);if(i<sections.length-1)location.hash=sections[i+1].id;else{notify('第一课已完成，记得留下自己的解释。');render();}
  }
  if(b.id==='menu-button'){const opened=$('#sidebar').classList.toggle('open');b.setAttribute('aria-expanded',opened);}
  if(b.id==='notes-toggle'||b.hasAttribute('data-open-notes'))openNotes();
  if(b.id==='notes-close')$('#notes-panel').close();
  if(b.id==='notes-download')downloadNotes();
});
function refreshMap(){document.querySelectorAll('[data-stage]').forEach(e=>{const selected=Number(e.dataset.stage)===stage;e.classList.toggle('selected',selected);e.setAttribute('aria-pressed',selected);});$('#stage-detail').innerHTML=stageDetail();$('.step-count').textContent=`STEP ${String(stage+1).padStart(2,'0')} / 08`;$('[data-stage-prev]').disabled=stage===0;$('[data-stage-next]').textContent=stage===7?'从头再走一遍':'下一步 →';}
document.addEventListener('input',event=>{
  const e=event.target;
  if(e.id==='token-position'){tokenPosition=Number(e.value);renderTokens();}
  if(e.id==='probability'){$('#probability-value').textContent=`${e.value}%`;$('#loss-value').textContent=tokenLoss(Number(e.value)/100).toFixed(3);}
  if(e.id==='record-position'){recordIndex=Number(e.value);renderChart();}
  if(e.id==='notes'){saved.notes=e.value;$('#notes-status').textContent=save()?'已自动保存到当前浏览器':'本地保存不可用，请下载笔记';}
});
document.addEventListener('change',event=>{if(event.target.name?.startsWith('question-')){answers[Number(event.target.name.split('-')[1])]=Number(event.target.value);if($('#answered-count'))$('#answered-count').textContent=`已作答 ${Object.keys(answers).length} / 5`;}});
document.addEventListener('submit',event=>{if(event.target.id==='quiz-form'){event.preventDefault();if(Object.keys(answers).length!==5){notify('请先完成所有题目。');return;}submitted=true;render();$('#question-0').scrollIntoView({behavior:'smooth',block:'start'});}});
function openNotes(){$('#notes').value=saved.notes;$('#notes-status').textContent=saved.notes?'已保存到当前浏览器':'尚未填写笔记';$('#notes-panel').showModal();$('#notes-toggle').setAttribute('aria-expanded','true');}
$('#notes-panel').addEventListener('close',()=>$('#notes-toggle').setAttribute('aria-expanded','false'));
function downloadNotes(){const content=`# MiniMind 第一课 · 我的学习笔记\n\n${saved.notes||'尚未填写笔记'}\n\n已标记理解：${saved.completed.map(id=>sections.find(s=>s.id===id).title).join('、')||'无'}\n`;const url=URL.createObjectURL(new Blob([content],{type:'text/markdown;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='MiniMind-第一课-我的笔记.md';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('笔记副本已开始下载。');}
window.addEventListener('hashchange',()=>{render();window.scrollTo({top:0,behavior:'instant'});$('#main').focus({preventScroll:true});});
document.addEventListener('keydown',event=>{if(event.key==='Escape'){$('#sidebar').classList.remove('open');$('#menu-button').setAttribute('aria-expanded','false');}});
render();
