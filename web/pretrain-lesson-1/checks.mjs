import assert from 'node:assert/strict';
import fs from 'node:fs';
import {tokens,stages,questions,tokenLoss,initialDemo,demoTransition,probability,gradeQuiz,updateCount} from './dist/core.mjs';

assert.equal(stages.length,8);
assert.deepEqual(tokens.map(t=>t.id),[1,3102,993,4520,294,1884,727,423,302,2,0]);
assert.equal(tokenLoss(.5),Math.log(2));
let s=initialDemo();
assert.deepEqual(demoTransition(s,'step'),s,'禁止无梯度更新');
assert.deepEqual(demoTransition(s,'backward'),s,'先前向再反向');
s=demoTransition(s,'forward');assert.equal(s.parameter,0);assert.equal(s.gradient,0);
s=demoTransition(s,'backward');assert.equal(s.batches,1);assert.equal(s.parameter,0);assert.equal(s.gradient,-.0625);
assert.equal(demoTransition(s,'step').parameter,0,'未满 8 批不能更新');
s=demoTransition(s,'fill');assert.equal(s.batches,8);assert.equal(s.gradient,-.5);assert.equal(s.parameter,0);
s=demoTransition(s,'step');assert.equal(s.parameter,.4);assert.equal(s.gradient,0);assert.equal(s.batches,0);assert.equal(s.updates,1);
assert(probability(s.parameter)>.5);
const complete={};questions.forEach((q,i)=>complete[i]=q.answer);assert.equal(gradeQuiz(complete),5);assert.equal(gradeQuiz({}),0);
const data=JSON.parse(fs.readFileSync('dist/assets/training-data.json','utf8'));
assert.equal(data.points.length,784);
assert.equal(data.points.at(-1).loss,1.636);
assert.equal(updateCount(data.points.at(-1)),9924);
assert.equal(updateCount(data.points.find(p=>p.epoch===1&&p.step===39695)),4962);
assert.equal(updateCount(data.points[0]),137);
for(const p of data.points){assert(Number.isFinite(p.loss)&&p.loss>0);assert(p.global>0&&p.global<=79390);}
for(const asset of ['index.html','styles.css','app.js','core.mjs','assets/train_pretrain.py','assets/pretrain.log','assets/lesson-1.md','assets/term-principles.json'])assert(fs.existsSync('dist/'+asset));
const principles=JSON.parse(fs.readFileSync('dist/assets/term-principles.json','utf8'));
assert.equal(Object.keys(principles).length,9);
for(const principle of Object.values(principles))for(const field of ['why','essence','example','connection','question','answer'])assert(principle[field]?.trim(),`概念缺少 ${field}`);
console.log('通过：token 编号、前向/梯度/更新状态、累积边界、自测评分、784 条真实日志、九个概念的完整讲解与资源完整性。');
