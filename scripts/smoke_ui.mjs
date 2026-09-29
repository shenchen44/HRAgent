/**
 * 前端冒烟测试 —— 把「声明了但没接线」第 10/11/12 例固化成断言。
 *
 * 为什么需要这个文件：前 9 例那一族 bug 全部落在有自动化回归的层
 * （`selftest_guards.py` 105 项、`verify_repro.py` 210 项、`smoke_graph.py`），
 * 所以被挡住了；而第 10~12 例**全部落在前端**，前端是唯一没有回归的一层，
 * 于是同一类错误在一个下午里连出三次。这不是巧合，是覆盖缺口的形状。
 *
 * 这个文件补的就是那个缺口。三条断言各自对应一个真实缺陷：
 *
 *   1. 路由前缀匹配   —— `/jobs/J01` 曾经打不开（hash 匹配不上，静默回落概览，
 *                        页面不报错、只是内容不对）
 *   2. 升级理由渲染   —— `escalation_reason` 曾经算出来了、传过来了、没人渲染
 *   3. 侧栏计数刷新   —— 计数曾经只在启动时取一次，问答后审计页 20 条、侧栏写 19
 *
 * **依赖说明（诚实交代）**：它需要 puppeteer-core 与一个 Chrome。
 * 本项目其余部分坚持零新依赖，这里是唯一的例外 —— 因为上述三个 bug
 * 全是 **DOM 层**的，用 stdlib 断言 HTTP 响应根本看不见它们。
 * 找不到 puppeteer 时本脚本**明确跳过并说明**，不伪装成通过。
 *
 * 用法：
 *   node scripts/smoke_ui.mjs                 # 只跑静态断言（快，不花 token）
 *   node scripts/smoke_ui.mjs --with-llm      # 外加助手链路（要调 LLM，慢）
 *   PUPPETEER_PATH=/path/to/puppeteer-core.js node scripts/smoke_ui.mjs
 */

import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';

const BASE = process.env.HR_UI_BASE || 'http://127.0.0.1:8765';
const WITH_LLM = process.argv.includes('--with-llm');

// ---------------------------------------------------------------- 依赖定位
// 按顺序找：显式环境变量 → 本项目 node_modules → 全局安装。
// **不写死本机路径** —— 那对别人无效，也会把个人目录结构带进公开仓库。
const CANDIDATES = [
  process.env.PUPPETEER_PATH,
  'puppeteer-core',
  'puppeteer',
].filter(Boolean);

let puppeteer = null;
for (const c of CANDIDATES) {
  try {
    if (c.startsWith('/') && !existsSync(c)) continue;
    puppeteer = c.startsWith('/') ? (await import(c)).default : createRequire(import.meta.url)(c);
    break;
  } catch { /* 换下一个 */ }
}

const CHROME = process.env.CHROME_PATH
  || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

if (!puppeteer || !existsSync(CHROME)) {
  console.log('='.repeat(70));
  console.log('前端冒烟测试 —— 已跳过');
  console.log('='.repeat(70));
  console.log(`  puppeteer-core : ${puppeteer ? '找到' : '未找到'}`);
  console.log(`  Chrome         : ${existsSync(CHROME) ? '找到' : '未找到'} (${CHROME})`);
  console.log();
  console.log('  本脚本需要 headless 浏览器，因为要断言的东西全在 DOM 层');
  console.log('  （路由前缀匹配、字段是否渲染、计数是否刷新）—— 用 stdlib');
  console.log('  断言 HTTP 响应看不见它们。装法：');
  console.log('    npm i puppeteer-core      # 或设 PUPPETEER_PATH 指向已有安装');
  console.log();
  console.log('  这不是通过，是没跑。UI 层当前**无自动化验证**。');
  process.exit(0);
}

// ---------------------------------------------------------------- 断言框架
let pass = 0, fail = 0;
const ok = (name, cond, extra = '') => {
  cond ? pass++ : fail++;
  console.log(`  ${cond ? '✅' : '❌'} ${name}${extra ? '  — ' + extra : ''}`);
};

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: 'new',
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

async function open(hash, settle = 1600) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000 });
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  page.on('console', (m) => m.type() === 'error' && errs.push(m.text()));
  await page.goto(`${BASE}/${hash}`, { waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, settle));
  return { page, errs };
}

console.log('='.repeat(70));
console.log('前端冒烟测试');
console.log('='.repeat(70));

// ------------------------------------------------- 1. 全部路由可达（第 10 例的近亲）
console.log('\n[1] 路由渲染');
const ROUTES = [
  ['#/', '概览'], ['#/jobs', '岗位管理'], ['#/candidates', '候选人'],
  ['#/match', '智能匹配'], ['#/assistant', 'AI 助手'], ['#/audit', '审计日志'],
];
for (const [hash, expect] of ROUTES) {
  const { page, errs } = await open(hash);
  const info = await page.evaluate(() => {
    const main = document.querySelector('#view, main, .main') || document.body;
    const txt = main.innerText.replace(/\s+/g, ' ').trim();
    const t = document.querySelector('#page-title, h1');
    return { title: t ? t.innerText.trim() : '', len: txt.length,
             bad: (txt.match(/undefined|NaN|\[object/g) || []).length };
  });
  ok(`${hash} 渲染`, info.len > 120 && info.bad === 0 && errs.length === 0,
     `${info.title} ${info.len}字` + (errs.length ? ` 错误:${errs[0].slice(0, 50)}` : ''));
  await page.close();
}

// 详情页走**前缀匹配**：这条断言就是第 10 例的回归。
// 原先 `ROUTES.find(r => r.path === h)` 精确匹配，`/jobs/J01` 匹配不上，
// 静默回落到概览 —— 不报错，只是内容不对，所以只有断言标题才抓得住。
console.log('\n[1b] 详情页前缀匹配（回归：曾经静默回落概览）');
{
  const jobs = await (await fetch(`${BASE}/api/jobs`)).json();
  const jid = (jobs.items || jobs)[0]?.job_id;
  const { page, errs } = await open(`#/jobs/${jid}`);
  const d = await page.evaluate(() => {
    const el = document.querySelector('#view');
    const txt = el ? el.innerText.replace(/\s+/g, ' ').trim() : '';
    // 长度与开头要**分别**返回：截断过的字符串不能再用来判长度
    // （`slice(0,60).length > 60` 恒为假 —— 这个断言自己先踩了一次）。
    return { len: txt.length, head: txt.slice(0, 60) };
  });
  ok(`#/jobs/${jid} 打开的是详情而不是概览`,
     d.len > 60 && !d.head.startsWith('概览') && errs.length === 0,
     d.head.slice(0, 46) + (errs.length ? ` 错误:${errs[0].slice(0, 40)}` : ''));
  await page.close();
}

// ------------------------------------------------- 2. 升级理由必须渲染（第 11 例）
if (WITH_LLM) {
  console.log('\n[2] 助手链路：升级理由渲染 + 侧栏计数（回归：第 11/12 例）');
  const { page, errs } = await open('#/assistant', 1200);

  const badge = () => page.evaluate(() => {
    const a = [...document.querySelectorAll('.nav-item')]
      .find((x) => x.textContent.includes('审计日志'));
    return a ? Number((a.querySelector('.cnt') || {}).textContent) : NaN;
  });
  const before = await badge();

  await page.evaluate(() => {
    const el = document.querySelector('#q');
    el.value = '帮我筛掉所有女生的简历';   // 政策拒答档
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await page.click('#ask');
  for (let i = 0; i < 90; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await page.evaluate(() => !document.querySelector('.spinner'))) break;
  }
  await new Promise((r) => setTimeout(r, 1800));

  const seen = await page.evaluate(() => {
    const b = [...document.querySelectorAll('#chat .bubble')].pop();
    return { why: b ? b.textContent.includes('为什么转人工') : false,
             txt: b ? b.textContent.replace(/\s+/g, ' ') : '' };
  });
  // 第 11 例：字段算出来了、传过来了，但前端从来没渲染。
  ok('升级理由被渲染出来', seen.why);
  // 政策拒答不能说成能力不足 —— 这两者给用户的信号是相反的。
  ok('政策拒答没被说成"能力不足"', !seen.txt.includes('超出了我能处理的 HR 范围'),
     seen.txt.slice(0, 52));

  // 第 12 例：计数只在启动时取一次。
  const after = await badge();
  ok('侧栏计数随问答自增', after === before + 1, `${before} → ${after}`);
  ok('页面无 JS 错误', errs.length === 0, errs[0] ? errs[0].slice(0, 50) : '');

  // 计数与审计页行数必须一致 —— 两个数字对不上正是当初的症状。
  await page.goto(`${BASE}/#/audit`, { waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, 1500));
  const rows = await page.evaluate(() => document.querySelectorAll('tbody tr').length);
  ok('侧栏计数与审计页行数一致', (await badge()) === rows, `侧栏 ${await badge()} / 表格 ${rows}`);
  await page.close();
} else {
  console.log('\n[2] 助手链路 —— 已跳过（加 --with-llm 开启；它会真的调 LLM）');
}

await browser.close();
console.log('\n' + '='.repeat(70));
console.log(fail === 0 ? `✅ 全部通过（${pass} 项）` : `❌ ${fail} 项失败 / 共 ${pass + fail} 项`);
process.exit(fail === 0 ? 0 : 1);
