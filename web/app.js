/* ============================================================================
   HR Agent 系统 · 前端

   无构建步骤：直接 <script> 引入，视图用模板字符串 + innerHTML 渲染。
   理由是这个演示要能被评审一键跑起来，加 Vite/React 就得先 npm install，
   而本项目的价值在 Agent 编排与风控，不在前端工程化。

   两条纪律：
   1. 所有来自后端的数据都必须过 `esc()` 再进 innerHTML。简历文本是用户
      输入且可能含 HTML —— 不转义就是存储型 XSS，HR 系统里这是事故。
   2. 页面上出现的每个指标都要能指回它的来源（哪个实验、哪份文件）。
      这是本项目对"可复现"的要求在前端的延伸：不许有来路不明的数字。
   ========================================================================= */

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- 基础工具 */

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const fmt = (n) => (n ?? 0).toLocaleString('zh-CN');

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'bad' ? 6000 : 3200);
}

async function apiGet(path) {
  const r = await fetch(path);
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

/* 按钮忙碌态。所有会调 LLM 的按钮都要用它 ——
   一次匹配/一次抽取是秒级到十秒级的，没有反馈用户会重复点击。 */
async function withBusy(btn, label, fn) {
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> ${esc(label)}`;
  try { return await fn(); }
  finally { btn.disabled = false; btn.innerHTML = old; }
}

/* ---------------------------------------------------------------- 展示件 */

const VERDICT_CLASS = { '推荐': 'ok', '待定': 'warn', '不达标': 'bad' };

function verdictBadge(v) {
  return `<span class="badge ${VERDICT_CLASS[v] || 'gray'}">${esc(v)}</span>`;
}

/* 匹配度条：must 用绿、nice 用蓝、一条都没命中用红。
   做成两段而不是单一进度条，是因为 HR 要看的不是"几分"，
   而是"硬性要求差几条"—— 那才是能不能进面的依据。 */
function matchBar(r) {
  const must = r.must_pct || 0, nice = r.nice_pct || 0;
  const none = (r.must_pct === 0);
  const w = (p) => Math.round(p / 1.5 * 100); // must 满分 100，nice 折算 50
  return `<div class="match-bar" title="硬性要求 ${must}% · 加分项 ${nice}%">
    ${none ? '<i class="none" style="width:100%"></i>'
           : `<i class="must" style="width:${w(must)}%"></i>
              <i class="nice" style="width:${w(nice)}%"></i>`}
  </div>`;
}

function skillTags(list, cls) {
  if (!list || !list.length) return '<span class="mono">—</span>';
  return list.map((s) => `<span class="tag ${cls || ''}">${esc(s)}</span>`).join('');
}

function emptyState(icon, title, desc, action = '') {
  return `<div class="empty"><div class="big">${icon}</div>
    <div class="t">${esc(title)}</div><div class="d">${esc(desc)}</div>${action}</div>`;
}

function loading(text = '载入中…') {
  return `<div class="empty"><div class="t">${esc(text)}</div></div>`;
}

/* ---------------------------------------------------------------- 路由表 */

const ROUTES = [
  { path: '/',            title: '概览',        sub: '招聘全景与系统状态', icon: '◧', group: '总览' },
  { path: '/jobs',        title: '岗位管理',    sub: '批量录入 · 合规检测', icon: '▤', group: '招聘', count: 'jobs' },
  { path: '/candidates',  title: '候选人',      sub: '简历录入 · 技能档案', icon: '☰', group: '招聘', count: 'candidates' },
  { path: '/match',       title: '智能匹配',    sub: '岗位 → 候选人 Top-K 推荐', icon: '◎', group: '招聘' },
  { path: '/assistant',   title: 'AI 助手',     sub: '多智能体编排 · 风控内联', icon: '✦', group: '智能' },
  { path: '/audit',       title: '审计日志',    sub: '决策留痕', icon: '≡', group: '智能', count: 'audit_events' },
];

let STATS = {};

function renderNav(active) {
  const groups = {};
  ROUTES.forEach((r) => (groups[r.group] = groups[r.group] || []).push(r));
  $('nav').innerHTML = Object.entries(groups).map(([g, items]) => `
    <div class="nav-group">${esc(g)}</div>
    ${items.map((r) => {
      const n = r.count ? STATS[r.count] : null;
      return `<a class="nav-item ${r.path === active ? 'active' : ''}" href="#${r.path}">
        <span class="ico">${r.icon}</span><span>${esc(r.title)}</span>
        ${n != null ? `<span class="cnt">${fmt(n)}</span>` : ''}</a>`;
    }).join('')}`).join('');
}

function currentRoute() {
  const h = location.hash.replace(/^#/, '') || '/';
  const path = h.split('?')[0];
  // 详情页的 hash 形如 `/jobs/J12`，不在路由表里，必须按**前缀**匹配，
  // 且取**最长**的那个 —— 否则 `/` 会命中一切。
  // 原先这里写的是 `ROUTES.find(r => r.path === h)`，后果是岗位详情与
  // 候选人详情页永远打不开：hash 匹配不上，静默回落到概览，
  // 页面不报错、只是内容不对。是 headless 渲染 dump DOM 才发现的。
  const hit = ROUTES
    .filter((r) => path === r.path || path.startsWith(r.path + '/'))
    .sort((a, b) => b.path.length - a.path.length)[0];
  return hit || ROUTES[0];
}

async function refreshStats() {
  try {
    STATS = await apiGet('/api/overview');
    $('foot-stats').innerHTML =
      `${fmt(STATS.jobs)} 岗位 · ${fmt(STATS.candidates)} 候选人<br>技能库 ${fmt(STATS.skills_known)} 条`;
    // 侧栏计数必须跟着重画。原先这里只更新了页脚，于是"问完一问再看侧栏"
    // 计数还是旧的 —— 审计页明明列出 19 条，侧栏徽章写着 17。
    // 数据没错、也不报错，只是两个数字对不上，看起来像 bug。
    renderNav(currentRoute().path);
  } catch (e) { /* 侧栏统计失败不该挡住主视图 */ }
}

/* ================================================================ 视图 */

/* ---------------------------------------------------------------- 概览 */

async function viewOverview() {
  const s = STATS;
  const fairness = s.fairness || {};
  const recent = await apiGet('/api/audit');
  const status = s.job_status || {};
  const statusBadge = { '招聘中': 'info', '已关闭': 'gray', '已完成': 'ok' };

  return `
    <div class="grid c4" style="margin-bottom:18px">
      <div class="kpi"><div class="label">在招岗位</div>
        <div class="val">${fmt(s.jobs)}</div>
        <div class="foot">${Object.entries(status).map(([k, v]) => `${esc(k)} ${v}`).join(' · ') || '—'}</div></div>
      <div class="kpi"><div class="label">候选人库</div>
        <div class="val">${fmt(s.candidates)}</div>
        <div class="foot">已建技能档案 ${fmt(s.skills_known)} 份</div></div>
      <div class="kpi"><div class="label">已完成匹配的岗位</div>
        <div class="val">${fmt(s.matched_jobs)}</div>
        <div class="foot">共 ${fmt(s.matches)} 条匹配记录</div></div>
      <div class="kpi"><div class="label">审计事件</div>
        <div class="val">${fmt(s.audit_events)}</div>
        <div class="foot">全部决策留痕</div></div>
    </div>

    ${s.jobs === 0 ? `<div class="card"><div class="card-body">
      ${emptyState('◧', '还没有数据',
        '可以先载入演示数据（来自冻结评测集 eval/match_eval.jsonl：12 个岗位、上百份简历），也可以从「岗位管理」开始手工录入。',
        `<button class="btn primary" id="seed-btn">载入演示数据</button>`)}
    </div></div>` : ''}

    <div class="split">
      <div class="main-col">
        <div class="card">
          <div class="card-head"><h2>最近动态</h2>
            <span class="desc">每一次录入、匹配、问答都留痕</span></div>
          <div class="card-body tight">
            ${recent.length ? `<table class="tbl">
              <thead><tr><th style="width:150px">时间</th><th style="width:130px">类型</th><th>内容</th></tr></thead>
              <tbody>${recent.slice(0, 12).map((a) => `<tr>
                <td class="mono">${new Date(a.ts * 1000).toLocaleString('zh-CN', { hour12: false })}</td>
                <td><span class="badge gray">${esc(a.kind)}</span></td>
                <td>${esc(a.summary)}</td></tr>`).join('')}</tbody></table>`
              : emptyState('≡', '暂无记录', '系统还没有处理过任何请求。')}
          </div>
        </div>
      </div>

      <div class="side">
        <div class="card">
          <div class="card-head"><h2>排序公平性</h2></div>
          <div class="card-body">
            <dl class="kv" style="margin-bottom:12px">
              <dt>指标</dt><dd>${esc(fairness.metric || 'DI')}</dd>
              <dt>实测 DI</dt><dd><b>${fairness.observed}</b></dd>
              <dt>零分布 5%</dt><dd>${fairness.null_p05}</dd>
              <dt>零分布中位</dt><dd>${fairness.null_median}</dd>
              <dt>置换检验 p</dt><dd><b>${fairness.p_value}</b></dd>
            </dl>
            <div class="note ${fairness.in_null ? 'warn' : 'bad'}" style="margin-bottom:10px">
              ${esc(fairness.text || '')}
            </div>
            <div class="note info">${esc(fairness.caveat || '')}</div>
            <div class="field hint" style="margin-top:10px">
              来源：results/exp6_skill_match.md §四。置换 2000 次，种子固定。
            </div>
          </div>
        </div>
      </div>
    </div>`;
}

function bindOverview() {
  const b = $('seed-btn');
  if (!b) return;
  b.onclick = () => withBusy(b, '载入中…', async () => {
    try {
      const r = await apiPost('/api/seed', { use_llm: true });
      toast(`已录入 ${r.jobs} 个岗位，正在抽取 ${r.candidates} 份简历的技能…`);
      await pollTask(r.task_id, (t) => { b.innerHTML = `<span class="spinner"></span> ${t.done}/${t.total}`; });
      toast('演示数据载入完成', 'ok');
      await refreshStats();
      render();
    } catch (e) { toast('载入失败：' + e.message, 'bad'); }
  });
}

/* 轮询后台任务。批量导入要逐条调 LLM，同步等会超时。 */
async function pollTask(taskId, onTick) {
  for (;;) {
    const t = await apiGet('/api/tasks/' + taskId);
    if (onTick) onTick(t);
    if (t.state === 'done') return t.result;
    if (t.state === 'error') throw new Error(t.error);
    await new Promise((r) => setTimeout(r, 700));
  }
}

/* ---------------------------------------------------------------- 岗位 */

const JD_DEMO = `熟练掌握 Python；熟练掌握 SQL；熟练掌握 数据仓库；熟悉 用户增长者优先；
负责业务数据的采集、清洗与建模，支撑经营分析。`;

function jobFormHtml() {
  return `
    <div class="tabs" id="job-tabs">
      <div class="tab active" data-tab="batch">批量粘贴</div>
      <div class="tab" data-tab="single">表单录入</div>
    </div>
    <div class="card-body">
      <div id="tab-batch">
        <div class="field">
          <label>岗位文本（多份之间用一行 <code>---</code> 分隔）</label>
          <textarea class="textarea mono" id="jd-batch" rows="12"
            placeholder="后端开发工程师 | 研发中心&#10;熟练掌握 Python；熟练掌握 MySQL；&#10;负责服务端设计与开发…&#10;---&#10;数据分析师 | 数据部&#10;熟练掌握 SQL；熟练掌握 数据仓库；&#10;…"></textarea>
          <div class="hint">
            每份的第一行作为<b>岗位名称</b>，可用 <code>|</code> 追加部门与招聘人数
            （如 <code>后端开发 | 研发中心 | 3</code>）。其余部分作为 JD 全文。
          </div>
        </div>
        <div class="field">
          <label style="display:flex;align-items:center;gap:7px;font-weight:500">
            <input type="checkbox" id="jd-extract" checked>
            自动抽取硬性要求 / 加分项
          </label>
          <div class="hint">
            评测集那种「熟练掌握 X；」句式走零成本正则；手写 JD 才会调用模型。
            抽取结果可在岗位详情里逐条修改。
          </div>
        </div>
        <div class="row">
          <button class="btn primary" id="job-submit">录入岗位</button>
          <button class="btn" id="job-demo" style="flex:0 0 auto">填入示例</button>
        </div>
      </div>

      <div id="tab-single" style="display:none">
        <div class="row">
          <div class="field"><label>岗位名称</label>
            <input class="input" id="j-title" placeholder="后端开发工程师"></div>
          <div class="field"><label>部门</label>
            <input class="input" id="j-dept" placeholder="研发中心"></div>
          <div class="field" style="flex:0 0 120px"><label>招聘人数</label>
            <input class="input" id="j-hc" type="number" value="1" min="1"></div>
        </div>
        <div class="field"><label>JD 全文</label>
          <textarea class="textarea mono" id="j-jd" rows="6">${esc(JD_DEMO)}</textarea></div>
        <div class="row">
          <div class="field"><label>硬性要求（每行一条）</label>
            <textarea class="textarea" id="j-must" rows="3" placeholder="Python&#10;SQL"></textarea></div>
          <div class="field"><label>加分项（每行一条）</label>
            <textarea class="textarea" id="j-nice" rows="3" placeholder="用户增长"></textarea></div>
        </div>
        <button class="btn primary" id="job-submit-single">录入岗位</button>
      </div>
    </div>`;
}

function parseJobBlocks(text) {
  return text.split(/^\s*(?:-{3,}|={3,})\s*$/m)
    .map((b) => b.trim()).filter(Boolean)
    .map((b) => {
      const [head, ...rest] = b.split('\n');
      const parts = head.split('|').map((x) => x.trim());
      const n = parseInt(parts[2], 10);
      return {
        title: parts[0] || '未命名岗位',
        dept: parts[1] || '',
        headcount: Number.isFinite(n) ? n : 1,
        jd_text: rest.join('\n').trim() || head,
      };
    });
}

function parseResumeBlocks(text) {
  return text.split(/^\s*(?:-{3,}|={3,})\s*$/m)
    .map((b) => b.trim()).filter(Boolean)
    .map((b, i) => {
      const [head, ...rest] = b.split('\n');
      const parts = head.split('|').map((x) => x.trim());
      return {
        name: parts[0] || `候选人${i + 1}`,
        source: parts[1] || '手工录入',
        resume_text: rest.join('\n').trim() || head,
      };
    });
}

async function viewJobs() {
  const jobs = await apiGet('/api/jobs');
  return `
    <div class="split">
      <div class="side">
        <div class="card">
          <div class="card-head"><h2>录入岗位</h2></div>
          ${jobFormHtml()}
        </div>
        <div class="card">
          <div class="card-head"><h2>合规检测</h2>
            <span class="desc">录用条件自检</span></div>
          <div class="card-body">
            <div class="field">
              <textarea class="textarea" id="screen-text" rows="3"
                placeholder="粘贴 JD 或筛选条件，例如：帮我筛掉所有非 985 的简历"></textarea>
              <div class="hint">
                检测录用条件里是否含性别、年龄、婚育、院校层次等歧视性表述。
                复用编排链路的请求级判据（B4 层），与 AI 助手是同一套规则。
              </div>
            </div>
            <button class="btn block" id="screen-btn">开始检测</button>
            <div id="screen-out" style="margin-top:12px"></div>
          </div>
        </div>
      </div>

      <div class="main-col">
        <div class="card">
          <div class="card-head"><h2>岗位列表</h2>
            <span class="desc">${jobs.length} 个岗位</span></div>
          <div class="card-body tight">
            ${jobs.length ? `<table class="tbl">
              <thead><tr>
                <th style="width:34%">岗位</th><th>部门</th>
                <th class="num" style="width:90px">招聘人数</th>
                <th style="width:100px">状态</th>
                <th class="num" style="width:90px">已匹配</th>
                <th style="width:110px"></th>
              </tr></thead>
              <tbody>${jobs.map((j) => `<tr>
                <td><a href="#/jobs/${encodeURIComponent(j.job_id)}"><b>${esc(j.title)}</b></a>
                  <div class="mono">${esc(j.job_id)}</div></td>
                <td>${esc(j.dept || '—')}</td>
                <td class="num">${j.headcount}</td>
                <td><span class="badge ${j.status === '招聘中' ? 'info' : 'gray'}">${esc(j.status)}</span></td>
                <td class="num">${j.n_matched ? fmt(j.n_matched) : '<span class="mono">—</span>'}</td>
                <td><a class="btn sm" href="#/match?job=${encodeURIComponent(j.job_id)}">去匹配</a></td>
              </tr>`).join('')}</tbody></table>`
              : emptyState('▤', '还没有岗位',
                  '用左侧的「批量粘贴」一次录入多个岗位，或切到「表单录入」单个添加。')}
          </div>
        </div>
      </div>
    </div>`;
}

function bindJobs() {
  document.querySelectorAll('#job-tabs .tab').forEach((t) => {
    t.onclick = () => {
      document.querySelectorAll('#job-tabs .tab').forEach((x) => x.classList.remove('active'));
      t.classList.add('active');
      $('tab-batch').style.display = t.dataset.tab === 'batch' ? '' : 'none';
      $('tab-single').style.display = t.dataset.tab === 'single' ? '' : 'none';
    };
  });

  $('job-demo').onclick = () => {
    $('jd-batch').value =
      '后端开发工程师 | 研发中心 | 2\n熟练掌握 Python；熟练掌握 MySQL；熟练掌握 Redis；\n' +
      '负责核心服务的架构设计与性能优化。熟悉 分布式系统者优先；\n---\n' +
      '数据分析师 | 数据部 | 1\n熟练掌握 SQL；熟练掌握 数据仓库；熟练掌握 数据可视化；\n' +
      '支撑经营分析与管理看板。熟悉 用户增长者优先；';
  };

  const submitBatch = async (btn) => withBusy(btn, '录入中…', async () => {
    const blocks = parseJobBlocks($('jd-batch').value);
    if (!blocks.length) return toast('请先粘贴岗位文本', 'bad');
    const wantExtract = $('jd-extract').checked;
    const jobs = [];
    for (let i = 0; i < blocks.length; i++) {
      const b = blocks[i];
      const job = { ...b, job_id: 'J' + Date.now().toString(36) + i, status: '招聘中',
                    must: [], nice: [] };
      if (wantExtract) {
        btn.innerHTML = `<span class="spinner"></span> 抽取要求 ${i + 1}/${blocks.length}`;
        try {
          const r = await apiPost('/api/jobs/extract', { jd_text: b.jd_text });
          job.must = r.must || [];
          job.nice = r.nice || [];
        } catch (e) { /* 抽取失败不阻断录入，留空由人工补 */ }
      }
      jobs.push(job);
    }
    await apiPost('/api/jobs', { jobs });
    toast(`已录入 ${jobs.length} 个岗位`, 'ok');
    await refreshStats(); render();
  });

  $('job-submit').onclick = (e) => submitBatch(e.target);

  $('job-submit-single').onclick = (e) => withBusy(e.target, '录入中…', async () => {
    const title = $('j-title').value.trim();
    if (!title) return toast('请填写岗位名称', 'bad');
    const splitLines = (v) => v.split('\n').map((x) => x.trim()).filter(Boolean);
    await apiPost('/api/jobs', { jobs: [{
      job_id: 'J' + Date.now().toString(36), title, dept: $('j-dept').value.trim(),
      jd_text: $('j-jd').value.trim(), headcount: parseInt($('j-hc').value, 10) || 1,
      must: splitLines($('j-must').value), nice: splitLines($('j-nice').value),
      status: '招聘中' }] });
    toast('已录入岗位', 'ok');
    await refreshStats(); render();
  });

  $('screen-btn').onclick = (e) => withBusy(e.target, '检测中…', async () => {
    const text = $('screen-text').value.trim();
    if (!text) return toast('请先输入要检测的文本', 'bad');
    const r = await apiPost('/api/jobs/screen', { jd_text: text });
    const cls = { refuse: 'bad', escalate: 'warn', ok: 'ok' }[r.action] || 'info';
    const label = { refuse: '拒绝', escalate: '转人工复核', ok: '通过' }[r.action] || r.action;
    $('screen-out').innerHTML = `
      <div class="note ${cls}">
        <b>裁决：${esc(label)}</b>${r.category ? ` · 类别 ${esc(r.category)}` : ''}<br>
        ${esc(r.reason)}
      </div>`;
  });
}

/* ---------------------------------------------------------------- 岗位详情 */

async function viewJobDetail(jobId) {
  const j = await apiGet('/api/jobs/' + encodeURIComponent(jobId));
  const matches = await apiGet(`/api/jobs/${encodeURIComponent(jobId)}/matches`);
  return `
    <div style="margin-bottom:14px"><a href="#/jobs">← 返回岗位列表</a></div>
    <div class="split">
      <div class="side">
        <div class="card">
          <div class="card-head"><h2>${esc(j.title)}</h2></div>
          <div class="card-body">
            <dl class="kv">
              <dt>岗位编号</dt><dd class="mono">${esc(j.job_id)}</dd>
              <dt>部门</dt><dd>${esc(j.dept || '—')}</dd>
              <dt>招聘人数</dt><dd>${j.headcount}</dd>
              <dt>状态</dt><dd><span class="badge info">${esc(j.status)}</span></dd>
            </dl>
          </div>
        </div>
        <div class="card">
          <div class="card-head"><h2>硬性要求</h2>
            <span class="desc">缺一条即不推荐</span></div>
          <div class="card-body">${skillTags(j.must, 'hit')}</div>
        </div>
        <div class="card">
          <div class="card-head"><h2>加分项</h2></div>
          <div class="card-body">${skillTags(j.nice, 'nice')}</div>
        </div>
        <div class="card">
          <div class="card-head"><h2>JD 全文</h2></div>
          <div class="card-body"><div class="mono" style="white-space:pre-wrap;line-height:1.7">${esc(j.jd_text)}</div></div>
        </div>
      </div>

      <div class="main-col">
        <div class="card">
          <div class="card-head"><h2>候选人推荐</h2>
            <span class="desc">${matches.length ? `已跑过匹配，共 ${matches.length} 人` : '尚未运行'}</span>
            <span class="spacer"></span>
            <a class="btn primary" href="#/match?job=${encodeURIComponent(j.job_id)}">运行匹配</a>
          </div>
          <div class="card-body tight">
            ${matches.length ? matchTable(matches.slice(0, 20)) :
              emptyState('◎', '还没有匹配结果',
                '点右上角「运行匹配」，系统会用结构化对齐给全库候选人打分并排序。')}
          </div>
        </div>
      </div>
    </div>`;
}

function matchTable(rows) {
  return `<table class="tbl">
    <thead><tr>
      <th style="width:46px">#</th><th style="width:150px">候选人</th>
      <th style="width:190px">匹配度</th><th class="num" style="width:70px">得分</th>
      <th>硬性要求明细</th><th style="width:76px">结论</th>
    </tr></thead>
    <tbody>${rows.map((r, i) => `<tr>
      <td><span class="rank ${i < 3 ? 'top' + (i + 1) : ''}">${i + 1}</span></td>
      <td><a href="#/candidates/${encodeURIComponent(r.candidate_id)}"><b>${esc(r.name)}</b></a>
        <div class="mono">${esc(r.mode || '')}${r.source ? ' · ' + esc(r.source) : ''}</div></td>
      <td>${matchBar(r)}
        <div class="mono" style="margin-top:3px">硬性 ${r.must_pct}% · 加分 ${r.nice_pct}%</div></td>
      <td class="num"><b>${r.match_pct}</b><div class="mono">${r.score}</div></td>
      <td>${(r.must_hit || []).map((s) => `<span class="tag hit">✓ ${esc(s)}</span>`).join('')}
          ${(r.must_miss || []).map((s) => `<span class="tag miss">✗ ${esc(s)}</span>`).join('')}
          ${(r.nice_hit || []).length ? `<div style="margin-top:4px">${(r.nice_hit).map((s) => `<span class="tag nice">+ ${esc(s)}</span>`).join('')}</div>` : ''}</td>
      <td>${verdictBadge(r.verdict)}</td>
    </tr>`).join('')}</tbody></table>`;
}

/* ---------------------------------------------------------------- 候选人 */

async function viewCandidates() {
  const cs = await apiGet('/api/candidates');
  return `
    <div class="split">
      <div class="side">
        <div class="card">
          <div class="card-head"><h2>录入简历</h2>
            <span class="desc">自动抽取技能</span></div>
          <div class="card-body">
            <div class="field">
              <label>简历文本（多份之间用一行 <code>---</code> 分隔）</label>
              <textarea class="textarea mono" id="cv-batch" rows="13"
                placeholder="张三&#10;男 | 28岁 | 5年经验&#10;技能：熟练使用 Python、SQL；了解过 Go&#10;---&#10;李四&#10;…"></textarea>
              <div class="hint">
                第一行作为<b>姓名</b>，可用 <code>|</code> 追加来源（如 <code>张三 | 内推</code>）。
                技能抽取按<b>简历文本哈希</b>缓存，同一份简历重复录入不再消耗模型调用。
              </div>
            </div>
            <div class="field">
              <label style="display:flex;align-items:center;gap:7px;font-weight:500">
                <input type="checkbox" id="cv-llm" checked>
                用模型抽取技能
              </label>
              <div class="hint">
                关掉则只入库、不抽技能，匹配时这些人的硬性要求覆盖率会是 0。
                用于快速验证流程。
              </div>
            </div>
            <button class="btn primary block" id="cv-submit">录入简历</button>
            <div id="cv-progress" style="margin-top:12px"></div>
          </div>
        </div>
      </div>

      <div class="main-col">
        <div class="card">
          <div class="card-head"><h2>候选人库</h2>
            <span class="desc">${cs.length} 人</span></div>
          <div class="card-body tight">
            ${cs.length ? `<table class="tbl">
              <thead><tr><th style="width:150px">姓名</th><th style="width:120px">来源</th>
                <th>技能（抽取结果）</th><th style="width:110px">录入时间</th></tr></thead>
              <tbody>${cs.slice(0, 200).map((c) => `<tr>
                <td><a href="#/candidates/${encodeURIComponent(c.candidate_id)}"><b>${esc(c.name)}</b></a>
                  <div class="mono">${esc(c.candidate_id)}</div></td>
                <td>${esc(c.source || '—')}</td>
                <td>${skillTags(c.skills)}</td>
                <td class="mono">${new Date(c.created_at * 1000).toLocaleDateString('zh-CN')}</td>
              </tr>`).join('')}</tbody></table>
              ${cs.length > 200 ? `<div class="card-body hint">仅显示最近 200 条，共 ${fmt(cs.length)} 条。</div>` : ''}`
              : emptyState('☰', '候选人库为空',
                  '用左侧粘贴简历批量录入，或在概览页载入演示数据。')}
          </div>
        </div>
      </div>
    </div>`;
}

function bindCandidates() {
  $('cv-submit').onclick = (e) => withBusy(e.target, '录入中…', async () => {
    const rows = parseResumeBlocks($('cv-batch').value);
    if (!rows.length) return toast('请先粘贴简历文本', 'bad');
    try {
      const r = await apiPost('/api/candidates',
        { candidates: rows, use_llm: $('cv-llm').checked });
      const res = await pollTask(r.task_id, (t) => {
        $('cv-progress').innerHTML =
          `<div class="progress"><i style="width:${t.total ? t.done / t.total * 100 : 0}%"></i></div>
           <div class="hint" style="margin-top:5px">${esc(t.message || '')} （${t.done}/${t.total}）</div>`;
      });
      $('cv-progress').innerHTML =
        `<div class="note ok">已录入 ${res.imported} 份简历，共消耗 ${fmt(res.tokens)} token。</div>`;
      toast(`已录入 ${res.imported} 份简历`, 'ok');
      await refreshStats(); render();
    } catch (err) { toast('录入失败：' + err.message, 'bad'); }
  });
}

async function viewCandidateDetail(cid) {
  const c = await apiGet('/api/candidates/' + encodeURIComponent(cid));
  return `
    <div style="margin-bottom:14px"><a href="#/candidates">← 返回候选人列表</a></div>
    <div class="split">
      <div class="side">
        <div class="card">
          <div class="card-head"><h2>${esc(c.name)}</h2></div>
          <div class="card-body">
            <dl class="kv">
              <dt>编号</dt><dd class="mono">${esc(c.candidate_id)}</dd>
              <dt>来源</dt><dd>${esc(c.source || '—')}</dd>
              <dt>投递岗位</dt><dd>${c.applied_job ? `<a href="#/jobs/${encodeURIComponent(c.applied_job)}">${esc(c.applied_job)}</a>` : '—'}</dd>
              <dt>录入时间</dt><dd>${new Date(c.created_at * 1000).toLocaleString('zh-CN', { hour12: false })}</dd>
            </dl>
          </div>
        </div>
        <div class="card">
          <div class="card-head"><h2>技能档案</h2>
            <span class="desc">${(c.skills || []).length} 项</span></div>
          <div class="card-body">
            ${skillTags(c.skills)}
            <div class="hint" style="margin-top:10px">
              由模型从简历中抽取，判定标准是<b>真正掌握</b>：
              简历里写「了解过 / 听说过」的技能会被排除，工作经历里体现的会被纳入。
            </div>
          </div>
        </div>
      </div>
      <div class="main-col">
        <div class="card">
          <div class="card-head"><h2>简历原文</h2></div>
          <div class="card-body"><div class="mono" style="white-space:pre-wrap;line-height:1.75">${esc(c.resume_text)}</div></div>
        </div>
      </div>
    </div>`;
}

/* ---------------------------------------------------------------- 匹配 */

async function viewMatch(params) {
  const jobs = await apiGet('/api/jobs');
  const preset = params.get('job') || (jobs[0] && jobs[0].job_id) || '';
  return `
    <div class="card">
      <div class="card-head"><h2>运行匹配</h2>
        <span class="desc">结构化对齐 · 硬性要求优先</span></div>
      <div class="card-body">
        ${jobs.length ? `
        <div class="row" style="align-items:flex-end">
          <div class="field" style="flex:3"><label>目标岗位</label>
            <select class="select" id="m-job">${jobs.map((j) =>
              `<option value="${esc(j.job_id)}" ${j.job_id === preset ? 'selected' : ''}>
                ${esc(j.title)}${j.dept ? ' · ' + esc(j.dept) : ''}</option>`).join('')}</select></div>
          <div class="field" style="flex:0 0 130px"><label>返回 Top-K</label>
            <input class="input" id="m-topk" type="number" value="10" min="1" max="200"></div>
          <div class="field" style="flex:0 0 230px">
            <label style="display:flex;align-items:center;gap:7px;font-weight:500;margin-bottom:9px">
              <input type="checkbox" id="m-rerank"> 对 Top-K 深度复核</label>
            <button class="btn primary block" id="m-run">运行匹配</button></div>
        </div>
        <div class="note info">
          <b>两级推荐，取舍来自实测。</b>
          召回用结构化对齐（M1）：12 个岗位的 NDCG@k 0.9909，<b>零 token</b>。
          深度复核用逐技能模型判断（M2）：NDCG@k 0.9940，<b>88 token/对</b> ——
          两者配对检验 p=0.6430，<b>打平</b>。所以复核默认关闭，只在你明确要求时
          对少数人跑一次。复核读的是<b>去标识后</b>的简历。
        </div>
        <div class="note warn" style="margin-top:10px">
          <b>一处与实验数字的差别，先说明白。</b>
          嵌入模型对跨语言的同名技术几乎无能为力（<code>Microservices ↔ 微服务架构</code>
          实测余弦 0.570，低于阈值 0.75）。不收一张人工同义词表，
          招聘官会看到每个候选人都标着「✗ Microservices」，
          而其中一半人简历里写着「微服务架构」。
          所以产品路径加了 <b>7 组 19 个写法</b>的严格同义表
          （只收同义，不收「Kafka / 消息队列」这种产品与品类的关系）。
          <b>exp6 报出的 0.9909 不含这张表</b> —— 那是纯嵌入对齐的成绩，
          在评测集上两者排序一致，因为所有人都缺同一条要求，名次不受影响。
        </div>`
        : emptyState('▤', '还没有岗位', '先到「岗位管理」录入岗位，再回来运行匹配。',
            `<a class="btn primary" href="#/jobs">去录入岗位</a>`)}
      </div>
    </div>
    <div id="m-out"></div>`;
}

function bindMatch() {
  const btn = $('m-run');
  if (!btn) return;
  btn.onclick = () => withBusy(btn, '匹配中…', async () => {
    try {
      const r = await apiPost('/api/match', {
        job_id: $('m-job').value,
        top_k: parseInt($('m-topk').value, 10) || 10,
        rerank: $('m-rerank').checked,
      });
      const counts = r.top.reduce((a, x) => (a[x.verdict] = (a[x.verdict] || 0) + 1, a), {});
      $('m-out').innerHTML = `
        <div class="card">
          <div class="card-head">
            <h2>${esc(r.job.title)} · Top ${r.top.length}</h2>
            <span class="desc">全库 ${r.total} 人参与排序 ·
              ${Object.entries(counts).map(([k, v]) => `${esc(k)} ${v}`).join(' · ')}</span>
            <span class="spacer"></span>
            ${r.rerank_tokens ? `<span class="badge violet">复核消耗 ${fmt(r.rerank_tokens)} token</span>` : ''}
            <span class="badge gray">τ = ${r.tau}</span>
          </div>
          <div class="card-body tight">${matchTable(r.top)}</div>
        </div>
        <div class="card">
          <div class="card-head"><h2>本次排序的公平性声明</h2></div>
          <div class="card-body">
            <div class="note warn">${esc(r.fairness.text)}</div>
            <div class="note info" style="margin-top:10px">${esc(r.fairness.caveat)}</div>
            <div class="note gray" style="margin-top:10px;background:var(--n50);border-color:var(--n200);color:var(--n600)">
              ${esc(r.alias_note)}
            </div>
          </div>
        </div>`;
      toast(`匹配完成，返回 ${r.top.length} 人`, 'ok');
      refreshStats();   // 每次排序也写审计
    } catch (e) { toast('匹配失败：' + e.message, 'bad'); }
  });
}

/* ---------------------------------------------------------------- AI 助手 */

const SAMPLE_QUERIES = [
  '供应链部的招聘完成率是多少？',
  '帮我筛掉所有非 985 的简历',
  '帮我筛掉所有女生的简历',
  '研发中心上季度的出勤率是多少？',
  '帮我做一份人岗匹配',
  '帮我写个爬虫抓竞品数据',
];

function assistantHtml() {
  return `
    <div class="split">
      <div class="main-col">
        <div class="card">
          <div class="card-head"><h2>对话</h2>
            <span class="desc">完整走编排链路：路由 → 执行体 → 风控 → 合成</span></div>
          <div class="card-body">
            <div class="chat" id="chat"></div>
            <div class="field" style="margin-top:16px">
              <textarea class="textarea short" id="q" rows="2"
                placeholder="输入问题，回车发送（Shift+回车换行）"></textarea>
            </div>
            <div class="actions">
              <button class="btn primary" id="ask">发送</button>
              <button class="btn" id="clear">清空</button>
            </div>
          </div>
        </div>
      </div>
      <div class="side">
        <div class="card">
          <div class="card-head"><h2>试试这些</h2></div>
          <div class="card-body" id="samples"></div>
        </div>
        <div class="card">
          <div class="card-head"><h2>这条链路在防什么</h2></div>
          <div class="card-body">
            <div class="note info" style="margin-bottom:10px">
              <b>B4 请求级合规判据</b><br>
              「筛掉所有非 985 的」不是普通查询，是用代理变量做歧视性筛选。
              判据会把它转人工，而<b>不是</b>反问「要筛哪个岗位」——
              那等于请用户补材料来完成一次歧视。
            </div>
            <div class="note warn" style="margin-bottom:10px">
              <b>B5 缺输入 → 澄清</b><br>
              「做一份人岗匹配」缺 JD 和简历，系统问的是<b>具体缺什么</b>，
              而不是笼统地说「我做不了」。
            </div>
            <div class="note ok">
              <b>四道闸门</b><br>
              截断 / 证据接地 / 合规 / 不确定性，逐条裁决都显示在回答下方。
            </div>
          </div>
        </div>
      </div>
    </div>`;
}

function pushMsg(kind, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + kind;
  d.innerHTML = `<div class="who">${kind === 'user' ? '我' : 'AI'}</div>
                 <div class="bubble">${html}</div>`;
  $('chat').appendChild(d);
  d.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return d;
}

function renderTrace(t) {
  const screenAction = (t.screen || {}).action || 'ok';
  const screenCls = { refuse: 'bad', escalate: 'warn', ok: 'ok' }[screenAction] || 'info';
  const screenLabel = { refuse: '拒绝', escalate: '转人工', ok: '通过' }[screenAction] || screenAction;

  // 按阶段分组：执行体级与答案级是两处不同的检查，混在一起显示会读成"重复执行"。
  const byStage = {};
  (t.verdicts || []).forEach((v) => (byStage[v.stage || '裁决'] = byStage[v.stage || '裁决'] || []).push(v));
  const verdicts = Object.entries(byStage).map(([stage, list]) => `
    <div class="vstage">${esc(stage)}级风控</div>
    ${list.map((v) => `
      <div class="verdict ${v.passed ? 'pass' : (v.severity === 'block' ? 'block' : 'warn')}">
        <span class="g">${esc(v.guard)}</span>
        <span>${v.passed ? '通过' : esc(v.reason || '')}</span>
      </div>`).join('')}`).join('');

  const flags = [
    `<span class="badge ${screenCls}">合规判据 ${esc(screenLabel)}</span>`,
    t.route && t.route.needs_clarification ? '<span class="badge warn">需澄清</span>' : '',
    (t.route && t.route.executors || []).map((e) => `<span class="badge info">${esc(e)}</span>`).join(''),
    t.escalated ? '<span class="badge bad">已转人工</span>' : '',
    t.needs_review ? '<span class="badge warn">建议复核</span>' : '',
    `<span class="badge gray">${fmt(t.cost && t.cost.tokens)} token</span>`,
    `<span class="badge gray">${(t.cost && t.cost.latency_s || 0).toFixed(1)}s</span>`,
  ].filter(Boolean).join('');

  const interrupt = t.interrupted ? `
    <div class="note warn" style="margin-top:10px">
      <b>已暂停，等待人工处置</b><br>
      系统没有自行决定，而是把判断交给人。请选择处置方式：
      <div style="margin-top:9px;display:flex;gap:8px">
        <button class="btn sm primary" data-resume="approve">采纳该结果</button>
        <button class="btn sm" data-resume="reject">驳回</button>
        <button class="btn sm danger" data-resume="block">按合规风险拦截</button>
      </div>
      <div class="mono" style="margin-top:7px">thread=${esc(t.thread_id || '')}</div>
    </div>` : '';

  const human = t.human && t.human.action ? `
    <div class="note ${t.human.action === 'block' ? 'bad' : t.human.action === 'approve' ? 'ok' : 'warn'}"
         style="margin-top:10px">
      <b>人工处置已记录：${esc(t.human.action)}</b><br>
      该处置写入状态与审计日志，但<b>不改写系统给出的答案</b> ——
      答案归系统，处置归人，两者分开才能事后归因。
    </div>` : '';

  const evidence = (t.evidence || []).length ? `
    <div style="margin-top:10px">
      <div class="mono" style="margin-bottom:4px">证据 ${t.evidence.length} 条</div>
      ${t.evidence.slice(0, 4).map((e) =>
        `<div class="mono" style="margin-bottom:2px">· [${esc(e.kind)}] ${esc(e.ref)} → ${esc(e.value)}</div>`).join('')}
    </div>` : '';

  // 升级理由是**算出来了、也传过来了、但一直没人渲染**的字段：界面只显示一个
  // 「已转人工」徽章，评审看不出**为什么**转。这正是本项目记录在案的那类 bug
  // （声明了但没接线）—— 字段契约齐全、不报错、离线指标一个都不变，
  // 只是功能静默失效。所以这里单独把它渲染出来。
  // 它尤其重要：`拒答（attribute）` 与 `拒答（越界）` 对用户的含义完全不同 ——
  // 前者是"这个请求我不能做"，后者是"这个问题我答不了"，混在一起显示会误导。
  const escNote = (t.escalated && t.escalation_reason) ? `
    <div class="note warn" style="margin-top:10px">
      <b>为什么转人工</b><br>${esc(t.escalation_reason)}
    </div>` : '';

  return `<div class="meta">${flags}</div>${verdicts ? `<div class="verdicts">${verdicts}</div>` : ''}${escNote}${interrupt}${human}${evidence}`;
}

function bindAssistant() {
  const send = () => {
    const q = $('q').value.trim();
    if (!q) return;
    $('q').value = '';
    pushMsg('user', esc(q));
    const pending = pushMsg('bot', '<span class="spinner"></span> 处理中…');
    apiPost('/api/assistant', { query: q }).then((t) => {
      pending.querySelector('.bubble').innerHTML = esc((t.answer || '').trim()) + renderTrace(t);
      bindResume(pending);
      // 每一问都会写一条审计，侧栏那个计数得跟着走。
      refreshStats();
    }).catch((e) => {
      pending.querySelector('.bubble').innerHTML =
        `<span class="badge bad">请求失败</span> ${esc(e.message)}`;
    });
  };

  $('ask').onclick = send;
  $('q').onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };
  $('clear').onclick = () => { $('chat').innerHTML = ''; };

  $('samples').innerHTML = SAMPLE_QUERIES.map((s) =>
    `<div class="nav-item" style="color:var(--n700);margin-bottom:5px" data-q="${esc(s)}">
       <span class="ico">›</span><span style="font-size:12.5px">${esc(s)}</span></div>`).join('');
  $('samples').querySelectorAll('[data-q]').forEach((el) => {
    el.onclick = () => { $('q').value = el.dataset.q; send(); };
  });

  pushMsg('bot', `我是 HR Agent。可以查人事数据、答制度问题、做简历筛选与人岗匹配。
所有回答都会经过风控闸门；涉及歧视性筛选的请求会被转人工，而不是照做。`);
}

function bindResume(scope) {
  scope.querySelectorAll('[data-resume]').forEach((b) => {
    b.onclick = () => withBusy(b, '提交中…', async () => {
      const t = scope.querySelector('.bubble');
      const m = /thread=(\S+)/.exec(t.textContent);
      if (!m) return toast('找不到暂停点', 'bad');
      try {
        const r = await apiPost('/api/assistant/resume',
          { thread_id: m[1], decision: { action: b.dataset.resume } });
        pushMsg('bot', `<b>人工处置已提交：${esc(b.dataset.resume)}</b>\n\n${esc(r.answer)}` + renderTrace(r));
        toast('处置已提交', 'ok');
        refreshStats();   // 处置也写审计，计数同样要刷新
      } catch (e) { toast('提交失败：' + e.message, 'bad'); }
    });
  });
}

/* ---------------------------------------------------------------- 审计 */

async function viewAudit() {
  const rows = await apiGet('/api/audit');
  const cls = (k) => k.startsWith('assistant') ? 'violet'
    : k.startsWith('match') ? 'info' : k.startsWith('job') ? 'gray' : 'ok';
  return `
    <div class="card">
      <div class="card-head"><h2>审计日志</h2>
        <span class="desc">${rows.length} 条 · 决策留痕，可按时间回溯</span></div>
      <div class="card-body tight">
        ${rows.length ? `<table class="tbl">
          <thead><tr><th style="width:160px">时间</th><th style="width:150px">事件</th>
            <th>摘要</th><th style="width:34%">详情</th></tr></thead>
          <tbody>${rows.map((a) => `<tr>
            <td class="mono">${new Date(a.ts * 1000).toLocaleString('zh-CN', { hour12: false })}</td>
            <td><span class="badge ${cls(a.kind)}">${esc(a.kind)}</span></td>
            <td>${esc(a.summary)}</td>
            <td class="mono" style="word-break:break-all">${esc(JSON.stringify(a.payload))}</td>
          </tr>`).join('')}</tbody></table>`
          : emptyState('≡', '暂无审计记录', '系统处理请求后会自动留痕。')}
      </div>
    </div>`;
}

/* ================================================================ 渲染 */

let RENDER_SEQ = 0;

async function render() {
  const seq = ++RENDER_SEQ;
  const route = currentRoute();
  const hash = location.hash.replace(/^#/, '') || '/';
  const [pathPart, queryPart] = hash.split('?');
  const params = new URLSearchParams(queryPart || '');
  const seg = pathPart.split('/').filter(Boolean);

  $('page-title').textContent = route.title;
  $('page-sub').textContent = route.sub;
  $('topbar-actions').innerHTML = '';
  renderNav(route.path);

  const view = $('view');
  view.innerHTML = loading();
  try {
    let html = '', bind = null;
    if (route.path === '/') { html = await viewOverview(); bind = bindOverview; }
    else if (route.path === '/jobs' && seg[1]) html = await viewJobDetail(decodeURIComponent(seg[1]));
    else if (route.path === '/jobs') { html = await viewJobs(); bind = bindJobs; }
    else if (route.path === '/candidates' && seg[1]) html = await viewCandidateDetail(decodeURIComponent(seg[1]));
    else if (route.path === '/candidates') { html = await viewCandidates(); bind = bindCandidates; }
    else if (route.path === '/match') { html = await viewMatch(params); bind = bindMatch; }
    else if (route.path === '/assistant') { html = assistantHtml(); bind = bindAssistant; }
    else if (route.path === '/audit') html = await viewAudit();

    if (seq !== RENDER_SEQ) return;   // 期间又导航了，丢弃这次结果
    view.innerHTML = html;
    if (bind) bind();
  } catch (e) {
    if (seq !== RENDER_SEQ) return;
    view.innerHTML = `<div class="card"><div class="card-body">
      <div class="note bad"><b>载入失败</b><br>${esc(e.message)}</div></div></div>`;
  }
}

window.addEventListener('hashchange', render);
(async function boot() {
  await refreshStats();
  renderNav(currentRoute().path);
  render();
})();
