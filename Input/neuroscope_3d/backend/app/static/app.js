// Landing page (/) and analyse page (/analyse). Each part below runs only on the page that has its elements.
// Analyse: one drop zone, a single 4-channel NIfTI or four sequence files (+ optional expert mask).
// Anything refused (here or by the server) or accepted with warnings is shown in a pop-up dialog.
const $ = (s, el = document) => el.querySelector(s);
const SEQS = ['T1', 'T1Gd', 'T2', 'FLAIR'];
const DECATHLON_ORDER = ['FLAIR', 'T1', 'T1Gd', 'T2'];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtMB = (b) => (b / 1048576).toFixed(b < 1048576 * 10 ? 2 : 1) + ' MB';

// ------------------------------------------------------------------ about: status, figures, evaluation, limits
async function loadAbout() {
  try {
    const a = await (await fetch('/api/about')).json();
    $('#status').innerHTML = `<i class="dot ${a.cuda ? 'on' : ''}"></i>${a.cuda ? 'GPU ready' : 'CPU only'} · ${esc(a.device)}`;
    $('#disclaimer-full').textContent = a.disclaimer;
    if (!$('#eval-table')) return;                     // landing-only sections below
    const ev = a.evaluation;
    if (ev) {
      $('#fig-wt').textContent = ev.summary.WT.dice_mean.toFixed(3);
      $('#fig-wt-note').textContent = `mean, ${ev.n_cases} held-out cases*`;
      $('#eval-intro').textContent = `${ev.n_cases} held-out labelled cases, sampled at random with a fixed seed. One study per patient. We did no training or tuning.`;
      const rows = ['WT', 'TC', 'ET'].map((r) => {
        const s = ev.summary[r];
        const name = { WT: 'Whole tumour', TC: 'Tumour core', ET: 'Enhancing' }[r];
        return `<tr><td><b>${r}</b><br><span class="small" style="font-family:var(--sans);color:var(--ink-3)">${name}</span></td>
          <td class="big">${s.dice_mean.toFixed(3)}<span class="sd">±${s.dice_sd.toFixed(2)}</span></td>
          <td><div class="bar"><i style="width:${(s.dice_mean * 100).toFixed(1)}%"></i></div></td>
          <td>${s.sensitivity_mean === null ? '—' : s.sensitivity_mean.toFixed(3)}</td>
          <td>${s.hd95_median_mm === null ? '—' : s.hd95_median_mm.toFixed(1) + ' mm'}</td></tr>`;
      }).join('');
      $('#eval-table').innerHTML = `<thead><tr><th>Region</th><th>Dice</th><th></th><th>Sensitivity</th><th>HD95 median</th></tr></thead><tbody>${rows}</tbody>`;
      $('#eval-disclosure').textContent = '* ' + ev.overlap_disclosure;
      $('#eval-cases').textContent = ev.case_ids.join('  ');
    } else {
      $('#eval-table').innerHTML = '<tbody><tr><td>Evaluation has not been run yet (python scripts/evaluate.py).</td></tr></tbody>';
    }
    $('#limits-list').innerHTML = a.limitations.map((l) => `<li>${esc(l)}</li>`).join('');
  } catch {
    $('#status').innerHTML = '<i class="dot"></i>server unreachable';
  }
}

// ------------------------------------------------------------------ hero plate: axial sweep
async function startPlate() {
  const canvas = $('#plate');
  if (!canvas) return;
  let meta;
  try { meta = await (await fetch('/static/hero/plate.json')).json(); } catch { return; }
  const img = new Image();
  img.src = '/static/hero/plate.jpg';
  await img.decode().catch(() => null);
  if (!img.naturalWidth) return;
  const W = meta.frame_width, H = meta.frame_height, n = meta.frames.length;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const read = $('#plate-read'), frame = $('.plate-frame');
  $('#plate-case').textContent = meta.case.split(' · ')[0];
  const draw = (i) => {
    ctx.drawImage(img, 0, i * H, W, H, 0, 0, W, H);
    const f = meta.frames[i];
    const area = (Number(f.area_mm2['1']) + Number(f.area_mm2['2']) + Number(f.area_mm2['4'])) / 100;
    read.innerHTML = `z ${f.z_mm.toFixed(0).padStart(4, ' ')} mm<br>WT ${area.toFixed(1)} cm² in slice<br>` +
      `<span style="opacity:.7">case WT ${meta.volumes_mL.WT} mL</span>`;
    frame.style.setProperty('--scan', `${(1 - i / (n - 1)) * 100}%`);
  };
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { draw(Math.floor(n / 2)); return; }
  let i = 0, dir = 1, last = 0, pause = 0;
  const tick = (t) => {
    requestAnimationFrame(tick);
    if (t - last < 90 || t < pause) return;
    last = t; draw(i);
    i += dir;
    if (i === n - 1 || i === 0) { dir = -dir; pause = t + 700; }
  };
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ NIfTI header (client-side preview only; the server decides)
async function readHeader(file) {
  let bytes;
  if (/\.gz$/i.test(file.name)) {
    const reader = file.stream().pipeThrough(new DecompressionStream('gzip')).getReader();
    const chunks = []; let size = 0;
    try {
      while (size < 560) { const { value, done } = await reader.read(); if (done) break; chunks.push(value); size += value.length; }
    } catch { return { error: 'not a valid gzip file' }; }
    reader.cancel().catch(() => {});
    bytes = new Uint8Array(size); let o = 0; for (const c of chunks) { bytes.set(c, o); o += c.length; }
  } else {
    bytes = new Uint8Array(await file.slice(0, 560).arrayBuffer());
  }
  if (bytes.length < 348) return { error: 'too short to be NIfTI' };
  const dv = new DataView(bytes.buffer);
  for (const le of [true, false]) {
    const sz = dv.getInt32(0, le);
    if (sz === 348) {
      const nd = dv.getInt16(40, le), dim = [1, 2, 3, 4].map((k) => dv.getInt16(40 + 2 * k, le));
      const pix = [1, 2, 3].map((k) => dv.getFloat32(76 + 4 * k, le));
      return { ndim: nd, dim: dim.slice(0, nd), pix };
    }
    if (sz === 540 && bytes.length >= 540) {
      const nd = Number(dv.getBigInt64(16, le)), dim = [1, 2, 3, 4].map((k) => Number(dv.getBigInt64(16 + 8 * k, le)));
      const pix = [1, 2, 3].map((k) => dv.getFloat64(104 + 8 * k, le));
      return { ndim: nd, dim: dim.slice(0, nd), pix };
    }
  }
  return { error: 'no NIfTI header found' };
}

// Same token rules as the server's modality detector
function detectRole(name) {
  const n = name.replace(/\.nii(\.gz)?$/i, '').replace(/\.gz$/i, '');
  const b = (re) => new RegExp(`(?:^|[_\\-.\\s])(${re})(?:$|[_\\-.\\s])`, 'i').test(n);
  if (b('seg|label|labels|mask|gt|expert')) return 'expert';
  if (b('t1(?:ce|gd|c|post|\\+c|\\+gd|-ce|-gd|-post|_ce|_gd|_post)')) return 'T1Gd';
  if (b('t2[-_]?flair|flair')) return 'FLAIR';
  if (b('t2(?:w|_w|-w|weighted|_weighted)?')) return 'T2';
  if (b('t1(?:w|_w|-w|weighted|_weighted)?')) return 'T1';
  return '';
}

// ------------------------------------------------------------------ manifest
const state = { items: [], order: [...DECATHLON_ORDER], busy: false };

async function addFiles(fileList) {
  for (const f of fileList) {
    if (state.items.some((it) => it.file.name === f.name && it.file.size === f.size)) continue;
    const head = await readHeader(f);
    const is4 = head.ndim === 4 && head.dim[3] === 4;
    state.items.push({ file: f, head, role: is4 ? 'study4d' : detectRole(f.name) });
  }
  render();
}

function plan(items = state.items) {
  const live = items.filter((it) => it.role && it.role !== 'ignore');
  const four = live.filter((it) => it.role === 'study4d');
  const seqs = live.filter((it) => SEQS.includes(it.role));
  const experts = live.filter((it) => it.role === 'expert');
  const problems = [];
  for (const it of live) {
    const n = `“${it.file.name}”`, h = it.head;
    if (it.file.size > 1024 ** 3) problems.push(`${n} is larger than 1 GB.`);
    else if (h.error) problems.push(`${n} cannot be read as NIfTI (${h.error}).`);
    else if (it.role !== 'study4d') {
      const nd = h.ndim === 4 && h.dim[3] === 1 ? 3 : h.ndim;
      if (nd !== 3) problems.push(`${n} is ${nd}D (${h.dim.join(' × ')}); ${it.role === 'expert' ? 'the expert mask' : 'each sequence'} must be a 3D volume.`);
    }
  }
  if (items.some((it) => !it.role)) problems.push('Assign a sequence to every file (or mark it “ignore”).');
  if (experts.length > 1) problems.push('Only one expert mask can be attached.');
  if (four.length > 1) problems.push('Drop only one 4-channel study at a time.');
  if (four.length && seqs.length) problems.push('Use either one 4-channel file or four sequence files, not both.');
  if (four.length === 1) {
    if (new Set(state.order).size !== 4) problems.push('Each channel must map to a different sequence.');
    return { mode: '4d', four: four[0], seqs, expert: experts[0], problems };
  }
  const missing = SEQS.filter((q) => !seqs.some((it) => it.role === q));
  const dupes = SEQS.filter((q) => seqs.filter((it) => it.role === q).length > 1);
  if (missing.length) problems.push(`Missing sequence: ${missing.join(', ')}. Upload one 4-channel file, or all four of T1, T1Gd, T2 and FLAIR.`);
  if (dupes.length) problems.push(`More than one file for: ${dupes.join(', ')}.`);
  const shapes = new Set(seqs.filter((it) => it.head.dim && it.head.ndim >= 3).map((it) => it.head.dim.slice(0, 3).join('×')));
  if (shapes.size > 1) problems.push(`The sequences have different grids (${[...shapes].join(' vs ')}); they must be co-registered.`);
  return { mode: '4files', seqs, expert: experts[0], problems };
}

function render() {
  const m = $('#manifest'), has = state.items.length > 0;
  m.hidden = !has; $('#clear').hidden = !has;
  if (!has) { $('#run').disabled = true; $('#hint').hidden = true; return; }
  const roleOpts = (cur, is4) => (is4 ? ['study4d', 'ignore'] : [...SEQS, 'expert', 'ignore'])
    .map((r) => `<option value="${r}" ${r === cur ? 'selected' : ''}>${{ study4d: '4-channel study', expert: 'Expert mask', ignore: 'ignore' }[r] || r}</option>`).join('');
  const rows = state.items.map((it, k) => {
    const h = it.head, is4 = h.ndim === 4 && h.dim[3] === 4;
    const geom = h.error ? `<span class="bad">${esc(h.error)}</span>`
      : `${h.dim.join(' × ')}${h.pix ? ` &nbsp;·&nbsp; ${h.pix.map((p) => p.toFixed(2)).join(' × ')} mm` : ''}`;
    return `<tr><td><select data-k="${k}" aria-label="Role of ${esc(it.file.name)}">
        ${it.role ? '' : '<option value="" selected>choose…</option>'}${roleOpts(it.role, is4)}</select></td>
      <td>${esc(it.file.name)}</td><td class="mono">${geom}</td><td class="mono">${fmtMB(it.file.size)}</td></tr>`;
  }).join('');
  const p = plan();
  let channels = '';
  if (p.mode === '4d') {
    channels = `<p class="hint" style="margin-top:16px">Which sequence is in each channel? Defaults follow the Medical Segmentation Decathlon (FLAIR, T1, T1Gd, T2).</p>
      <div class="channels">${[0, 1, 2, 3].map((c) => `<label>CHANNEL ${c}<select data-ch="${c}">${SEQS.map((s) =>
        `<option ${state.order[c] === s ? 'selected' : ''}>${s}</option>`).join('')}</select></label>`).join('')}</div>`;
  }
  m.innerHTML = `<table><thead><tr><th>Role</th><th>File</th><th>Grid &amp; spacing (header)</th><th>Size</th></tr></thead><tbody>${rows}</tbody></table>${channels}`;
  m.querySelectorAll('select[data-k]').forEach((s) => s.addEventListener('change', () => { state.items[+s.dataset.k].role = s.value; render(); }));
  m.querySelectorAll('select[data-ch]').forEach((s) => s.addEventListener('change', () => { state.order[+s.dataset.ch] = s.value; render(); }));

  const hint = $('#hint');
  hint.hidden = false;
  hint.innerHTML = p.problems.length ? `<span class="bad">${p.problems.map(esc).join(' ')}</span>`
    : p.mode === '4d' ? `Ready: one 4-channel study (${state.order.join(', ')})${p.expert ? ' + expert mask' : ''}. It is split into four sequences on the server.`
      : `Ready: T1, T1Gd, T2 and FLAIR${p.expert ? ' + expert mask' : ''}.`;
  $('#run').disabled = state.busy;                      // problems are explained in a pop-up when Run is pressed
}

// ------------------------------------------------------------------ run
const steps = {};
document.querySelectorAll('#steps li').forEach((li) => { steps[li.dataset.step] = li; });
function step(name, cls, text = '') {
  for (const li of [steps[name], $(`#loader-steps li[data-step="${name}"]`)]) {
    if (!li) continue;
    li.className = cls; li.querySelector('time').textContent = text;
  }
  if (cls === 'run' && LOADER.stages[name]) loaderStage(name);
}
const secs = (t0) => ((performance.now() - t0) / 1000).toFixed(1) + ' s';

// ------------------------------------------------------------------ loading screen
// The server reports no progress during segmentation, so each stage creeps towards its end of the bar and jumps there
// when the stage really finishes. The brain only ever shows what the bar shows.
const SEG_KEY = 'voxels.lastSegSeconds';
const LOADER = {
  stages: {                                            // [from, to, time constant in s, heading]
    upload: [0, 0.2, 2, 'Uploading the files'],
    checks: [0.2, 0.32, 1.5, 'Checking structure and suitability'],
    segment: [0.32, 0.9, 6, 'Segmenting with SegResNet'],
    finish: [0.9, 0.985, 1.2, 'Measuring volumes and building surfaces'],
  },
  target: 0, shown: 0, stage: null, t0: 0, raf: 0, last: 0, brain: null, open: false,
};
const brainModule = $('#loader') ? import('/static/brainloader.js').catch(() => null) : null;

function loaderStage(name) {
  const [a, b, tau, text] = LOADER.stages[name];
  let k = tau;
  if (name === 'segment') {                            // ~85% of the way at the last run's duration
    let last = 14; try { last = Number(localStorage.getItem(SEG_KEY)) || 14; } catch {}
    k = Math.max(2, last / 1.9);
  }
  LOADER.stage = { a, b, k }; LOADER.t0 = performance.now();
  LOADER.target = Math.max(LOADER.target, a);
  $('#loader-stage').textContent = text; $('#loader-stage').classList.remove('done');
}
function loaderAt(p) { LOADER.target = Math.max(LOADER.target, p); }        // a real measurement (upload fraction, stage end)

function loaderFrame(t) {
  if (!LOADER.open) return;
  LOADER.raf = requestAnimationFrame(loaderFrame);
  const dt = Math.min(0.1, (t - (LOADER.last || t)) / 1000); LOADER.last = t;
  const s = LOADER.stage;
  if (s) LOADER.target = Math.max(LOADER.target, s.a + (s.b - s.a) * (1 - Math.exp(-(t - LOADER.t0) / 1000 / s.k)));
  const gap = LOADER.target - LOADER.shown;           // smooth, never backwards, at most ~60% of the brain per second
  if (gap > 0) LOADER.shown += Math.min(gap, Math.max(0.02, gap * 4) * dt, 0.6 * dt);
  $('#loader-bar').style.width = `${(LOADER.shown * 100).toFixed(1)}%`;
  $('#loader-pct').textContent = `${Math.floor(LOADER.shown * 100)}%`;
  LOADER.brain?.draw(LOADER.shown, t);
}

async function loaderOpen(fresh = true) {
  const el = $('#loader'); if (!el) return;
  if (fresh) {
    Object.assign(LOADER, { target: 0, shown: 0, stage: null });
    el.querySelectorAll('#loader-steps li').forEach((li) => { li.className = ''; li.querySelector('time').textContent = ''; });
  }
  el.classList.remove('closing'); el.hidden = false;
  document.body.classList.add('loading'); $('main').inert = true;
  LOADER.open = true; LOADER.last = 0;
  cancelAnimationFrame(LOADER.raf); LOADER.raf = requestAnimationFrame(loaderFrame);
  const mod = await brainModule;
  if (mod && !LOADER.brain) LOADER.brain = mod.createBrain($('#loader-canvas'));   // null without WebGL: bar and steps still work
}

function loaderClose() {
  const el = $('#loader'); if (!el || el.hidden) return Promise.resolve();
  el.classList.add('closing');
  return new Promise((resolve) => setTimeout(() => {
    LOADER.open = false; cancelAnimationFrame(LOADER.raf);
    el.hidden = true; el.classList.remove('closing');
    document.body.classList.remove('loading'); $('main').inert = false;
    resolve();
  }, 350));
}

// run complete: let the brain finish building, hold it a moment, then fade out
async function loaderFinish() {
  if (!LOADER.open) return;
  LOADER.stage = null; loaderAt(1);
  $('#loader-stage').textContent = 'Done'; $('#loader-stage').classList.add('done');
  await new Promise((resolve) => {
    const wait = () => (LOADER.shown >= 0.999 || !LOADER.open ? resolve() : setTimeout(wait, 50));
    wait();
  });
  await new Promise((r) => setTimeout(r, 650));
  await loaderClose();
}

function upload(url, form, onProgress) {
  return new Promise((resolve) => {
    const x = new XMLHttpRequest();
    x.open('POST', url);
    x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    x.upload.onload = () => onProgress(1);
    x.onload = () => { let body = {}; try { body = JSON.parse(x.responseText); } catch {} resolve({ ok: x.status < 300, status: x.status, body }); };
    x.onerror = () => resolve({ ok: false, status: 0, body: { message: 'Network error — is the server running?' } });
    x.send(form);
  });
}

function showError(code, msg) {
  const e = $('#error'); e.hidden = false;
  e.innerHTML = `<b>${esc(code || 'ERROR')}</b>${esc(msg || 'Something went wrong.')}`;
}

// Pop-up for refusals and warnings. Resolves true for "Continue", false for "Close"/"Stop" or Escape.
function popup({ kind, title, code, items, note, cont }) {
  const dlg = $('#alert-dialog');
  dlg.className = `alert-dialog ${kind}`;
  $('#alert-kind').textContent = kind === 'warn' ? 'Warning' : 'Input refused';
  $('#alert-title').textContent = title;
  $('#alert-code').textContent = code || '';
  $('#alert-code').hidden = !code;
  $('#alert-list').innerHTML = items.map((x) => `<li>${esc(x)}</li>`).join('');
  $('#alert-note').textContent = note || '';
  $('#alert-continue').hidden = !cont;
  $('#alert-close').textContent = cont ? 'Stop' : 'Close';
  return new Promise((resolve) => {
    const done = (v) => { dlg.close(); resolve(v); };
    $('#alert-continue').onclick = () => done(true);
    $('#alert-close').onclick = () => done(false);
    dlg.oncancel = (ev) => { ev.preventDefault(); done(false); };
    dlg.showModal();
    (cont ? $('#alert-continue') : $('#alert-close')).focus();
  });
}

const NOT_RUN = 'Nothing was sent to the model and no result was produced.';

async function run(p = plan()) {
  try {
    await runSteps(p);
  } catch (e) {                                         // e.g. the connection dropped mid-run
    await loaderClose();
    state.busy = false; refresh();
    popup({ kind: 'refuse', title: 'The run was interrupted', code: 'CLIENT_ERROR',
            items: [String(e && e.message || e)], note: 'No result was produced. Check that the server is running and try again.' });
  }
}

async function runSteps(p) {
  if (p.problems.length) {
    popup({ kind: 'refuse', title: 'These files cannot be analysed', code: 'CHECKED_IN_BROWSER', items: p.problems,
            note: `${NOT_RUN} Fix the files listed above and press Run again.` });
    return;
  }
  state.busy = true; refresh();
  $('#error').hidden = true; $('#result').hidden = true; $('#log').hidden = false;
  Object.keys(steps).forEach((k) => step(k, ''));
  loaderOpen();

  const form = new FormData();
  let url;
  if (p.mode === '4d') {
    url = '/api/upload/validate-4d';
    form.append('file', p.four.file); form.append('channel_order', state.order.join(','));
  } else {
    url = '/api/upload/validate';
    for (const it of p.seqs) form.append(it.role.toLowerCase(), it.file);
  }
  if (p.expert) form.append('expert', p.expert.file);

  const t0 = performance.now();
  step('upload', 'run');
  let t1;
  const up = await upload(url, form, (f) => {
    if (f < 1) { step('upload', 'run', `${Math.round(f * 100)}%`); loaderAt(f * 0.2); }
    else if (!t1) { t1 = performance.now(); step('upload', 'ok', secs(t0)); loaderAt(0.2); step('checks', 'run'); }
  });
  if (!t1) { t1 = performance.now(); step('upload', 'ok', secs(t0)); }
  if (!up.ok) {
    step('checks', 'fail', 'refused');
    await loaderClose();
    showError(up.body.error_code, up.body.message);
    state.busy = false; refresh();
    popup({ kind: 'refuse', title: 'The server refused this study', code: up.body.error_code || `HTTP ${up.status}`,
            items: [up.body.message || 'The upload could not be processed.'], note: NOT_RUN });
    return;
  }
  step('checks', 'ok', secs(t1)); loaderAt(0.32);
  const sid = up.body.study_id;
  rememberStudy(sid);
  const warns = up.body.warnings || [];
  if (warns.length) {
    await loaderClose();
    const go = await popup({
      kind: 'warn', title: 'Accepted, with suitability warnings', code: 'SUITABILITY_WARNING', items: warns, cont: true,
      note: 'The files are structurally valid, but they may not suit this model. Results may be unreliable. '
        + 'Continue to run the model anyway, or stop here (the upload is deleted automatically within 24 h).',
    });
    if (!go) { step('segment', 'fail', 'stopped'); state.busy = false; refresh(); return; }
    loaderOpen(false);
  }

  const t2 = performance.now();
  step('segment', 'run');
  const seg = await fetch('/api/segmentation/run', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ study_id: sid }),
  });
  const r = await seg.json().catch(() => ({}));
  if (!seg.ok) {
    step('segment', 'fail', 'failed');
    await loaderClose();
    showError(r.error_code || 'SEGMENTATION_FAILED', r.message);
    state.busy = false; refresh();
    popup({ kind: 'refuse', title: 'Segmentation did not complete', code: r.error_code || 'SEGMENTATION_FAILED',
            items: [r.message || 'The model could not process this study.'], note: 'No result was produced.' });
    return;
  }
  try { localStorage.setItem(SEG_KEY, ((performance.now() - t2) / 1000).toFixed(1)); } catch {}
  step('segment', 'ok', `${secs(t2)} · model ${r.provenance.inference_seconds}s`); loaderAt(0.9);

  const t3 = performance.now();
  step('finish', 'run');
  const [meta] = await Promise.all([
    fetch(`/api/viewer/${sid}/meta`).then((x) => x.json()),
    fetch(`/api/viewer/${sid}/meshes`).then((x) => x.ok), // warm the surface cache so the viewer opens instantly
  ]);
  step('finish', 'ok', `${secs(t3)} · total ${secs(t0)}`);
  await loaderFinish();
  showResult(sid, r, meta, up.body);
  state.busy = false; refresh();
}

function showResult(sid, r, meta, up) {
  const v = r.estimated_volumes, el = $('#result');
  const cell = (k, label, sw, major) => `<div class="${major ? 'major' : ''}"><div class="k">${sw ? `<i class="swatch ${sw}"></i>` : ''}${label}</div>
    <div class="v">${v[k].mL.toFixed(2)}<small>mL</small></div><div class="n">${v[k].voxels.toLocaleString('en-US')} vox × ${v[k].voxel_volume_mm3.toFixed(3)} mm³</div></div>`;
  const warn = r.warnings.length ? `<div class="warnbox"><b>Suitability warnings</b><ul>${r.warnings.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div>` : '';
  const ag = meta.agreement ? `<div class="agree"><span class="eyebrow">Agreement with your expert mask (this case)</span>
    <table><tr><th>Region</th><th>Dice</th><th>Sensitivity</th><th>HD95</th></tr>${Object.entries(meta.agreement).map(([k, a]) =>
      `<tr><td>${k}</td><td>${a.dice.toFixed(3)}</td><td>${a.sensitivity === null ? '—' : a.sensitivity.toFixed(3)}</td><td>${a.hd95_mm === null ? '—' : a.hd95_mm.toFixed(1) + ' mm'}</td></tr>`).join('')}</table></div>` : '';
  const src = up.metadata.source_4d ? ` · split from ${esc(up.metadata.source_4d.file)} (${up.metadata.source_4d.channel_order.join(', ')})` : '';
  el.innerHTML = `
    <div class="result-head">
      <h3>Suspected regions predicted by the model, <em>estimated volumes</em></h3>
      <span class="mono small">study ${esc(sid)}</span>
    </div>
    <div class="vols">
      ${cell('WT', 'WT · 1+2+4', '', true)}${cell('TC', 'TC · 1+4', '', true)}${cell('ET', 'ET · 4', 's-et', true)}
      ${cell('NCR', 'NCR · 1', 's-ncr')}${cell('ED', 'ED · 2', 's-ed')}
    </div>
    ${warn}
    ${r.uncertainty ? `<p class="trust"><span class="eyebrow">How much to trust it</span><br>
      Re-run on the scan mirrored along each axis (${r.uncertainty.passes} passes), WT lands between
      <b>${r.uncertainty.regions.WT.volume_mL_min.toFixed(1)}</b> and <b>${r.uncertainty.regions.WT.volume_mL_max.toFixed(1)} mL</b>;
      ${(r.uncertainty.wt_uncertain_fraction * 100).toFixed(1)}% of its voxels are disputed between passes.
      The viewer shows where, and can run a modality ablation.</p>` : ''}
    ${ag}
    <div class="result-actions">
      <a class="btn" href="${r.viewer_url}">Open 3D &amp; slice viewer <span class="arr">→</span></a>
      <a class="btn ghost" href="${r.report_url}" target="_blank">Report</a>
      <a class="btn ghost" href="/api/report/${encodeURIComponent(sid)}.pdf" target="_blank">PDF</a>
      <a class="btn ghost" href="${r.prediction_url}">Label map (.nii.gz)</a>
    </div>
    <p class="prov">${esc(r.model)} · ${esc(r.provenance.device)} · ${r.provenance.inference_seconds}s inference ·
      grid ${r.shape.join('×')} @ ${r.spacing_mm.map((s) => s.toFixed(2)).join('×')} mm · commit ${esc(String(r.provenance.code_commit).slice(0, 7))}${src}<br>
      V = N × Δx·Δy·Δz; mL = mm³ ÷ 1000. Not a diagnosis.</p>`;
  el.hidden = false;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ------------------------------------------------------------------ four separate files: one slot per sequence
// Same checks, run log, result and pop-ups as the drop zone; only the way files are chosen differs.
const SLOT_ROLES = [...SEQS, 'expert'];
const slots = {};                                       // role -> { file, head }
const slotsEl = $('#slots');
const slotItems = () => SLOT_ROLES.filter((r) => slots[r]).map((r) => ({ ...slots[r], role: r }));

function refresh() { render(); renderSlots(); }

async function setSlot(role, file) {
  if (!file) return;
  slots[role] = { file, head: await readHeader(file) };
  renderSlots();
}

function renderSlots() {
  if (!slotsEl) return;
  slotsEl.querySelectorAll('.slot').forEach((el) => {
    const s = slots[el.dataset.role], h = s && s.head;
    el.classList.toggle('filled', !!s);
    el.classList.toggle('bad', !!(h && h.error));
    el.querySelector('.slot-file').textContent = s ? s.file.name : (el.dataset.role === 'expert' ? 'browse (optional)' : 'browse or drop a file');
    el.querySelector('.slot-meta').innerHTML = !s ? '' : h.error ? `<span class="bad">${esc(h.error)}</span>`
      : `${h.dim.join(' × ')} &nbsp;·&nbsp; ${fmtMB(s.file.size)}`;
    el.querySelector('.slot-remove').hidden = !s;
  });
  const items = slotItems(), has = items.length > 0, hint = $('#four-hint');
  $('#four-clear').hidden = !has;
  hint.hidden = !has;
  if (has) {
    const p = plan(items);
    hint.innerHTML = p.problems.length ? `<span class="bad">${p.problems.map(esc).join(' ')}</span>`
      : `Ready: T1, T1Gd, T2 and FLAIR${p.expert ? ' + expert mask' : ''}.`;
  }
  $('#four-run').disabled = state.busy || !has;
}

if (slotsEl) {
  slotsEl.querySelectorAll('.slot').forEach((el) => {
    const role = el.dataset.role, inp = el.querySelector('input');
    inp.addEventListener('change', () => { setSlot(role, inp.files[0]); inp.value = ''; });
    ['dragenter', 'dragover'].forEach((e) => el.addEventListener(e, (ev) => { ev.preventDefault(); el.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((e) => el.addEventListener(e, () => el.classList.remove('over')));
    el.addEventListener('drop', (ev) => { ev.preventDefault(); setSlot(role, ev.dataTransfer.files[0]); });
    el.querySelector('.slot-remove').addEventListener('click', () => { delete slots[role]; renderSlots(); });
  });
  $('#four-run').addEventListener('click', () => run(plan(slotItems())));
  $('#four-clear').addEventListener('click', () => {
    SLOT_ROLES.forEach((r) => delete slots[r]);
    $('#error').hidden = true; $('#result').hidden = true;
    $('#log').hidden = true; Object.keys(steps).forEach((k) => step(k, '')); renderSlots();
  });
  renderSlots();
}

// ------------------------------------------------------------------ wiring
const drop = $('#drop'), input = $('#file-input');
if (drop) {
  input.addEventListener('change', () => { addFiles([...input.files]); input.value = ''; });
  ['dragenter', 'dragover'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, () => drop.classList.remove('over')));
  drop.addEventListener('drop', (ev) => { ev.preventDefault(); addFiles([...ev.dataTransfer.files]); });
  $('#run').addEventListener('click', () => run());
  $('#clear').addEventListener('click', () => {
    state.items = []; state.order = [...DECATHLON_ORDER];
    $('#error').hidden = true; $('#result').hidden = true;
    $('#log').hidden = true; Object.keys(steps).forEach((k) => step(k, '')); render();
  });
}

// ------------------------------------------------------------------ privacy & security: retention list + countdown
// This browser remembers the IDs of studies it uploaded (and nothing else), so only it is shown their file names.
const MINE_KEY = 'neuroscope.myStudies';
function myStudies() {
  try { const v = JSON.parse(localStorage.getItem(MINE_KEY) || '[]'); return Array.isArray(v) ? v : []; } catch { return []; }
}
function setMyStudies(ids) { try { localStorage.setItem(MINE_KEY, JSON.stringify(ids.slice(-50))); } catch {} }
function rememberStudy(sid) { const ids = myStudies(); if (!ids.includes(sid)) setMyStudies([...ids, sid]); }

const hms = (s) => {
  s = Math.max(0, Math.floor(s));
  return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map((n) => String(n).padStart(2, '0')).join(':');
};

async function startPrivacy() {
  const list = $('#pv-list');
  if (!list) return;
  const ring = $('#pv-ring'), clock = $('#pv-clock'), label = $('#pv-clock-label'), note = $('#pv-clock-note');
  $('#pv-ticks').innerHTML = Array.from({ length: 24 }, (_, h) => {
    const a = (h / 24) * 2 * Math.PI, r1 = h % 6 ? 44 : 40;
    return `<line x1="${60 + r1 * Math.sin(a)}" y1="${60 - r1 * Math.cos(a)}" x2="${60 + 47 * Math.sin(a)}" y2="${60 - 47 * Math.cos(a)}"/>`;
  }).join('');
  let data = null, mine = {}, offset = 0;
  const DAY = 24 * 3600;

  async function refresh() {
    try {
      data = await (await fetch('/api/privacy/retention', { cache: 'no-store' })).json();
      offset = data.now_ts * 1000 - Date.now();
      $('#pv-ttl').textContent = data.ttl_hours;
      $('#pv-sweep').textContent = data.sweep_minutes;
      const ids = myStudies();
      if (ids.length) {
        const r = await (await fetch('/api/privacy/retention/mine', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ study_ids: ids }),
        })).json();
        setMyStudies(r.live_ids);
        mine = Object.fromEntries(r.studies.map((x) => [x.ref, x.files]));
      } else mine = {};
      render();
    } catch {
      list.innerHTML = '<tr><td colspan="5">Retention data unavailable (is the server running?).</td></tr>';
    }
  }

  function render() {
    if (!data.studies.length) {
      list.innerHTML = '<tr><td colspan="5">Nothing is stored right now.</td></tr>';
      return;
    }
    list.innerHTML = data.studies.map((s) => {
      const files = mine[s.ref];
      const detail = files ? `<tr class="pv-files"><td></td><td colspan="4"><details><summary>${files.length} files from this browser, all deleted together</summary>
        <ul>${files.map((f) => `<li><span>${esc(f.name)}</span><span class="mono">${fmtMB(f.bytes)}</span></li>`).join('')}</ul></details></td></tr>` : '';
      return `<tr><td class="mono">${esc(s.ref)}</td><td>${s.kind === 'upload' ? (files ? 'Your upload' : 'Upload') : 'Precomputed sample'}</td>
        <td>${s.n_files}</td><td>${fmtMB(s.total_bytes)}</td><td class="mono" data-exp="${s.expires_ts}">—</td></tr>${detail}`;
    }).join('');
    tick();
  }

  function tick() {
    if (!data) return;
    const now = (Date.now() + offset) / 1000;
    list.querySelectorAll('td[data-exp]').forEach((td) => {
      const left = Number(td.dataset.exp) - now;
      td.textContent = left > 0 ? hms(left) : `due · within ${data.sweep_minutes} min`;
    });
    const next = data.studies.find((s) => s.expires_ts - now > -data.sweep_minutes * 60);
    if (next) {
      const left = next.expires_ts - now;
      label.textContent = 'Next deletion in';
      clock.textContent = left > 0 ? hms(left) : '00:00:00';
      note.textContent = `study ${next.ref} · ${next.n_files} files · ${fmtMB(next.total_bytes)}`;
      ring.style.strokeDashoffset = 1000 * (1 - Math.max(0, left) / DAY);
    } else {
      label.textContent = 'Each new upload is kept for';
      clock.textContent = hms(data.ttl_hours * 3600);
      note.textContent = 'nothing stored right now';
      ring.style.strokeDashoffset = 1000 * (1 - (data.ttl_hours * 3600) / DAY);
    }
  }

  await refresh();
  setInterval(tick, 1000);
  setInterval(refresh, 60000);
}

loadAbout();
startPlate();
startPrivacy();
