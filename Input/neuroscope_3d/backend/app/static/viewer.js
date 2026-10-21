// Voxels viewer: 3D meshes in world mm + linked axial/coronal/sagittal slices with one crosshair.
// Every slice is reachable: per-view slider (with a strip showing where the predicted region is), step and play
// buttons, mouse wheel, arrow keys over a view, and an all-slices gallery that moves the crosshair on click.
// Volumes arrive in canonical RAS voxel order (axis0 -> Right, axis1 -> Anterior, axis2 -> Superior), x fastest.
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const studyId = decodeURIComponent(location.pathname.split('/').filter(Boolean).pop());
const $ = (s) => document.querySelector(s);
const API = `/api/viewer/${encodeURIComponent(studyId)}`;

const LABEL_RGB = { 1: [214, 69, 44], 2: [120, 168, 112], 4: [232, 176, 48] };
const LABEL_NAME = { 0: 'background', 1: 'NCR (1)', 2: 'ED (2)', 4: 'ET (4)' };
const AGREE_RGB = { both: [120, 168, 112], fp: [214, 69, 44], fn: [111, 163, 181] };
// how many of the 4 test-time-augmentation passes call a voxel WT
const UNCERT_RGB = { 1: [214, 69, 44], 2: [232, 176, 48], 3: [226, 214, 160], 4: [120, 168, 112] };
const REGION_CODES = { NCR: [1], ED: [2], ET: [4], TC: [1, 4], WT: [1, 2, 4] };

const S = {
  meta: null, X: 0, Y: 0, Z: 0, sp: [1, 1, 1], A: null, Ainv: null, Oinv: null,
  vols: {}, cross: [0, 0, 0], bg: 'flair', overlay: 'pred', alpha: 0.5, codes: new Set([1, 2, 4]),
  prof: null, hover: null, cine: null,
};

// ------------------------------------------------------------------ helpers
function status(msg, err = false) { const el = $('#status'); el.textContent = msg; el.classList.toggle('err', err); }
async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.message || `${r.status} ${r.statusText}`);
  return j;
}
async function getVol(name) {
  if (S.vols[name]) return S.vols[name];
  const r = await fetch(`${API}/volume/${name}`);
  if (!r.ok) throw new Error(`Could not load ${name} volume (${r.status})`);
  const a = new Uint8Array(await r.arrayBuffer());
  if (a.length !== S.X * S.Y * S.Z) throw new Error(`${name}: size ${a.length} != ${S.X}x${S.Y}x${S.Z}`);
  return (S.vols[name] = a);
}
const mv = (M, v) => M.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]);
function inv4(m) {
  const a = m.map((r, i) => [...r, ...[0, 1, 2, 3].map((j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < 4; c++) {
    let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    const d = a[c][c]; for (let j = 0; j < 8; j++) a[c][j] /= d;
    for (let r = 0; r < 4; r++) if (r !== c) { const f = a[r][c]; for (let j = 0; j < 8; j++) a[r][j] -= f * a[c][j]; }
  }
  return a.map((r) => r.slice(4));
}
const idx = (x, y, z) => x + S.X * (y + S.Y * z);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (v, d = 1) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));
const toWorld = (v) => mv(S.A, [v[0], v[1], v[2], 1]).slice(0, 3);

// ------------------------------------------------------------------ slice views
// Radiological display: axial/coronal show patient Right on screen-left; sagittal shows Anterior on the left.
const VIEWS = {
  axial: { axis: 2, labels: ['R', 'L', 'A', 'P'], wh: () => [S.X, S.Y], mm: () => [S.X * S.sp[0], S.Y * S.sp[1]],
    toVox: (u, v) => [S.X - 1 - u, S.Y - 1 - v, S.cross[2]], fromVox: (c) => [S.X - 1 - c[0], S.Y - 1 - c[1]] },
  coronal: { axis: 1, labels: ['R', 'L', 'S', 'I'], wh: () => [S.X, S.Z], mm: () => [S.X * S.sp[0], S.Z * S.sp[2]],
    toVox: (u, v) => [S.X - 1 - u, S.cross[1], S.Z - 1 - v], fromVox: (c) => [S.X - 1 - c[0], S.Z - 1 - c[2]] },
  sagittal: { axis: 0, labels: ['A', 'P', 'S', 'I'], wh: () => [S.Y, S.Z], mm: () => [S.Y * S.sp[1], S.Z * S.sp[2]],
    toVox: (u, v) => [S.cross[0], S.Y - 1 - u, S.Z - 1 - v], fromVox: (c) => [S.Y - 1 - c[1], S.Z - 1 - c[2]] },
};

function buildSlicePanels() {
  for (const panel of document.querySelectorAll('.panel.slice')) {
    const name = panel.dataset.view, V = VIEWS[name], box = panel.querySelector('.sbox');
    box.innerHTML = '';
    const stack = document.createElement('div'); stack.className = 'stack';
    const img = document.createElement('canvas'), cross = document.createElement('canvas');
    const [w, h] = V.wh(); img.width = w; img.height = h;
    stack.append(img, cross); box.append(stack);
    ['l', 'r', 't', 'b'].forEach((p, i) => {
      const s = document.createElement('span'); s.className = `olabel ${p}`; s.textContent = V.labels[i]; box.append(s);
    });
    Object.assign(V, { panel, box, stack, img, cross, ctx: img.getContext('2d'), cctx: cross.getContext('2d'),
      imageData: img.getContext('2d').createImageData(w, h) });

    let dragging = false;
    const pick = (ev) => {
      const r = cross.getBoundingClientRect();
      const u = clamp(Math.floor(((ev.clientX - r.left) / r.width) * w), 0, w - 1);
      const v = clamp(Math.floor(((ev.clientY - r.top) / r.height) * h), 0, h - 1);
      setCross(V.toVox(u, v));
    };
    cross.addEventListener('pointerdown', (ev) => { dragging = true; cross.setPointerCapture(ev.pointerId); pick(ev); });
    cross.addEventListener('pointermove', (ev) => { if (dragging) pick(ev); });
    cross.addEventListener('pointerup', () => { dragging = false; });
    cross.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const c = [...S.cross]; c[V.axis] += ev.deltaY < 0 ? 1 : -1; setCross(c);
    }, { passive: false });
    panel.addEventListener('pointerenter', () => { S.hover = name; });
    panel.addEventListener('pointerleave', () => { if (S.hover === name) S.hover = null; });
    buildSliceControls(name, V, panel);
    if (!V.ro) { V.ro = new ResizeObserver(() => layoutSlices()); V.ro.observe(box); }   // follows enlarge / full screen
  }
  layoutSlices();
}

// ------------------------------------------------------------------ enlarge / full screen (every view)
// Enlarge: the chosen view takes most of the grid; the other three stay live beside it and the side controls stay
// where they are. Full screen: the browser's full-screen mode for that one view (Esc leaves it).
const ICON = {
  enlarge: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 4h6v6M10 20H4v-6M20 4l-7 7M4 20l7-7"/></svg>',
  restore: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 4l-6 6M14 4v6h6M4 20l6-6M10 20v-6H4"/></svg>',
  full: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  unfull: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
};
const panelName = (p) => (p.dataset.view ? p.dataset.view[0].toUpperCase() + p.dataset.view.slice(1) : '3D');

function buildPanelTools() {
  for (const panel of document.querySelectorAll('.views .panel')) {
    const tools = document.createElement('div');
    tools.className = 'ptools';
    tools.innerHTML = '<button type="button" class="pt-enlarge"></button><button type="button" class="pt-full"></button>';
    panel.append(tools);
    tools.querySelector('.pt-enlarge').addEventListener('click', () => setFocus(panel.classList.contains('big') ? null : panel));
    tools.querySelector('.pt-full').addEventListener('click', () => toggleFullscreen(panel));
    panel.addEventListener('dblclick', (ev) => {                       // double-click a view to enlarge / restore
      if (ev.target.closest('button, input, .sctl')) return;
      setFocus(panel.classList.contains('big') ? null : panel);
    });
    panel.addEventListener('pointerenter', () => { S.hoverPanel = panel; });
    panel.addEventListener('pointerleave', () => { if (S.hoverPanel === panel) S.hoverPanel = null; });
  }
  if (!document.fullscreenEnabled) document.querySelectorAll('.pt-full').forEach((b) => { b.hidden = true; });
  updatePanelTools();
}

function updatePanelTools() {
  for (const panel of document.querySelectorAll('.views .panel')) {
    const n = panelName(panel), big = panel.classList.contains('big'), full = document.fullscreenElement === panel;
    const e = panel.querySelector('.pt-enlarge'), f = panel.querySelector('.pt-full');
    if (!e) continue;
    e.innerHTML = big ? ICON.restore : ICON.enlarge;
    e.title = big ? 'Back to the four-view grid (Esc)' : `Enlarge the ${n} view (F)`;
    e.setAttribute('aria-label', e.title); e.setAttribute('aria-pressed', String(big));
    f.innerHTML = full ? ICON.unfull : ICON.full;
    f.title = full ? 'Leave full screen (Esc)' : `Show the ${n} view full screen`;
    f.setAttribute('aria-label', f.title);
  }
}

function setFocus(panel) {
  const views = $('.views');
  views.querySelectorAll('.panel').forEach((p) => p.classList.toggle('big', p === panel));
  views.classList.toggle('focus', !!panel);
  updatePanelTools();
  if (panel) views.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function toggleFullscreen(panel) {
  if (document.fullscreenElement === panel) document.exitFullscreen();
  else panel.requestFullscreen?.().catch((err) => status(`Full screen is not available: ${err.message}`, true));
}
document.addEventListener('fullscreenchange', () => { updatePanelTools(); layoutSlices(); });

// Slider + step buttons + play under each slice view; the strip behind the slider shows, per slice, how much of
// the predicted region (WT) it contains, so the tumour's extent along that axis is visible at a glance.
function buildSliceControls(name, V, panel) {
  panel.querySelector('.sctl')?.remove();
  const n = [S.X, S.Y, S.Z][V.axis], ctl = document.createElement('div');
  ctl.className = 'sctl';
  ctl.innerHTML = `<button type="button" class="step" data-d="-1" aria-label="Previous ${name} slice">‹</button>
    <div class="track"><canvas class="prof" aria-hidden="true"></canvas>
      <input type="range" min="1" max="${n}" step="1" value="${S.cross[V.axis] + 1}" aria-label="${name} slice number"></div>
    <button type="button" class="step" data-d="1" aria-label="Next ${name} slice">›</button>
    <button type="button" class="play" aria-label="Play through the ${name} slices" title="Play through the slices">▶</button>`;
  panel.append(ctl);
  V.range = ctl.querySelector('input');
  V.play = ctl.querySelector('.play');
  V.range.addEventListener('input', () => setSlice(V, Number(V.range.value) - 1));
  ctl.querySelectorAll('.step').forEach((b) => b.addEventListener('click', () => setSlice(V, S.cross[V.axis] + Number(b.dataset.d))));
  V.play.addEventListener('click', () => toggleCine(V));
  drawProfile(V, ctl.querySelector('.prof'));
}

function drawProfile(V, c) {
  const n = [S.X, S.Y, S.Z][V.axis], prof = S.prof && S.prof[V.axis];
  c.width = n; c.height = 20;
  const g = c.getContext('2d'); g.clearRect(0, 0, n, 20);
  g.fillStyle = 'rgba(255,255,255,0.08)'; g.fillRect(0, 9, n, 2);
  if (!prof) return;
  let max = 0; for (const v of prof) max = Math.max(max, v);
  if (!max) return;
  g.fillStyle = 'rgba(232,176,48,0.85)';
  for (let k = 0; k < n; k++) if (prof[k]) { const hgt = Math.max(2, Math.round((prof[k] / max) * 18)); g.fillRect(k, 10 - hgt / 2, 1, hgt); }
}

function setSlice(V, k) {
  const c = [...S.cross]; c[V.axis] = k; setCross(c);
}

function toggleCine(V) {
  const same = S.cine && S.cine.V === V;
  stopCine();
  if (same) return;
  const n = [S.X, S.Y, S.Z][V.axis];
  S.cine = { V, timer: setInterval(() => setSlice(V, (S.cross[V.axis] + 1) % n), 110) };
  V.play.textContent = '❚❚'; V.play.setAttribute('aria-label', 'Pause'); V.play.title = 'Pause';
}

function stopCine() {
  if (!S.cine) return;
  clearInterval(S.cine.timer);
  const P = S.cine.V.play; P.textContent = '▶'; P.setAttribute('aria-label', 'Play through the slices'); P.title = 'Play through the slices';
  S.cine = null;
}

// Per-axis count of predicted (WT) voxels in every slice; area in cm² = count × in-plane voxel area ÷ 100.
function computeProfiles() {
  const p = S.vols.pred;
  if (!p) { S.prof = null; return; }
  const prof = [new Uint32Array(S.X), new Uint32Array(S.Y), new Uint32Array(S.Z)];
  let i = 0;
  for (let z = 0; z < S.Z; z++) for (let y = 0; y < S.Y; y++) for (let x = 0; x < S.X; x++, i++) {
    if (p[i]) { prof[0][x]++; prof[1][y]++; prof[2][z]++; }
  }
  S.prof = prof;
}
const inPlaneMM2 = (axis) => (axis === 2 ? S.sp[0] * S.sp[1] : axis === 1 ? S.sp[0] * S.sp[2] : S.sp[1] * S.sp[2]);
const sliceAreaCM2 = (axis, k) => (S.prof ? (S.prof[axis][k] * inPlaneMM2(axis)) / 100 : null);

function layoutSlices() {
  for (const V of Object.values(VIEWS)) {
    if (!V.box) continue;
    const [mw, mh] = V.mm();
    const bw = V.box.clientWidth - 8, bh = V.box.clientHeight - 30;
    const k = Math.max(0.1, Math.min(bw / mw, bh / mh));
    V.stack.style.width = `${mw * k}px`; V.stack.style.height = `${mh * k}px`;
    V.cross.width = Math.round(mw * k * devicePixelRatio); V.cross.height = Math.round(mh * k * devicePixelRatio);
    drawCross(V);
  }
}

function overlayLUTs() {
  const inCodes = new Uint8Array(256); for (const c of S.codes) inCodes[c] = 1;
  const region = new Uint8Array(256);
  if (S.overlay.startsWith('agree-')) for (const c of REGION_CODES[S.overlay.slice(6)]) region[c] = 1;
  return { inCodes, region };
}

// Display settings for shading: background volume, overlay mode, opacity and colour look-ups.
function shadeCtx() {
  const { inCodes, region } = overlayLUTs();
  return { bg: S.vols[S.bg], pred: S.vols.pred, expert: S.vols.expert, uncert: S.vols.uncert, a: S.alpha, mode: S.overlay,
    inCodes, region };
}

// Writes the RGBA of voxel i (background grey blended with the overlay colour) into d at offset p.
function shade(d, p, i, k) {
  const g = k.bg ? k.bg[i] : 0, mode = k.mode;
  let c = null;
  if (mode === 'uncert') c = k.uncert && k.uncert[i] ? UNCERT_RGB[k.uncert[i]] : null;
  else if (mode === 'pred' && k.pred && k.inCodes[k.pred[i]]) c = LABEL_RGB[k.pred[i]];
  else if (mode === 'expert' && k.expert && k.inCodes[k.expert[i]]) c = LABEL_RGB[k.expert[i]];
  else if (mode.startsWith('agree-') && k.pred && k.expert) {
    const pm = k.region[k.pred[i]], em = k.region[k.expert[i]];
    c = pm && em ? AGREE_RGB.both : pm ? AGREE_RGB.fp : em ? AGREE_RGB.fn : null;
  }
  if (c) { d[p] = g + (c[0] - g) * k.a; d[p + 1] = g + (c[1] - g) * k.a; d[p + 2] = g + (c[2] - g) * k.a; }
  else { d[p] = d[p + 1] = d[p + 2] = g; }
  d[p + 3] = 255;
}

function renderSlice(V) {
  const [w, h] = V.wh(), d = V.imageData.data, k = shadeCtx();
  let p = 0;
  for (let v = 0; v < h; v++) {
    for (let u = 0; u < w; u++, p += 4) {
      const [x, y, z] = V.toVox(u, v);
      shade(d, p, idx(x, y, z), k);
    }
  }
  V.ctx.putImageData(V.imageData, 0, 0);
  const s = S.cross[V.axis], area = sliceAreaCM2(V.axis, s);
  V.panel.querySelector('.idx').textContent = `slice ${s + 1} / ${[S.X, S.Y, S.Z][V.axis]}` +
    (area === null ? '' : ` · WT ${area.toFixed(1)} cm²`);
  if (V.range) V.range.value = s + 1;
  drawCross(V);
}

function drawCross(V) {
  if (!V.cctx) return;
  const [w, h] = V.wh(), [u, v] = V.fromVox(S.cross), cw = V.cross.width, ch = V.cross.height;
  const px = ((u + 0.5) / w) * cw, py = ((v + 0.5) / h) * ch, g = V.cctx;
  g.clearRect(0, 0, cw, ch);
  g.strokeStyle = 'rgba(242,237,227,0.8)'; g.lineWidth = Math.max(1, devicePixelRatio);
  g.beginPath(); g.moveTo(px, 0); g.lineTo(px, ch); g.moveTo(0, py); g.lineTo(cw, py); g.stroke();
}

const renderAllSlices = () => Object.values(VIEWS).forEach((V) => V.ctx && renderSlice(V));

// ------------------------------------------------------------------ crosshair
function setCross(c) {
  S.cross = [clamp(Math.round(c[0]), 0, S.X - 1), clamp(Math.round(c[1]), 0, S.Y - 1), clamp(Math.round(c[2]), 0, S.Z - 1)];
  renderAllSlices();
  update3DCross();
  markGalleryCurrent();
  const w = toWorld(S.cross), o = mv(S.Oinv, [...w, 1]).map((x) => Math.round(x));
  const i = idx(...S.cross);
  $('#c-vox').textContent = `i ${o[0]}, j ${o[1]}, k ${o[2]}`;
  $('#c-world').textContent = `x ${fmt(w[0])}, y ${fmt(w[1])}, z ${fmt(w[2])}`;
  $('#c-pred').textContent = S.vols.pred ? LABEL_NAME[S.vols.pred[i]] ?? S.vols.pred[i] : '—';
  if (S.vols.expert) $('#c-expert').textContent = LABEL_NAME[S.vols.expert[i]] ?? S.vols.expert[i];
  $('#c-int').textContent = S.vols[S.bg] ? `${S.bg.toUpperCase()} display value ${S.vols[S.bg][i]} / 255` : '—';
}

// ------------------------------------------------------------------ all-slices gallery
// One small image per slice along the chosen axis, drawn only when scrolled into view. Click one to move the
// crosshair to that slice; the three views and the 3D view follow.
const G = { view: 'axial', thumbs: [], io: null };

function buildGallery() {
  const box = $('#gallery');
  if (!box) return;
  const name = $('#g-axis').value, V = VIEWS[name], n = [S.X, S.Y, S.Z][V.axis];
  const onlyRegion = $('#g-filter').value === 'region';
  const size = Number($('#g-size')?.value || 104);
  box.style.setProperty('--thumb', `${size}px`);
  const [w, h] = V.wh(), [mw, mh] = V.mm(), step = Math.max(1, Math.ceil(Math.max(w, h) / (size * 1.08)));
  if (G.io) G.io.disconnect();
  G.io = new IntersectionObserver((ents) => ents.forEach((e) => e.isIntersecting && drawThumb(e.target)),
    { root: box, rootMargin: '160px' });
  box.innerHTML = '';
  G.view = name; G.thumbs = []; G.step = step;
  for (let k = 0; k < n; k++) {
    const has = S.prof ? S.prof[V.axis][k] > 0 : false;
    if (onlyRegion && !has) continue;
    const area = sliceAreaCM2(V.axis, k);
    const b = document.createElement('div');
    b.className = `thumb${has ? ' has' : ''}`; b.dataset.k = k;
    b.innerHTML = `<button type="button" class="thumb-go"
        aria-label="${name} slice ${k + 1}${area ? `, predicted region ${area.toFixed(1)} cm²` : ''}: open it large">
        <canvas width="${Math.floor(w / step)}" height="${Math.floor(h / step)}" style="aspect-ratio:${mw} / ${mh}"></canvas>
        <span>${k + 1}${area ? ` · ${area.toFixed(1)} cm²` : ''}</span></button>
      <button type="button" class="thumb-zoom" title="Enlarge this slice" aria-label="Enlarge ${name} slice ${k + 1}">${ICON.enlarge}</button>`;
    b.querySelector('.thumb-go').addEventListener('click', () => openLightbox(name, k));
    b.querySelector('.thumb-zoom').addEventListener('click', () => openLightbox(name, k));
    box.append(b); G.thumbs.push(b); G.io.observe(b);
  }
  $('#g-count').textContent = G.thumbs.length
    ? `${G.thumbs.length} of ${n} ${name} slices` : `No ${name} slice contains the predicted region.`;
  markGalleryCurrent();
}

function drawThumb(b) {
  if (b.dataset.drawn === '1') return;
  const V = VIEWS[G.view], sl = Number(b.dataset.k), c = b.querySelector('canvas'), st = G.step;
  const tw = c.width, th = c.height, ctx = c.getContext('2d'), img = ctx.createImageData(tw, th), d = img.data, k = shadeCtx();
  let p = 0;
  for (let v = 0; v < th; v++) {
    for (let u = 0; u < tw; u++, p += 4) {
      const vox = V.toVox(u * st, v * st); vox[V.axis] = sl;
      shade(d, p, idx(vox[0], vox[1], vox[2]), k);
    }
  }
  ctx.putImageData(img, 0, 0);
  b.dataset.drawn = '1';
}

// Redraw thumbnails after the background, overlay, opacity or label choice changes.
function refreshGallery() {
  if (!G.io) return;
  G.io.disconnect();
  for (const b of G.thumbs) { b.dataset.drawn = ''; G.io.observe(b); }
}

function markGalleryCurrent() {
  if (!G.thumbs.length) return;
  const k = String(S.cross[VIEWS[G.view].axis]);
  for (const b of G.thumbs) b.classList.toggle('current', b.dataset.k === k);
}

// Lightbox: one slice at full resolution, same background and overlay; ‹ › or arrow keys step through every slice
// (the views and the 3D crosshair follow). Wheel or + / − zooms, drag pans, Fit resets.
const LB = { view: 'axial', k: 0, z: 1, x: 0, y: 0 };

function openLightbox(name, k) {
  const dlg = $('#lightbox');
  if (!dlg) return;
  LB.view = name;
  zoomLightbox(1, true);
  drawLightbox(k);
  if (!dlg.open) dlg.showModal();
}

function zoomLightbox(z, reset) {
  LB.z = clamp(z, 1, 8);
  if (reset || LB.z === 1) { LB.x = 0; LB.y = 0; }
  $('#lb-stack').style.transform = `translate(${LB.x}px, ${LB.y}px) scale(${LB.z})`;
  $('#lb-zoom').textContent = `${Math.round(LB.z * 100)}%`;
  $('#lb-stack').classList.toggle('zoomed', LB.z > 1);
}

function drawLightbox(k) {
  const V = VIEWS[LB.view], n = [S.X, S.Y, S.Z][V.axis];
  LB.k = clamp(k, 0, n - 1);
  if (S.cross[V.axis] !== LB.k) setSlice(V, LB.k);                  // keep the views and 3D in step
  const [w, h] = V.wh(), [mw, mh] = V.mm(), c = $('#lb-canvas'), ctx = c.getContext('2d');
  c.width = w; c.height = h; c.style.aspectRatio = `${mw} / ${mh}`;
  const img = ctx.createImageData(w, h), d = img.data, sc = shadeCtx();
  let p = 0;
  for (let v = 0; v < h; v++) {
    for (let u = 0; u < w; u++, p += 4) { const vox = V.toVox(u, v); vox[V.axis] = LB.k; shade(d, p, idx(vox[0], vox[1], vox[2]), sc); }
  }
  ctx.putImageData(img, 0, 0);
  const area = sliceAreaCM2(V.axis, LB.k);
  $('#lb-title').textContent = `${LB.view[0].toUpperCase() + LB.view.slice(1)} · slice ${LB.k + 1} / ${n}${area === null ? '' : ` · WT ${area.toFixed(1)} cm²`}`;
  ['l', 'r', 't', 'b'].forEach((s, i) => { $(`#lb-${s}`).textContent = V.labels[i]; });
}

function initLightbox() {
  const dlg = $('#lightbox');
  if (!dlg) return;
  dlg.querySelectorAll('.lb-nav').forEach((b) => b.addEventListener('click', () => drawLightbox(LB.k + Number(b.dataset.d))));
  $('#lb-close').addEventListener('click', () => dlg.close());
  $('#lb-goto').addEventListener('click', () => { dlg.close(); $('.views').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
  $('#lb-in').addEventListener('click', () => zoomLightbox(LB.z * 1.5));
  $('#lb-out').addEventListener('click', () => zoomLightbox(LB.z / 1.5));
  $('#lb-fit').addEventListener('click', () => zoomLightbox(1, true));
  const frame = dlg.querySelector('.lb-frame');
  frame.addEventListener('wheel', (ev) => { ev.preventDefault(); zoomLightbox(LB.z * Math.exp(-ev.deltaY * 0.0015)); }, { passive: false });
  let drag = null;
  frame.addEventListener('pointerdown', (ev) => {
    if (LB.z === 1) return;
    drag = [ev.clientX - LB.x, ev.clientY - LB.y]; frame.setPointerCapture(ev.pointerId);
  });
  frame.addEventListener('pointermove', (ev) => { if (drag) { LB.x = ev.clientX - drag[0]; LB.y = ev.clientY - drag[1]; zoomLightbox(LB.z); } });
  frame.addEventListener('pointerup', () => { drag = null; });
  frame.addEventListener('dblclick', () => zoomLightbox(LB.z > 1 ? 1 : 2.5, LB.z > 1));
  dlg.addEventListener('click', (ev) => { if (ev.target === dlg) dlg.close(); });        // click on the backdrop
  dlg.addEventListener('keydown', (ev) => {
    const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 10, PageDown: -10 }[ev.key];
    if (d) { ev.preventDefault(); drawLightbox(LB.k + d); }
    else if (ev.key === '+' || ev.key === '=') zoomLightbox(LB.z * 1.5);
    else if (ev.key === '-') zoomLightbox(LB.z / 1.5);
    else if (ev.key === '0') zoomLightbox(1, true);
  });
}

// Arrow keys / PageUp / PageDown / Home / End step through slices of the view under the pointer.
// F enlarges the view under the pointer (or restores the grid); Esc restores the grid.
document.addEventListener('keydown', (ev) => {
  if (ev.target.closest && ev.target.closest('input, select, textarea')) return;
  if ($('#lightbox')?.open || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if ((ev.key === 'f' || ev.key === 'F') && S.hoverPanel) {
    ev.preventDefault(); setFocus(S.hoverPanel.classList.contains('big') ? null : S.hoverPanel); return;
  }
  if (ev.key === 'Escape' && !document.fullscreenElement && $('.views').classList.contains('focus')) { setFocus(null); return; }
  const V = S.hover && VIEWS[S.hover];
  if (!V || !V.ctx) return;
  const n = [S.X, S.Y, S.Z][V.axis], k = S.cross[V.axis];
  const step = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 10, PageDown: -10 }[ev.key];
  const to = step ? k + step : ev.key === 'Home' ? 0 : ev.key === 'End' ? n - 1 : null;
  if (to === null) return;
  ev.preventDefault();
  setSlice(V, to);
});

// ------------------------------------------------------------------ 3D
const T = { meshes: {} };

function init3D() {
  const el = $('#view3d');
  T.renderer = new THREE.WebGLRenderer({ antialias: true });
  T.renderer.setPixelRatio(devicePixelRatio);
  el.append(T.renderer.domElement);
  T.scene = new THREE.Scene(); T.scene.background = new THREE.Color(0x0a0908);
  T.camera = new THREE.PerspectiveCamera(35, 1, 1, 5000); T.camera.up.set(0, 0, 1);
  T.scene.add(T.camera);
  T.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(0.3, 0.4, 1); T.camera.add(key);
  T.controls = new OrbitControls(T.camera, T.renderer.domElement);
  T.controls.enableDamping = true;

  const mat = new THREE.MeshBasicMaterial({ color: 0xf2ede3, depthTest: false });
  T.dot = new THREE.Mesh(new THREE.SphereGeometry(2.2, 16, 12), mat); T.dot.renderOrder = 10; T.scene.add(T.dot);
  T.lines = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xf2ede3, transparent: true, opacity: 0.7, depthTest: false }));
  T.lines.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(18), 3));
  T.lines.renderOrder = 10; T.scene.add(T.lines);

  new ResizeObserver(() => {
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) return;
    T.renderer.setSize(r.width, r.height, false); T.camera.aspect = r.width / r.height; T.camera.updateProjectionMatrix();
  }).observe(el);

  // Click (not drag) on a visible surface sets the crosshair
  let down = null;
  T.renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  T.renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = T.renderer.domElement.getBoundingClientRect();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), T.camera);
    const targets = Object.values(T.meshes).filter((m) => m.visible);
    const hit = ray.intersectObjects(targets, false)[0];
    if (hit) setCross(mv(S.Ainv, [hit.point.x, hit.point.y, hit.point.z, 1]));
  });

  (function loop() { requestAnimationFrame(loop); T.controls.update(); T.renderer.render(T.scene, T.camera); })();
}

function b64(s, Type) { const bin = atob(s), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Type(u8.buffer); }

function textSprite(text) {
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d'); g.font = '500 40px "IBM Plex Mono", monospace'; g.fillStyle = '#f2ede3'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, 32, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.set(14, 14, 1); s.renderOrder = 11; return s;
}

function buildMeshes(data) {
  for (const m of Object.values(T.meshes)) { T.scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
  T.meshes = {};
  for (const [key, d] of Object.entries(data)) {
    if (!d.vertices) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(b64(d.vertices, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(b64(d.faces, Uint32Array), 1));
    g.computeVertexNormals();
    const isBrain = key === 'brain';
    const mat = new THREE.MeshPhongMaterial({ color: d.color, transparent: true, opacity: d.opacity, side: THREE.DoubleSide,
      depthWrite: !isBrain && d.opacity > 0.6, shininess: 25 });
    const mesh = new THREE.Mesh(g, mat); mesh.renderOrder = isBrain ? 5 : 1;
    mesh.visible = d.source !== 'expert';
    T.scene.add(mesh); T.meshes[key] = mesh;
  }
  buildLayerControls(data);
  if (!T.framed) frameCamera();
}

function frameCamera() {
  const box = new THREE.Box3();
  const ref = T.meshes.brain || Object.values(T.meshes)[0];
  if (ref) box.setFromObject(ref); else box.setFromPoints([new THREE.Vector3(...toWorld([0, 0, 0])), new THREE.Vector3(...toWorld([S.X, S.Y, S.Z]))]);
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()), r = size.length() / 2;
  T.center = c; T.radius = r;
  T.controls.target.copy(c);
  T.camera.position.set(c.x - r * 0.9, c.y + r * 2.2, c.z + r * 0.9); // anterior-left-superior view
  T.camera.near = r / 50; T.camera.far = r * 20; T.camera.updateProjectionMatrix();
  if (!T.orient) {
    T.orient = new THREE.Group();
    const e = size.clone().multiplyScalar(0.5).addScalar(14);
    [['R', [e.x, 0, 0]], ['L', [-e.x, 0, 0]], ['A', [0, e.y, 0]], ['P', [0, -e.y, 0]], ['S', [0, 0, e.z]], ['I', [0, 0, -e.z]]]
      .forEach(([t, p]) => { const s = textSprite(t); s.position.set(c.x + p[0], c.y + p[1], c.z + p[2]); T.orient.add(s); });
    T.scene.add(T.orient);
  }
  T.framed = true;
}

function update3DCross() {
  if (!T.dot) return;
  const w = toWorld(S.cross); T.dot.position.set(...w);
  const [x, y, z] = S.cross, pts = [
    [0, y, z], [S.X - 1, y, z], [x, 0, z], [x, S.Y - 1, z], [x, y, 0], [x, y, S.Z - 1]].map(toWorld).flat();
  T.lines.geometry.attributes.position.array.set(pts); T.lines.geometry.attributes.position.needsUpdate = true;
  T.lines.geometry.computeBoundingSphere();
}

function buildLayerControls(data) {
  const box = $('#layers'); box.innerHTML = '';
  for (const [key, mesh] of Object.entries(T.meshes)) {
    const d = data[key], row = document.createElement('div'); row.className = 'layer';
    row.innerHTML = `<input type="checkbox" ${mesh.visible ? 'checked' : ''} aria-label="Show ${d.label}">
      <span><i style="background:${d.color}"></i> ${d.label}</span>
      <input type="range" min="0" max="1" step="0.05" value="${mesh.material.opacity}" aria-label="Opacity of ${d.label}">
      <a class="stl" href="${API}/stl?layers=${encodeURIComponent(key)}" download title="Download ${d.label} as STL for 3D printing">STL</a>`;
    const [cb, , range] = row.children;
    cb.addEventListener('change', () => { mesh.visible = cb.checked; });
    range.addEventListener('input', () => {
      mesh.material.opacity = Number(range.value);
      mesh.material.depthWrite = key !== 'brain' && mesh.material.opacity > 0.6; mesh.material.needsUpdate = true;
    });
    box.append(row);
  }
  if (!Object.keys(T.meshes).length) box.textContent = 'No surfaces available.';
  $('#stl-visible').hidden = !Object.keys(T.meshes).length;
}

// One binary STL of the layers ticked in the 3D view, built on the server from the same surfaces.
async function downloadVisibleSTL(btn) {
  const keys = Object.entries(T.meshes).filter(([, m]) => m.visible).map(([k]) => k);
  if (!keys.length) { status('Tick at least one 3D layer to export as STL.', true); return; }
  btn.disabled = true;
  try {
    const r = await fetch(`${API}/stl?layers=${keys.map(encodeURIComponent).join(',')}`);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).message || `${r.status} ${r.statusText}`);
    const name = /filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') || '')?.[1] || `voxels_${studyId}.stl`;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(await r.blob()); a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    status(`Saved ${name} (${keys.length} layer${keys.length > 1 ? 's' : ''}, millimetres).`);
  } catch (err) { status(`STL export failed: ${err.message}`, true); }
  btn.disabled = false;
}

// ------------------------------------------------------------------ tables
function renderTables() {
  const m = S.meta, sp = m.spacing_mm, vox = m.voxel_volume_mm3, mvols = m.volumes.model, ev = m.volumes.expert;
  const wt = mvols ? mvols.WT : null;
  $('#formula').innerHTML =
    `V = N × (Δx × Δy × Δz) = N × (${sp.map((s) => s.toFixed(3)).join(' × ')}) mm³ = N × ${vox.toFixed(3)} mm³ ; ` +
    `mL = mm³ ÷ 1000 (1 mL = 1,000 mm³)` +
    (wt ? `<br>Example, WT: ${wt.voxels.toLocaleString()} × ${vox.toFixed(3)} mm³ = ${wt.mm3.toLocaleString(undefined, { maximumFractionDigits: 1 })} mm³ = ${wt.mL.toFixed(2)} mL` : '');
  const head = ['Region', 'BraTS codes', 'Model N', 'Model mm³', 'Model mL (est.)']
    .concat(ev ? ['Expert N', 'Expert mm³', 'Expert mL', 'Model − expert mL'] : []);
  let html = `<tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr>`;
  for (const r of ['NCR', 'ED', 'ET', 'TC', 'WT']) {
    const a = mvols ? mvols[r] : null, b = ev ? ev[r] : null;
    html += `<tr><td>${r}</td><td>${REGION_CODES[r].join(', ')}</td><td>${a ? a.voxels.toLocaleString() : '—'}</td>` +
      `<td>${a ? a.mm3.toLocaleString(undefined, { maximumFractionDigits: 1 }) : '—'}</td><td><b>${a ? a.mL.toFixed(2) : '—'}</b></td>` +
      (ev ? `<td>${b.voxels.toLocaleString()}</td><td>${b.mm3.toLocaleString(undefined, { maximumFractionDigits: 1 })}</td>` +
        `<td><b>${b.mL.toFixed(2)}</b></td><td>${(a.mL - b.mL).toFixed(2)}</td>` : '') + '</tr>';
  }
  $('#vol-table').innerHTML = html;
  $('#agreement').innerHTML = '';                      // agreement now lives in the expert comparison section
  renderExpert(m);
}

// ------------------------------------------------------------------ expert comparison (this case only)
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const sgn = (v, d) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);

function renderExpert(m) {
  const out = [], ag = m.agreement, rc = m.readings_comparison;
  if (ag) {
    const fn = m.expert_mask?.filename;
    out.push(`<h3>Expert mask${fn ? ` · ${esc(fn)}` : ''}</h3><div class="tablewrap"><table class="grid">
      <tr><th>Region</th><th>Dice</th><th>Sensitivity</th><th>Precision</th><th>IoU</th><th>HD95 (mm)</th><th>Model voxels</th><th>Expert voxels</th></tr>
      ${['WT', 'TC', 'ET'].map((r) => { const v = ag[r];
        return `<tr><td>${r}</td><td><b>${fmt(v.dice, 3)}</b></td><td>${fmt(v.sensitivity, 3)}</td><td>${fmt(v.precision, 3)}</td>
          <td>${fmt(v.iou, 3)}</td><td>${fmt(v.hd95_mm, 1)}</td><td>${v.pred_voxels.toLocaleString()}</td><td>${v.ref_voxels.toLocaleString()}</td></tr>`; }).join('')}
      </table></div>
      ${(m.expert_notes || []).map((n) => `<p class="xnote">${esc(n)}</p>`).join('')}
      <p class="xdef">Dice = 2 × overlap ÷ (model + expert voxels) · sensitivity = overlap ÷ expert voxels · precision = overlap ÷ model voxels · IoU = overlap ÷ union · HD95 = 95th-percentile surface distance (mm). WT = 1+2+4, TC = 1+4, ET = 4. — = undefined (empty region).</p>`);
  } else if (m.has_expert && !m.has_prediction) {
    out.push('<p class="xempty">Expert mask attached; run segmentation to compare it with the model.</p>');
  }
  if (rc) {
    out.push(`<h3>Doctor's volume readings · entered ${esc(m.doctor_readings.entered_utc.replace('T', ' ').slice(0, 16))} UTC</h3>
      <div class="tablewrap"><table class="grid">
      <tr><th>Region</th><th>Model (mL)</th><th>Entered (mL)</th><th>Model − entered (mL)</th><th>Absolute (mL)</th><th>Difference (%)</th></tr>
      ${Object.entries(rc).map(([r, v]) => `<tr><td>${r}</td><td>${v.model_mL.toFixed(2)}</td><td>${v.doctor_mL.toFixed(2)}</td>
        <td><b>${sgn(v.difference_mL, 2)}</b></td><td>${v.abs_difference_mL.toFixed(2)}</td>
        <td>${v.percent_difference === null ? '— (entered 0)' : sgn(v.percent_difference, 1) + '%'}</td></tr>`).join('')}
      </table></div>
      <p class="xdef">Difference = model − entered; % = difference ÷ entered. Volumes alone cannot give Dice, sensitivity or any
        overlap score: equal volumes do not mean the same voxels. Upload an expert mask for those.</p>`);
  }
  if (!out.length) {
    out.push(`<p class="xempty">No expert reference for this study yet. Use <b>Expert comparison</b> in the side panel to
      upload an expert-drawn mask (overlap scores) or enter a doctor's WT / TC / ET volumes (volume differences).</p>`);
  }
  $('#expert-results').innerHTML = out.join('');
  $('#expert-remove').hidden = !m.has_expert;
  $('#readings-clear').hidden = !m.doctor_readings;
  const f = $('#readings-form');
  if (m.doctor_readings && !f.dataset.touched) {
    for (const r of ['WT', 'TC', 'ET']) f.elements[r].value = m.doctor_readings.volumes_mL[r] ?? '';
  }
}

function setMsg(el, text, ok) { el.textContent = text; el.className = `small ${ok ? 'msg-ok' : 'msg-err'}`; }

async function refreshExpert() {
  applyMeta(await getJSON(`${API}/meta`));
}

function renderTrust(m) {
  const u = m.uncertainty, parts = [];
  if (u) {
    const row = (r) => { const x = u.regions[r]; const dmin = Math.min(...x.dice_vs_main_per_pass);
      return `<tr><td>${r}</td><td>${x.volume_mL_min.toFixed(1)}–${x.volume_mL_max.toFixed(1)}</td><td>${dmin.toFixed(3)}</td></tr>`; };
    parts.push(`<p>Test-time augmentation: the model was re-run on the scan mirrored along each axis (${u.passes} passes).
      <b>${(u.wt_uncertain_fraction * 100).toFixed(1)}%</b> of WT voxels are disputed between passes.</p>
      <table class="kv"><tr><td>Region</td><td>mL range</td><td>min Dice</td></tr>${['WT', 'TC', 'ET'].map(row).join('')}</table>
      <p class="muted">Choose “Uncertainty (TTA, WT)” under Slices to see where.</p>`);
  } else {
    parts.push('<p class="muted">No test-time-augmentation result for this study.</p>');
  }
  if (m.agreement) parts.push(`<p>Against the supplied expert mask: WT Dice <b>${m.agreement.WT.dice.toFixed(3)}</b>.</p>`);
  $('#trust').innerHTML = parts.join('');
  renderAblation(m.ablation);
}

function renderAblation(a) {
  if (!a) { $('#ablation').innerHTML = ''; return; }
  $('#run-ablation').hidden = true;
  const rows = Object.entries(a.removed_sequence).map(([s, v]) =>
    `<tr><td>without ${s.toUpperCase()}</td>${['WT', 'TC', 'ET'].map((r) => `<td>${v[r].dice_vs_full.toFixed(2)}</td>`).join('')}</tr>`).join('');
  $('#ablation').innerHTML = `<p style="margin-top:10px"><b>${a.label}</b></p>
    <table class="kv"><tr><td>Dice vs full input</td><td>WT</td><td>TC</td><td>ET</td></tr>${rows}</table>
    <p class="muted">${a.method}</p>`;
}

function applyMeta(m) {
  S.meta = m;
  [S.X, S.Y, S.Z] = m.canonical_shape; S.sp = m.canonical_spacing_mm;
  S.A = m.canonical_affine; S.Ainv = inv4(S.A); S.Oinv = inv4(m.original_affine);
  $('#backup').hidden = !m.backup;
  if (m.backup) {
    $('#backup').textContent = `PRECOMPUTED BACKUP CASE — ${m.backup.case}, generated ${m.backup.generated_utc} by this ` +
      'pipeline. This is NOT a live run on an uploaded case.';
  }
  const w = m.suitability?.warnings || [];
  $('#warnings').hidden = !w.length;
  $('#warnings').innerHTML = w.length ? `<b>Input suitability warnings</b><ul>${w.map((x) => `<li>${x}</li>`).join('')}</ul>` : '';
  for (const o of $('#overlay').options) {
    if (o.value === 'uncert') o.disabled = !m.has_uncertainty;
    else if (o.value !== 'pred' && o.value !== 'none') o.disabled = !m.has_expert;
  }
  renderTrust(m);
  $('#c-expert-row').hidden = !m.has_expert;
  $('#disclaimer').textContent = m.disclaimer;
  renderTables();
}

// ------------------------------------------------------------------ boot
async function main() {
  $('#study').textContent = studyId;
  $('#lnk-report').href = `/report/${studyId}`; $('#lnk-pdf').href = `/api/report/${studyId}.pdf`;
  $('#lnk-nifti').href = `/api/segmentation/prediction/${studyId}`;
  try {
    applyMeta(await getJSON(`${API}/meta`));
    if (!S.meta.has_prediction) status('No prediction yet — run segmentation from the upload page first.', true);
    status('Loading volumes…');
    await Promise.all([getVol('flair'), S.meta.has_prediction ? getVol('pred') : null, S.meta.has_expert ? getVol('expert') : null]);
    computeProfiles();
    buildSlicePanels();
    init3D();
    buildPanelTools();
    initLightbox();
    setCross(S.meta.lesion_centroid_canonical || [S.X >> 1, S.Y >> 1, S.Z >> 1]);
    buildGallery();
    status('Building surfaces…');
    buildMeshes(await getJSON(`${API}/meshes`));
    update3DCross();
    status('Ready. Click or drag in a slice; change slice with the slider, ‹ ›, ▶, the mouse wheel or ↑/↓ over a view; '
      + 'or click a 3D surface. Double-click a view (or its ↗ button) to enlarge it; click any slice under “All slices” to open it large.');
  } catch (e) { status(`Error: ${e.message}`, true); console.error(e); }
}

$('#bg-seq').addEventListener('change', async (e) => {
  try { status('Loading sequence…'); await getVol(e.target.value); S.bg = e.target.value; setCross(S.cross); refreshGallery(); status('Ready.'); }
  catch (err) { status(err.message, true); }
});
$('#overlay').addEventListener('change', async (e) => {
  if (e.target.value === 'uncert') {
    try { await getVol('uncert'); } catch (err) { status(err.message, true); return; }
  }
  S.overlay = e.target.value; $('#agree-legend').hidden = !S.overlay.startsWith('agree-');
  $('#uncert-legend').hidden = S.overlay !== 'uncert';
  $('#label-toggles').style.opacity = S.overlay.startsWith('agree-') ? 0.4 : 1; renderAllSlices(); refreshGallery();
});
$('#ov-alpha').addEventListener('input', (e) => { S.alpha = Number(e.target.value); renderAllSlices(); refreshGallery(); });
document.querySelectorAll('#label-toggles input').forEach((cb) => cb.addEventListener('change', () => {
  const c = Number(cb.dataset.code); cb.checked ? S.codes.add(c) : S.codes.delete(c); renderAllSlices(); refreshGallery();
}));
$('#goto-lesion').addEventListener('click', () => S.meta?.lesion_centroid_canonical && setCross(S.meta.lesion_centroid_canonical));
$('#reset-cam').addEventListener('click', () => { T.framed = false; frameCamera(); });
$('#stl-visible').addEventListener('click', (e) => downloadVisibleSTL(e.currentTarget));
$('#expert-file').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const msg = $('#expert-msg'); msg.textContent = 'Uploading and validating…';
  try {
    const fd = new FormData(); fd.append('file', f);
    await getJSON(`${API}/expert`, { method: 'POST', body: fd });
    delete S.vols.expert;
    applyMeta(await getJSON(`${API}/meta`));
    await getVol('expert');
    buildMeshes(await getJSON(`${API}/meshes`));
    setCross(S.cross);
    refreshGallery();
    msg.textContent = 'Expert mask attached. Scores are under “Expert comparison” below the viewer; choose "Expert mask" '
      + 'or "Agreement" under Slices to see it, and expert surfaces are in 3D layers.';
  } catch (err) { msg.textContent = `Rejected: ${err.message}`; }
  e.target.value = '';
});
$('#expert-remove').addEventListener('click', async (e) => {
  e.target.disabled = true;
  try {
    await getJSON(`${API}/expert`, { method: 'DELETE' });
    delete S.vols.expert;
    if (S.overlay === 'expert' || S.overlay.startsWith('agree-')) { S.overlay = 'pred'; $('#overlay').value = 'pred'; }
    await refreshExpert();
    buildMeshes(await getJSON(`${API}/meshes`));
    setCross(S.cross); refreshGallery();
    setMsg($('#expert-msg'), 'Expert mask removed.', true);
  } catch (err) { setMsg($('#expert-msg'), err.message, false); }
  e.target.disabled = false;
});
$('#readings-form').addEventListener('input', (e) => { e.currentTarget.dataset.touched = '1'; });
$('#readings-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.currentTarget, body = {}, msg = $('#readings-msg');
  for (const r of ['WT', 'TC', 'ET']) {
    const raw = f.elements[r].value.trim();
    if (raw === '') continue;
    const v = Number(raw);
    if (!Number.isFinite(v)) { setMsg(msg, `${r} must be a number of mL.`, false); return; }
    body[r] = v;
  }
  if (!Object.keys(body).length) { setMsg(msg, 'Enter at least one of WT, TC or ET in mL.', false); return; }
  try {
    await getJSON(`${API}/readings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    delete f.dataset.touched;
    await refreshExpert();
    setMsg(msg, 'Saved. See “Expert comparison” below the viewer.', true);
    $('#expert-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) { setMsg(msg, err.message, false); }
});
$('#readings-clear').addEventListener('click', async () => {
  try {
    await getJSON(`${API}/readings`, { method: 'DELETE' });
    const f = $('#readings-form'); f.reset(); delete f.dataset.touched;
    await refreshExpert();
    setMsg($('#readings-msg'), 'Readings removed.', true);
  } catch (err) { setMsg($('#readings-msg'), err.message, false); }
});
$('#run-ablation').addEventListener('click', async (e) => {
  e.target.disabled = true; e.target.textContent = 'Running 5 passes…';
  try { renderAblation(await getJSON(`${API}/ablation`, { method: 'POST' })); }
  catch (err) { $('#ablation').textContent = err.message; e.target.disabled = false; e.target.textContent = 'Run modality ablation'; }
});
window.addEventListener('resize', layoutSlices);
$('#g-axis')?.addEventListener('change', buildGallery);
$('#g-filter')?.addEventListener('change', buildGallery);
$('#g-size')?.addEventListener('change', buildGallery);
main();
