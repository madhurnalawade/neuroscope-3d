// Loading illustration for the analyse page. It starts on a glass brain (vessels and ventricles); as the run
// progresses a solid brain forms over it from the bottom up, behind a glowing scan line. Both pictures are
// reference illustrations (static/loader/), aligned offline; neither is drawn from any scan.
const FW = 720, FH = 640;                              // frame both images share
const Y_START = 590, Y_END = 70;                      // scan line: just below the solid brain to just above it

const load = (src) => new Promise((resolve) => { const i = new Image(); i.onload = () => resolve(i); i.onerror = () => resolve(null); i.src = src; });
const layer = () => { const c = document.createElement('canvas'); c.width = FW; c.height = FH; return [c, c.getContext('2d')]; };
const smooth = (a, b, x) => { const u = Math.min(1, Math.max(0, (x - a) / (b - a))); return u * u * (3 - 2 * u); };

export function createBrain(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const [fin, finCtx] = layer(), [ini, iniCtx] = layer(), [band, bandCtx] = layer(), [frame, frameCtx] = layer();
  const [uni, uniCtx] = layer();                       // union of both silhouettes
  let imgInit = null, imgFinal = null, rows = null, t0 = null, doneAt = null, lastT = null;
  const sparks = [];

  Promise.all([load('/static/loader/brain_initial.webp'), load('/static/loader/brain_final.webp')]).then(([a, b]) => {
    if (!a || !b) return;
    uniCtx.drawImage(a, 0, 0); uniCtx.drawImage(b, 0, 0);
    // per-row horizontal extent of the brains, so sparks start on the brain and not in empty space
    const px = uniCtx.getImageData(0, 0, FW, FH).data;
    rows = Array.from({ length: FH }, (_, y) => {
      let lo = -1, hi = -1;
      for (let x = 0; x < FW; x++) if (px[(y * FW + x) * 4 + 3] > 80) { if (lo < 0) lo = x; hi = x; }
      return lo < 0 ? null : [lo, hi];
    });
    imgInit = a; imgFinal = b;
  });

  function fit() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
  }

  // vertical alpha mask: transparent above row a and opaque below row b (the reverse when `invert`)
  function mask(c, a, b, invert) {
    const g = c.createLinearGradient(0, a, 0, b);
    g.addColorStop(0, invert ? 'rgba(0,0,0,1)' : 'rgba(0,0,0,0)');
    g.addColorStop(1, invert ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,1)');
    c.globalCompositeOperation = 'destination-in'; c.fillStyle = g; c.fillRect(0, 0, FW, FH);
    c.globalCompositeOperation = 'source-over';
  }

  function drawSparks(c, y, dt, emit) {
    const span = rows && rows[Math.round(Math.min(FH - 1, Math.max(0, y)))];
    if (emit && span && !still) for (let n = 0; n < 2; n++) {
      sparks.push({ x: span[0] + Math.random() * (span[1] - span[0]), y, vy: -(18 + Math.random() * 40), life: 1, r: 0.8 + Math.random() * 1.8 });
    }
    c.globalCompositeOperation = 'lighter';
    for (let i = sparks.length - 1; i >= 0; i--) {
      const s = sparks[i];
      s.y += s.vy * dt; s.life -= dt * 1.4;
      if (s.life <= 0) { sparks.splice(i, 1); continue; }
      c.fillStyle = `rgba(255, 196, 150, ${0.75 * s.life})`;
      c.beginPath(); c.arc(s.x, s.y, s.r, 0, Math.PI * 2); c.fill();
    }
    c.globalCompositeOperation = 'source-over';
  }

  // p in [0, 1]: how far the run has got; t: time in ms
  function draw(p, t) {
    fit();
    const dt = lastT === null ? 0 : Math.min(0.1, (t - lastT) / 1000); lastT = t;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!imgInit) return;
    if (t0 === null) t0 = t;

    const intro = Math.min(1, (t - t0) / 700);                       // the glass brain fades in first
    const y = Y_START + (Y_END - Y_START) * Math.min(1, p);          // scan line, in image rows
    const done = p >= 0.999;
    if (done && doneAt === null) doneAt = t;
    if (!done) doneAt = null;

    // glass brain above the line; it lingers a little below it so the vessels melt into tissue
    iniCtx.clearRect(0, 0, FW, FH);
    iniCtx.globalAlpha = intro * (1 - smooth(0.75, 1, p)) * (still ? 1 : 0.88 + 0.12 * Math.sin(t / 420));   // gone by the end
    iniCtx.drawImage(imgInit, 0, 0); iniCtx.globalAlpha = 1;
    if (p > 0) mask(iniCtx, y - 12, y + 90, true);

    // solid brain below the line
    finCtx.clearRect(0, 0, FW, FH);
    if (p > 0) { finCtx.drawImage(imgFinal, 0, 0); if (!done) mask(finCtx, y - 30, y + 8, false); }

    frameCtx.clearRect(0, 0, FW, FH);
    frameCtx.drawImage(ini, 0, 0);
    frameCtx.drawImage(fin, 0, 0);

    if (p > 0 && !done) {                                            // glowing scan band, only where there is brain
      bandCtx.clearRect(0, 0, FW, FH);
      const g = bandCtx.createLinearGradient(0, y - 46, 0, y + 46);
      g.addColorStop(0, 'rgba(232,120,80,0)'); g.addColorStop(0.47, 'rgba(255,170,120,0.55)');
      g.addColorStop(0.5, 'rgba(255,236,214,1)'); g.addColorStop(0.53, 'rgba(255,170,120,0.55)'); g.addColorStop(1, 'rgba(232,120,80,0)');
      bandCtx.fillStyle = g; bandCtx.fillRect(0, y - 46, FW, 92);
      bandCtx.globalCompositeOperation = 'destination-in'; bandCtx.drawImage(uni, 0, 0);
      bandCtx.globalCompositeOperation = 'source-over';
      frameCtx.globalCompositeOperation = 'lighter';
      frameCtx.globalAlpha = still ? 0.8 : 0.7 + 0.3 * Math.sin(t / 160) ** 2;
      frameCtx.drawImage(band, 0, 0);
      frameCtx.globalAlpha = 1; frameCtx.globalCompositeOperation = 'source-over';
    }
    drawSparks(frameCtx, y, dt, p > 0 && !done);

    if (done && !still) {                                            // a soft flash when the brain is complete
      const k = Math.max(0, 1 - (t - doneAt) / 900);
      if (k > 0) {
        frameCtx.globalCompositeOperation = 'lighter'; frameCtx.globalAlpha = 0.35 * k; frameCtx.drawImage(imgFinal, 0, 0);
        frameCtx.globalAlpha = 1; frameCtx.globalCompositeOperation = 'source-over';
      }
    }

    const s = Math.min(canvas.width / FW, canvas.height / FH), w = FW * s, h = FH * s;   // contain the frame
    ctx.drawImage(frame, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  }

  return { draw };
}
