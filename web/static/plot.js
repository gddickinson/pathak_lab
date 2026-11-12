// Small plotting helpers: SVG scatter and line charts, a canvas track map, number formatting and a sortable table.
import { esc } from './core.js';

export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? '–'
  : Number.isInteger(v) ? String(v) : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v.toFixed(d)).toString();

const W = 420, H = 280, M = { l: 50, r: 12, t: 12, b: 40 };
function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(10));
  return out;
}
function extent(vals) {
  const f = vals.filter(Number.isFinite);
  if (!f.length) return [0, 1];
  let lo = Math.min(...f), hi = Math.max(...f);
  if (hi === lo) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}
function frame(xd, yd, xlabel, ylabel) {
  const sx = v => M.l + (v - xd[0]) / (xd[1] - xd[0]) * (W - M.l - M.r);
  const sy = v => H - M.b - (v - yd[0]) / (yd[1] - yd[0]) * (H - M.t - M.b);
  let s = '';
  for (const v of ticks(...xd)) s += `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${M.t}" y2="${H - M.b}"/><text x="${sx(v)}" y="${H - M.b + 14}" text-anchor="middle">${fmt(v, 2)}</text>`;
  for (const v of ticks(...yd, 4)) s += `<line class="grid" x1="${M.l}" x2="${W - M.r}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${M.l - 5}" y="${sy(v) + 4}" text-anchor="end">${fmt(v, 2)}</text>`;
  s += `<line class="axis" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/><line class="axis" x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}"/>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(xlabel)}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
  return { sx, sy, s };
}

// points: [{x, y, color, title}]
export function scatter(points, { xlabel = '', ylabel = '', r = 3.5 } = {}) {
  const ok = points.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!ok.length) return '<p class="mute small">No finite values to plot.</p>';
  const { sx, sy, s } = frame(extent(ok.map(p => p.x)), extent(ok.map(p => p.y)), xlabel, ylabel);
  const dots = ok.map(p => `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${r}" fill="${p.color}" fill-opacity=".8"><title>${esc(p.title || '')}</title></circle>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} against ${esc(xlabel)}">${s}${dots}</svg>`;
}

// series: [{pts: [[x, y], ...], color, title}]
export function lines(series, { xlabel = '', ylabel = '' } = {}) {
  const all = series.flatMap(se => se.pts.filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])));
  if (!all.length) return '<p class="mute small">No finite values to plot.</p>';
  const { sx, sy, s } = frame(extent(all.map(p => p[0])), extent(all.map(p => p[1])), xlabel, ylabel);
  const paths = series.map(se => {
    const d = se.pts.filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])).map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join('');
    return d ? `<path d="${d}" fill="none" stroke="${se.color}" stroke-opacity=".75" stroke-width="1.3"><title>${esc(se.title || '')}</title></path>` : '';
  }).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} against ${esc(xlabel)}">${s}${paths}</svg>`;
}

// Track map on a canvas: dots for localisations, polylines for tracks. Coordinates in pixels of a size x size field.
export function drawTracks(canvas, { size, dots = [], tracks = [], highlight = null }) {
  const px = 600, k = px / size;
  canvas.width = px; canvas.height = px;
  const g = canvas.getContext('2d');
  g.fillStyle = '#020308'; g.fillRect(0, 0, px, px);
  for (const d of dots) { g.fillStyle = d.color; g.globalAlpha = 0.55; g.fillRect(d.x * k - 1.2, d.y * k - 1.2, 2.4, 2.4); }
  g.globalAlpha = 1; g.lineWidth = 1.4;
  for (const t of tracks) {
    if (t.pts.length < 2) continue;
    g.strokeStyle = t.color; g.globalAlpha = highlight === null || highlight === t.id ? 0.9 : 0.25;
    g.beginPath(); t.pts.forEach(([x, y], i) => (i ? g.lineTo(x * k, y * k) : g.moveTo(x * k, y * k))); g.stroke();
  }
  g.globalAlpha = 1;
  // scale bar: 5 pixels
  g.fillStyle = '#e8f1ff'; g.fillRect(px - 20 - 5 * k, px - 18, 5 * k, 3);
  g.font = '12px ui-monospace, monospace'; g.fillText('5 px', px - 20 - 5 * k, px - 24);
}

// Sortable table. cols: [{key, label, fmt?}]; rows: objects. onRow(row) on click / Enter.
export function table(host, cols, rows, { onRow = null, limit = 400 } = {}) {
  let key = null, dir = 1;
  const draw = () => {
    const rs = key ? [...rows].sort((a, b) => ((a[key] ?? -Infinity) > (b[key] ?? -Infinity) ? dir : -dir)) : rows;
    host.innerHTML = `<div class="scroll"><table class="tbl"><thead><tr>${cols.map(c => `<th><button data-k="${c.key}">${esc(c.label)}${key === c.key ? (dir > 0 ? ' ▲' : ' ▼') : ''}</button></th>`).join('')}</tr></thead><tbody>${
      rs.slice(0, limit).map((r, i) => `<tr${onRow ? ` class="click" tabindex="0" data-i="${rows.indexOf(r)}"` : ''}>${cols.map(c => `<td>${c.fmt ? c.fmt(r[c.key], r) : esc(fmt(r[c.key]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${
      rs.length > limit ? `<p class="small mute">First ${limit} of ${rs.length} rows.</p>` : ''}`;
    host.querySelectorAll('th button').forEach(b => b.onclick = () => { dir = key === b.dataset.k ? -dir : 1; key = b.dataset.k; draw(); });
    if (onRow) host.querySelectorAll('tr.click').forEach(tr => {
      const go = () => onRow(rows[+tr.dataset.i]);
      tr.onclick = go; tr.onkeydown = e => { if (e.key === 'Enter') go(); };
    });
  };
  draw();
}

export const legend = items => `<div class="legend">${items.map(([label, color]) => `<span style="--c:${color}">${esc(label)}</span>`).join('')}</div>`;
