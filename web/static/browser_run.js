// The Run page on the static site: the steps that have JS ports (link, features, Steps 5, 6, 8, 11) run in the
// browser on the uploaded CSV and return the same summary shape as the Python server (web/analysis.py).
import { linkPoints, locsToTxy, tracksTable } from './link_engine.js';
import { featuresTable } from './features_engine.js';
import { addDiffusion, addLocalizationError, addNNCounts, addVelocity } from './mobility_engine.js';

export const BROWSER_STEPS = ['link', 'features', 'diffusion', 'velocity', 'nncount', 'locerr'];
const TRACK_COLS = ['n_segments', 'radius_gyration', 'asymmetry', 'skewness', 'kurtosis', 'fracDimension', 'netDispl',
  'Straight', 'track_length', 'meanLag', 'meanVelocity', 'meanLocDistanceFromCenter', 'track_intensity_mean'];

export function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += ch; continue; }
    if (ch === '"') q = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') { if (ch === '\r' && text[i + 1] === '\n') i++; row.push(cell); cell = ''; if (row.some(c => c !== '')) rows.push(row); row = []; }
    else cell += ch;
  }
  row.push(cell); if (row.some(c => c !== '')) rows.push(row);
  const head = rows.shift() || [];
  return rows.map(r => Object.fromEntries(head.map((h, i) => {
    const v = r[i];
    return [h, v === undefined || v === '' ? NaN : Number.isFinite(+v) ? +v : v];
  })));
}

export function toCsv(rows) {
  const cols = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const cell = v => (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
  return [cols.join(','), ...rows.map(r => cols.map(c => cell(r[c])).join(','))].join('\n');
}

export function runInBrowser(table, p, steps, name) {
  const cols = new Set(Object.keys(table[0] || {}));
  const kind = ['track_number', 'frame', 'x', 'y'].every(c => cols.has(c)) ? 'tracks'
    : ['frame', 'x [nm]', 'y [nm]'].every(c => cols.has(c)) ? 'locs' : null;
  if (!kind) throw new Error('Expected a ThunderSTORM localisation file (frame, x [nm], y [nm]) or a tracks table (track_number, frame, x, y).');
  if (kind === 'locs' && !steps.includes('link')) throw new Error("A localisation file must be linked first: tick 'Link' (Step 2).");
  if (kind === 'tracks' && steps.includes('link')) throw new Error("This file is already linked (it has track_number): untick 'Link'.");
  for (const [s, req] of [['diffusion', 'features'], ['velocity', 'diffusion']]) if (steps.includes(s) && !steps.includes(req)) throw new Error(`Step '${s}' needs '${req}' too.`);
  const warnings = ['Run in your browser with the JS ports of the Python functions (tested against them).'];
  let rows = table, nLocs = null;
  if (kind === 'locs') {
    nLocs = table.length;
    const locs = table.map((r, i) => ({ frame: r.frame, id: Number.isFinite(r.id) ? r.id : i + 1, xnm: r['x [nm]'], ynm: r['y [nm]'] }));
    if (!cols.has('id')) warnings.push("No 'id' column: localisations were numbered 1..n in file order.");
    const txy = locsToTxy(locs, p.pixel_size_nm);
    const L = linkPoints(txy, p.skip_frames, p.link_distance);
    if (L.recursiveFailure) warnings.push('Recursion limit reached while linking (Step 2 would skip this file).');
    warnings.push('No recording given: intensity columns are empty (Step 2 reads them from the .tif).');
    rows = tracksTable(txy, L, locs.map(l => l.id), null);
    if (!rows.length) throw new Error('No localisations could be linked with these settings.');
  }
  const nLinked = new Set(rows.map(r => r.track_number)).size;
  if (steps.includes('features')) {
    rows = rows.map((r, i) => ({ ...r, id: Number.isFinite(r.id) ? r.id : i, intensity: Number.isFinite(r.intensity) ? r.intensity : null }));
    try { rows = featuresTable(rows, p.min_segments); } catch (e) { throw new Error(`Track features failed (${e.message}). Step 2 skips the whole file in this case.`); }
    if (!rows.length) throw new Error('No track is longer than the minimum number of segments.');
  }
  if (steps.includes('diffusion')) rows = addDiffusion(rows);
  if (steps.includes('velocity')) rows = addVelocity(rows);
  if (steps.includes('nncount')) rows = addNNCounts(rows, p.radii);
  if (steps.includes('locerr')) rows = addLocalizationError(rows);
  return summarise(rows, { kind, params: p, steps, warnings, name, nLinked, nLocs });
}

function summarise(rows, { kind, params, steps, warnings, name, nLinked, nLocs }) {
  const by = new Map();
  for (const r of rows) if (Number.isFinite(r.track_number)) { if (!by.has(r.track_number)) by.set(r.track_number, []); by.get(r.track_number).push(r); }
  const cols = TRACK_COLS.filter(c => rows.some(r => c in r));
  const num = v => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const ids = [...by.keys()].sort((a, b) => a - b);
  const coords = {};
  for (const t of ids.slice(0, 1500)) {
    const rs = [...by.get(t)].sort((a, b) => a.frame - b.frame);
    coords[t] = [rs.map(r => r.frame), rs.map(r => num(r.x)), rs.map(r => num(r.y))];
  }
  return { name, kind, params, steps, warnings, in_browser: true, n_rows: rows.length, n_locs: nLocs, n_tracks_linked: nLinked,
    n_tracks: by.size, n_unlinked: rows.length - [...by.values()].reduce((s, v) => s + v.length, 0),
    columns: [...new Set(rows.flatMap(r => Object.keys(r)))], track_columns: cols,
    tracks: ids.slice(0, 2000).map(t => ({ track_number: t, n_points: by.get(t).length, ...Object.fromEntries(cols.map(c => [c, num(by.get(t)[0][c])])) })),
    coords, coords_truncated: by.size > 1500, unlinked_xy: [], rows };
}
