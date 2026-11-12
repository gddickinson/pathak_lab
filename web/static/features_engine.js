// Track features (Step 2), ported from flika_scripts/piezo1_analysis/2_analysis/trackFeatures.py:
//   RadiusGyrationAsymmetrySkewnessKurtosis, FractalDimension, NetDisplacementEfficiency, SummedSinesCosines,
//   addLagDisplacementToDF, getNN and addFeaturesToTracksDF (the table Step 2 saves as *_tracksRG.csv).
// Points are [x, y] in pixels. Node-tested against the Python (engines.test.mjs).
import { npRound } from './link_engine.js';

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

// numpy's np.linalg.eig on the symmetric 2x2 gyration tensor [[a, b], [b, d]] (LAPACK dgeev/dlanv2): the first
// eigenvalue is the larger one when a >= d, the smaller one otherwise; eigenvectors form a rotation matrix. Python
// takes the eigenvector of the largest eigenvalue (first one on ties). Its sign is LAPACK's choice, and the sign of
// the skewness follows it: this reproduces numpy's sign exactly.
export function dominantEigenvector(a, b, d) {
  if (b === 0) return a >= d ? [1, 0] : [0, 1];
  const p = (a - d) / 2, r = Math.hypot(p, b);
  if (a >= d) { const z = p + r, tau = Math.hypot(b, z); return [z / tau, b / tau]; }
  const z = p - r, tau = Math.hypot(b, z);
  return [-b / tau, z / tau];
}

// scipy.stats.skew / kurtosis (biased, Fisher), NaN when the variance is ~0 as scipy does.
function moments(v) {
  const m = mean(v);
  let m2 = 0, m3 = 0, m4 = 0;
  for (const x of v) { const d = x - m; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
  m2 /= v.length; m3 /= v.length; m4 /= v.length;
  if (m2 <= (1e-15 * m) ** 2) return [NaN, NaN];
  return [m3 / m2 ** 1.5, m4 / (m2 * m2) - 3];
}

export function rgAsymmetrySkewKurtosis(points) {
  const n = points.length, cx = mean(points.map(p => p[0])), cy = mean(points.map(p => p[1]));
  let a = 0, b = 0, d = 0;
  for (const [x, y] of points) { a += (x - cx) ** 2; b += (x - cx) * (y - cy); d += (y - cy) ** 2; }
  a /= n; b /= n; d /= n;
  const half = (a - d) / 2, r = Math.hypot(half, b), l1 = (a + d) / 2 + r, l2 = (a + d) / 2 - r;
  const rg = Math.sqrt(l1 + l2);
  const asymmetry = -Math.log(1 - ((l1 - l2) ** 2) / (2 * (l1 + l2) ** 2));
  const v = dominantEigenvector(a, b, d), vv = v[0] * v[0] + v[1] * v[1];
  const proj = [];
  for (let i = 1; i < n; i++) proj.push(((points[i][0] - points[i - 1][0]) * v[0] + (points[i][1] - points[i - 1][1]) * v[1]) / vv);
  const [skewness, kurtosis] = moments(proj);
  return { rg, asymmetry, skewness, kurtosis };
}

export function fractalDimension(points) {
  const [x0, y0] = points[0];
  const slopes = points.filter(([x, y]) => x !== x0 || y !== y0).map(([x, y]) => (x !== x0 ? (y - y0) / (x - x0) : null));
  if (slopes.every(s => s === slopes[0])) throw new Error('Fractal Dimension cannot be calculated for points that are collinear');
  let L = 0, D = 0;
  for (let i = 1; i < points.length; i++) L += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  // largest distance between convex-hull vertices = largest distance between any two points
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
    D = Math.max(D, Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]));
  }
  const n = points.length;
  return Math.log(n) / Math.log(n * D / L);
}

export function netDisplacementEfficiency(points) {
  const n = points.length;
  const net = Math.hypot(points[0][0] - points[n - 1][0], points[0][1] - points[n - 1][1]);
  let ss = 0;
  for (let i = 1; i < n; i++) ss += (points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2;
  return { netDispl: net, efficiency: net * net / ((n - 1) * ss) };
}

// Mean cosine of the turning angles (the 'Straight' feature), after removing consecutive repeated positions.
export function straightness(points) {
  const drop = new Set();
  for (let i = 0; i + 1 < points.length; i++) if (points[i + 1][0] === points[i][0] && points[i + 1][1] === points[i][1]) drop.add(i);
  const p = points.filter((_, i) => !drop.has(i));
  const cos = [];
  for (let i = 0; i + 2 < p.length; i++) {
    const ab = [p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]], bc = [p[i + 2][0] - p[i + 1][0], p[i + 2][1] - p[i + 1][1]];
    cos.push((ab[0] * bc[0] + ab[1] * bc[1]) / (Math.hypot(...ab) * Math.hypot(...bc)));
  }
  return cos.length ? mean(cos) : NaN;
}

// Distance from each point to its nearest neighbour in the same frame (NaN when alone in its frame), as getNN.
export function nnDistInFrame(rows) {
  const byFrame = new Map();
  rows.forEach((r, i) => { if (!byFrame.has(r.frame)) byFrame.set(r.frame, []); byFrame.get(r.frame).push(i); });
  const out = new Array(rows.length).fill(NaN);
  for (const idx of byFrame.values()) {
    if (idx.length < 2) continue;
    for (const i of idx) {
      let best = Infinity;
      for (const j of idx) if (j !== i) best = Math.min(best, Math.hypot(rows[i].x - rows[j].x, rows[i].y - rows[j].y));
      out[i] = best;
    }
  }
  return out;
}

const groups = rows => {
  const g = new Map();
  rows.forEach((r, i) => { if (!g.has(r.track_number)) g.set(r.track_number, []); g.get(r.track_number).push(i); });
  return g;
};

const ROUND3 = ['track_length', 'radius_gyration', 'asymmetry', 'skewness', 'kurtosis', 'radius_gyration_scaled',
  'radius_gyration_scaled_nSegments', 'radius_gyration_scaled_trackLength', 'lag', 'meanLag', 'fracDimension',
  'netDispl', 'Straight', 'nnDist'];

// addFeaturesToTracksDF: keep tracks with more than minSegments points, add per-track features, the step length to
// the next point ('lag', empty on a track's last point), meanLag, track_length, scaled Rg and nnDist; round as saved.
export function featuresTable(input, minSegments = 6) {
  const counts = new Map();
  for (const r of input) counts.set(r.track_number, (counts.get(r.track_number) || 0) + 1);
  let rows = input.map(r => ({ ...r, n_segments: counts.get(r.track_number) }));
  if (minSegments !== 0) rows = rows.filter(r => r.n_segments > minSegments);
  const g = groups(rows);
  for (const idx of g.values()) {
    const pts = idx.map(i => [rows[i].x, rows[i].y]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
    const f = rgAsymmetrySkewKurtosis(pts);
    const inten = idx.map(i => rows[i].intensity).filter(v => v !== null && Number.isFinite(v));
    // np.mean / np.std of a pandas Series skip missing values (population std, ddof=0)
    const im = inten.length ? mean(inten) : NaN;
    const isd = inten.length ? Math.sqrt(mean(inten.map(v => (v - im) ** 2))) : NaN;
    const fd = fractalDimension(pts), nd = netDisplacementEfficiency(pts).netDispl, st = straightness(pts);
    for (const i of idx) Object.assign(rows[i], { radius_gyration: f.rg, asymmetry: f.asymmetry, skewness: f.skewness,
      kurtosis: f.kurtosis, track_intensity_mean: im, track_intensity_std: isd, fracDimension: fd, netDispl: nd, Straight: st });
  }
  rows.forEach((r, i) => {
    const nx = rows[i + 1];
    r.lag = nx ? Math.sqrt((nx.x - r.x) ** 2 + (nx.y - r.y) ** 2) : NaN;
  });
  for (const idx of g.values()) {
    rows[idx[idx.length - 1]].lag = NaN;
    const lags = idx.map(i => rows[i].lag).filter(Number.isFinite);
    const meanLag = lags.length ? mean(lags) : NaN, total = lags.reduce((s, v) => s + v, 0);
    for (const i of idx) Object.assign(rows[i], { meanLag, track_length: total });
  }
  const nn = nnDistInFrame(rows);
  rows.forEach((r, i) => {
    r.radius_gyration_scaled = r.radius_gyration / r.meanLag;
    r.radius_gyration_scaled_nSegments = r.radius_gyration / r.n_segments;
    r.radius_gyration_scaled_trackLength = r.radius_gyration / r.track_length;
    r.nnDist = nn[i];
    for (const k of ROUND3) r[k] = npRound(r[k], 3);
    r.track_intensity_mean = npRound(r.track_intensity_mean, 2);
    r.track_intensity_std = npRound(r.track_intensity_std, 2);
  });
  return rows;
}
