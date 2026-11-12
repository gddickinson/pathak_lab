// Diffusion, velocity, neighbour counts and localisation error, ported from flika_scripts/piezo1_analysis/2_analysis:
//   Step_5_addDiffusion.addDiffusiontoDF, Step_6_addVelocity.addVelocitytoDF, Step_8_addNNcounts.getNNinFrame,
//   Step_11_addLocalizationError.addLocalizationError. Rows are objects with track_number, frame, x, y (pixels) and
//   'lag' (from the features table). Node-tested against the Python (engines.test.mjs).

const order = rows => {
  const g = new Map();
  rows.forEach((r, i) => { if (!g.has(r.track_number)) g.set(r.track_number, []); g.get(r.track_number).push(i); });
  return g;
};
const nanMean = v => { const f = v.filter(Number.isFinite); return f.length ? f.reduce((s, x) => s + x, 0) / f.length : NaN; };

// Step 5: positions relative to the track's first point, frame lag from it, distance from origin and squares.
// Tracks are regrouped in order of first appearance, as the Python concatenates them.
export function addDiffusion(rows) {
  const out = [];
  for (const idx of order(rows).values()) {
    const tr = idx.map(i => ({ ...rows[i] }));
    const minFrame = Math.min(...tr.map(r => r.frame));
    const o = tr.find(r => r.frame === minFrame);
    for (const r of tr) {
      r.zeroed_X = r.x - o.x; r.zeroed_Y = r.y - o.y; r.lagNumber = r.frame - minFrame;
      r.distanceFromOrigin = Math.sqrt(r.zeroed_X ** 2 + r.zeroed_Y ** 2);
      out.push(r);
    }
  }
  for (const r of out) { r.d_squared = r.distanceFromOrigin ** 2; r.lag_squared = Number.isFinite(r.lag) ? r.lag ** 2 : NaN; }
  return out;
}

// Step 6: rate of change of distance from origin, frame interval to the next row (empty after the last row of the
// table or when not positive), velocity = lag / dt, direction from the origin in degrees (0-360), track mean velocity.
export function addVelocity(rows) {
  const out = [];
  for (const idx of order(rows).values()) {
    idx.forEach((i, k) => {
      const r = { ...rows[i] };
      if (k === 0) r['dy-dt: distance'] = 0;
      else { const p = rows[idx[k - 1]]; r['dy-dt: distance'] = (r.distanceFromOrigin - p.distanceFromOrigin) / (r.lagNumber - p.lagNumber); }
      out.push(r);
    });
  }
  out.forEach((r, i) => {
    const dt = (i + 1 < out.length ? out[i + 1].frame : 0) - r.frame;
    r.dt = dt <= 0 ? NaN : dt;
    r.velocity = r.lag / r.dt;
    let deg = Math.atan2(r.zeroed_Y, r.zeroed_X) / Math.PI * 180;
    if (deg < 0) deg = 360 + deg;
    r.direction_Relative_To_Origin = deg;
  });
  for (const idx of order(out).values()) {
    const m = nanMean(idx.map(i => out[i].velocity));
    for (const i of idx) out[i].meanVelocity = m;
  }
  return out;
}

// Step 8: number of other localisations in the same frame within each radius (distance <= r, pixels).
export function addNNCounts(rows, radii = [3, 5, 10, 20, 30]) {
  const out = rows.map(r => ({ ...r }));
  const byFrame = new Map();
  out.forEach((r, i) => { if (!byFrame.has(r.frame)) byFrame.set(r.frame, []); byFrame.get(r.frame).push(i); });
  for (const rad of radii) {
    const key = `nnCountInFrame_within_${rad}_pixels`;
    for (const idx of byFrame.values()) {
      for (const i of idx) {
        let c = 0;
        for (const j of idx) if (Math.sqrt((out[i].x - out[j].x) ** 2 + (out[i].y - out[j].y) ** 2) <= rad) c++;
        out[i][key] = c - 1;
      }
    }
  }
  return out;
}

// Step 11: track mean position, each point's distance from it, and the track mean of that distance.
export function addLocalizationError(rows) {
  const out = rows.map(r => ({ ...r }));
  for (const idx of order(out).values()) {
    const mx = nanMean(idx.map(i => out[i].x)), my = nanMean(idx.map(i => out[i].y));
    for (const i of idx) {
      out[i].mean_X = mx; out[i].mean_Y = my;
      out[i].distanceFromMean = Math.sqrt((mx - out[i].x) ** 2 + (my - out[i].y) ** 2);
    }
    const md = nanMean(idx.map(i => out[i].distanceFromMean));
    for (const i of idx) out[i].meanLocDistanceFromCenter = md;
  }
  return out;
}
