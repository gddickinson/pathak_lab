// Linking (Step 2), ported from flika_scripts/piezo1_analysis/2_analysis/pointLinking.py:
//   addID + load_points  -> locsToTxy      PointLinker.link_pts / extend_track -> linkPoints
//   savetracksCSV (table part) -> tracksTable
// Node-tested against the Python on the same input (engines.test.mjs, tests/fixtures/engine_cases.json).

// ThunderSTORM rows {frame, id, xnm, ynm} -> [frame, x, y] in pixels. addID makes frames start at 0 (frame - 1);
// load_points divides the nm coordinates by the pixel size.
export const PIXEL_SIZE_NM = 108;  // pipeline_defaults.PIXEL_SIZE_NM (the lab's camera); the Run page sends /api/info's value

export function locsToTxy(locs, pixelSize = PIXEL_SIZE_NM) {
  return locs.map(r => [Math.trunc(r.frame) - 1, r.xnm / pixelSize, r.ynm / pixelSize]);
}

// PointLinker.link_pts: every unassigned point starts a track (frames in increasing order, points in file order);
// a track grows to the nearest unassigned point of the first later frame (up to maxFramesSkipped + 1 frames ahead)
// that has one strictly closer than maxDistance. Python recurses once per added point and stops at
// depth 10000 (recursiveFailure); this loop applies the same limit.
export function linkPoints(txy, maxFramesSkipped, maxDistance) {
  const nFrames = txy.reduce((m, p) => Math.max(m, p[0]), -1) + 1;
  const ptsIdxByFrame = Array.from({ length: nFrames }, () => []);
  txy.forEach((p, i) => { if (p[0] >= 0 && Number.isInteger(p[0])) ptsIdxByFrame[p[0]].push(i); });
  const remaining = ptsIdxByFrame.map(a => a.map(() => true));
  const frames = [...new Set(txy.map(p => Math.trunc(p[0])))].sort((a, b) => a - b);
  const tracks = [];
  let recursiveFailure = false;
  for (const frame of frames) {
    for (let k = 0; k < ptsIdxByFrame[frame].length; k++) {
      if (!remaining[frame][k]) continue;
      remaining[frame][k] = false;
      const track = [ptsIdxByFrame[frame][k]];
      for (let depth = 0; ; depth++) {
        if (depth >= 10000) { recursiveFailure = true; break; }
        const pt = txy[track[track.length - 1]];
        let next = -1, nextFrame = -1;
        for (let dt = 1; dt <= maxFramesSkipped + 1; dt++) {
          const f = Math.trunc(pt[0]) + dt;
          if (f >= nFrames) break;
          let best = Infinity, bestK = -1, any = false;
          for (let j = 0; j < ptsIdxByFrame[f].length; j++) {
            if (!remaining[f][j]) continue;
            any = true;
            const q = txy[ptsIdxByFrame[f][j]];
            const d = Math.sqrt((q[1] - pt[1]) ** 2 + (q[2] - pt[2]) ** 2);
            if (d < best) { best = d; bestK = j; }
          }
          if (!any) continue;
          if (best < maxDistance) { next = bestK; nextFrame = f; break; }
        }
        if (next < 0) break;
        track.push(ptsIdxByFrame[nextFrame][next]);
        remaining[nextFrame][next] = false;
      }
      tracks.push(track);
    }
  }
  return { tracks, recursiveFailure, ptsIdxByFrame };
}

// numpy-style rounding (rint: halves to even) at `d` decimals, as pandas .round().
export function npRound(x, d) {
  if (x === null || !Number.isFinite(x)) return x;
  const f = 10 ** d, y = x * f;
  const r = Math.abs(y % 1) === 0.5 ? 2 * Math.round(y / 2) : Math.round(y);
  return r / f;
}

// savetracksCSV: rows of tracks with more than one point, numbered 0.. in track order; ids looked up through the
// frame-ordered point index (as the Python does); intensity rounded to 2 decimals; x/y [nm] = pixels x pixelSize.
export function tracksTable(txy, linked, ids, intensities, pixelSize = PIXEL_SIZE_NM) {
  const flat = linked.ptsIdxByFrame.flat();
  const rows = [];
  let n = 0;
  for (const t of linked.tracks) {
    if (t.length < 2) continue;
    for (const i of t) {
      const p = txy[i];
      rows.push({ track_number: n, frame: Math.trunc(p[0]), x: p[1], y: p[2],
        intensity: intensities ? npRound(intensities[i], 2) : null, id: ids[flat[i]],
        'x [nm]': p[1] * pixelSize, 'y [nm]': p[2] * pixelSize });
    }
    n++;
  }
  return rows;
}
