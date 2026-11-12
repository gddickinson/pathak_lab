// Shared state of the demo pages: one synthetic dataset and the settings, so Linking -> Features -> Mobility follow
// the same tracks. Everything is computed by the JS ports of the repo's Python (link/features/mobility engines).
import { linkPoints, locsToTxy, tracksTable } from './link_engine.js';
import { featuresTable } from './features_engine.js';
import { addDiffusion, addLocalizationError, addNNCounts, addVelocity } from './mobility_engine.js';
import { synthesize } from './synth.js';

// Linking defaults: Step 2's __main__ uses gapSize 18 and distance 3 px; minLinkSegments 6.
export const settings = { seed: 7, perKind: 10, frames: 80, background: 40, skip: 18, distance: 3, minSeg: 6,
  radii: [3, 5, 10, 20, 30] };
let cache = {};
export function update(patch) { Object.assign(settings, patch); cache = {}; }

export function data() {
  if (!cache.data) cache.data = synthesize({ seed: settings.seed, perKind: settings.perKind, frames: settings.frames, background: settings.background });
  return cache.data;
}

export function linked() {
  if (!cache.linked) {
    const d = data(), txy = locsToTxy(d.locs, d.pixelSize);
    const L = linkPoints(txy, settings.skip, settings.distance);
    const table = tracksTable(txy, L, d.locs.map(l => l.id), d.locs.map(l => l.intensity));
    const byId = new Map(d.locs.map((l, i) => [l.id, d.truth[i]]));
    cache.linked = { txy, ...L, table, truthOf: id => byId.get(id) };
  }
  return cache.linked;
}

// Simulated kind of a track = the kind of most of its points; 'pure' when all points come from one particle.
export function trackTruth() {
  if (!cache.truth) {
    const L = linked(), m = new Map();
    for (const r of L.table) {
      const t = L.truthOf(r.id);
      if (!m.has(r.track_number)) m.set(r.track_number, []);
      m.get(r.track_number).push(t);
    }
    cache.truth = new Map([...m].map(([k, ts]) => {
      const c = {};
      for (const t of ts) c[t.kind] = (c[t.kind] || 0) + 1;
      const kind = Object.entries(c).sort((a, b) => b[1] - a[1])[0][0];
      return [k, { kind, pure: ts.every(t => t.particle === ts[0].particle && t.particle >= 0) }];
    }));
  }
  return cache.truth;
}

export function features() {
  if (!cache.features) {
    try { cache.features = { rows: featuresTable(linked().table, settings.minSeg) }; }
    catch (e) { cache.features = { rows: [], error: e.message }; }
  }
  return cache.features;
}

export function mobility() {
  if (!cache.mobility) cache.mobility = addLocalizationError(addNNCounts(addVelocity(addDiffusion(features().rows)), settings.radii));
  return cache.mobility;
}

// One row per track (its first row), for tables and scatter plots.
export function perTrack(rows) {
  const seen = new Map();
  for (const r of rows) if (!seen.has(r.track_number)) seen.set(r.track_number, r);
  return [...seen.values()];
}
