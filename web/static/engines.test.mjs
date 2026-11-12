// Node test: the JS ports give the same numbers as the repo's Python on the same synthetic input.
// Fixture: tests/fixtures/engine_cases.json (python scripts/make_engine_fixture.py). Run: node web/static/engines.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { PIXEL_SIZE_NM, linkPoints, locsToTxy, tracksTable } from './link_engine.js';
import { featuresTable, fractalDimension, netDisplacementEfficiency, rgAsymmetrySkewKurtosis, straightness } from './features_engine.js';
import { addDiffusion, addLocalizationError, addNNCounts, addVelocity } from './mobility_engine.js';

const fx = JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/engine_cases.json', import.meta.url)), 'utf8'));
let checks = 0;
function close(a, b, what, tol = 1e-9) {
  checks++;
  const missA = a === null || a === undefined || Number.isNaN(a), missB = b === null || b === undefined || Number.isNaN(b);
  if (missA || missB) { assert.ok(missA && missB, `${what}: ${a} vs ${b}`); return; }
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${what}: JS ${a} vs Python ${b}`);
}
const toRows = (cols, rows) => rows.map(r => Object.fromEntries(cols.map((c, i) => [c, r[i]])));
function compareTable(js, cols, py, what) {
  assert.equal(js.length, py.length, `${what}: row count`);
  py.forEach((row, i) => cols.forEach((c, k) => close(js[i][c], row[k], `${what} row ${i} ${c}`)));
}

// Linking + savetracksCSV table
const locs = fx.locs.map(([frame, id, xnm, ynm, intensity]) => ({ frame, id, xnm, ynm, intensity }));
const txy = locsToTxy(locs, fx.pixel_size_nm);
for (const c of fx.link) {
  const linked = linkPoints(txy, c.skip, c.distance);
  assert.deepEqual(linked.tracks, c.tracks, `tracks skip=${c.skip} distance=${c.distance}`);
  assert.equal(linked.recursiveFailure, c.recursiveFailure);
  checks++;
  const table = tracksTable(txy, linked, locs.map(l => l.id), locs.map(l => l.intensity));
  compareTable(table, c.table_columns, c.table, `table skip=${c.skip}`);
}

// Pixel size: the JS default is pipeline_defaults.PIXEL_SIZE_NM; another size round-trips x [nm] as savetracksCSV does
assert.equal(PIXEL_SIZE_NM, fx.pixel_size_nm, 'JS PIXEL_SIZE_NM = pipeline_defaults.PIXEL_SIZE_NM'); checks++;
{
  const c = fx.pixel_case, t2 = locsToTxy(locs, c.pixel_size_nm), linked = linkPoints(t2, c.skip, c.distance);
  const table = tracksTable(t2, linked, locs.map(l => l.id), null, c.pixel_size_nm);
  compareTable(table, c.table_columns, c.table, `table pixel size ${c.pixel_size_nm}`);
}

// Per-track features (unrounded)
const tracks = toRows(fx.link[1].table_columns, fx.link[1].table);
for (const t of fx.per_track) {
  const pts = tracks.filter(r => r.track_number === t.track_number).map(r => [r.x, r.y]);
  const f = rgAsymmetrySkewKurtosis(pts), nd = netDisplacementEfficiency(pts);
  close(f.rg, t.rg, 'rg'); close(f.asymmetry, t.asymmetry, 'asymmetry'); close(f.skewness, t.skewness, 'skewness');
  close(f.kurtosis, t.kurtosis, 'kurtosis'); close(fractalDimension(pts), t.fracDimension, 'fracDimension');
  close(nd.netDispl, t.netDispl, 'netDispl'); close(nd.efficiency, t.efficiency, 'efficiency');
  close(straightness(pts), t.straight, 'straight');
}
for (const [name, e] of Object.entries(fx.edge_tracks)) {
  const f = rgAsymmetrySkewKurtosis(e.points);
  for (const k of ['rg', 'asymmetry', 'skewness', 'kurtosis']) close(f[k], e[k], `${name} ${k}`);
  close(straightness(e.points), e.straight, `${name} straight`);
  close(fractalDimension(e.points), e.fracDimension, `${name} fracDimension`);
}
assert.throws(() => fractalDimension([[0, 0], [1, 1], [2, 2]]), /collinear/);

// The features table (rounded as saved)
const feat = featuresTable(tracks, fx.features.min_segments);
compareTable(feat, fx.features.columns, fx.features.rows, 'features');

// Steps 5, 6, 8, 11 on the Python features table
const pyFeat = toRows(fx.features.columns, fx.features.rows);
const mob = addLocalizationError(addNNCounts(addVelocity(addDiffusion(pyFeat)), fx.mobility.radii));
compareTable(mob, fx.mobility.columns, fx.mobility.rows, 'mobility');

console.log(`engines: ${checks} checks passed (linking, features, steps 5/6/8/11)`);
