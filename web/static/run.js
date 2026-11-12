// Run: upload a ThunderSTORM localisation file or a tracks table and run pipeline steps. Local app: the repo's
// Python (web/analysis.py). Static site: the JS ports in the browser (browser_run.js) for the steps that have them.
import { $, api, app, download, esc, isStatic, route } from './core.js';
import { drawTracks, fmt, table } from './plot.js';

const STEP_INFO = [
  ['link', 'Link localisations into tracks', 'Step 2 · pointLinking (addID, load_points, link_pts, savetracksCSV)'],
  ['features', 'Track features', 'Step 2 · trackFeatures.addFeaturesToTracksDF'],
  ['svm', 'SVM classes (mobile / intermediate / trapped)', 'Step 2 · svmClassification.predict_SPT_class · slow (grid search)'],
  ['nn', 'Nearest localisation in the frame', 'Steps 3 + 4 · getNN, addNNtoDF'],
  ['diffusion', 'Distance from origin', 'Step 5 · addDiffusiontoDF'],
  ['velocity', 'Velocity and direction', 'Step 6 · addVelocitytoDF'],
  ['missing', 'Add unlinked localisations', 'Step 7 · addMissingLocsToDF'],
  ['nncount', 'Neighbour counts within radii', 'Step 8 · getNNinFrame'],
  ['locerr', 'Track centre and spread', 'Step 11 · addLocalizationError'],
];
const BROWSER = ['link', 'features', 'diffusion', 'velocity', 'nncount', 'locerr'];
const DEFAULT_ON = ['link', 'features', 'diffusion', 'velocity', 'nncount', 'locerr'];
const FT_LABEL = { boxcox: 'Box-Cox (default)', none: 'None (legacy, as published)' };
let last = null;

route('run', async () => {
  const info = await api('/info'), d = info.defaults, st = isStatic();
  const num = (id, label, v, attrs, hint) => `<label for="${id}">${label}</label><input id="${id}" name="${id}" ${attrs} value="${v}">${hint ? `<p class="hint">${hint}</p>` : ''}`;
  app().innerHTML = `<p class="kicker">Your data</p><h1>Run the pipeline on your files</h1>
  <p class="lead">${st ? 'On this static site the steps with JS ports run in your browser; nothing is uploaded anywhere. The SVM, Steps 3+4 and 7 need the local app.'
    : 'The local app calls the repo\'s own Python functions on your file and keeps nothing after the last few runs.'}</p>
  <div class="side">
  <form class="panel params" id="pf" novalidate>
    <fieldset><legend>Input</legend>
      <label for="file">ThunderSTORM localisations (frame, x [nm], y [nm], id) or tracks (track_number, frame, x, y in px), .csv</label>
      <input id="file" name="file" type="file" accept=".csv,text/csv">
      ${st ? '' : `<label class="check"><input type="checkbox" name="example" value="1" id="example"> <span>Use the repo's example results table (minimal_model, local only)</span></label>
      <label for="tiff">Recording (.tif, optional: per-point intensities, as Step 2)</label><input id="tiff" name="tiff" type="file" accept=".tif,.tiff">
      <label for="training">SVM training features (.csv, optional; default: the file in the repo)</label><input id="training" name="training" type="file" accept=".csv">`}
    </fieldset>
    <fieldset><legend>Settings</legend>
      ${num('pixel_size_nm', 'Pixel size (nm)', d.pixel_size_nm, 'type="number" min="0" step="any"')}
      ${num('skip_frames', 'Gap: frames that may be skipped', d.skip_frames, 'type="number" min="0" step="1"', 'Step 2 __main__: 18')}
      ${num('link_distance', 'Linking distance (px)', d.link_distance, 'type="number" min="0" step="any"')}
      ${num('min_segments', 'Keep tracks with more than (points)', d.min_segments, 'type="number" min="0" step="1"')}
      ${num('radii', 'Neighbour radii (px, comma separated)', d.radii, 'type="text"')}
      <label for="feature_transform">SVM feature transform</label>
      <select id="feature_transform" name="feature_transform" ${st ? 'disabled' : ''}>
        ${(info.feature_transforms || [d.feature_transform]).map(t => `<option value="${t}" ${t === d.feature_transform ? 'selected' : ''}>${esc(FT_LABEL[t] || t)}</option>`).join('')}
      </select><p class="hint">Box-Cox lambdas are fitted on the training tracks and reused for yours. "None" is the legacy
      behaviour that published results used (the old code's Box-Cox never took effect).${st ? ' The SVM needs the local app.' : ''}</p>
    </fieldset>
    <fieldset><legend>Steps</legend>
      ${STEP_INFO.map(([k, l, h]) => `<label class="check"><input type="checkbox" name="step_${k}" value="1" ${DEFAULT_ON.includes(k) ? 'checked' : ''} ${st && !BROWSER.includes(k) ? 'disabled' : ''}>
        <span>${esc(l)}<br><small class="mute">${esc(h)}${st && !BROWSER.includes(k) ? ' · local app only' : ''}</small></span></label>`).join('')}
    </fieldset>
    <button type="submit" id="go">Run</button>
    <p id="msg" class="small" role="status"></p>
  </form>
  <div id="out">${last ? '' : '<p class="mute">Results appear here.</p>'}</div></div>`;
  const form = $('#pf');
  form.onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(form);
    $('#go').disabled = true; $('#msg').textContent = 'Running...';
    try {
      last = await api('/run', { body: fd });
      $('#msg').textContent = '';
      show(last);
    } catch (err) {
      $('#msg').innerHTML = `<span class="err">${esc(err.message.replace(/^\d+ /, '').replace(/^\{"detail":"(.*)"\}$/, '$1'))}</span>`;
    } finally { $('#go').disabled = false; }
  };
  if (last) show(last);
});

function show(r) {
  const out = $('#out');
  const tiles = [[r.n_locs ?? '–', 'localisations'], [r.n_tracks_linked, 'tracks linked'], [r.n_tracks, 'tracks in result'],
    [r.n_rows, 'rows'], [r.columns.length, 'columns']];
  out.innerHTML = `<h2 style="margin-top:0">${esc(r.name)}</h2>
    ${r.warnings.map(w => `<p class="small warn">${esc(w)}</p>`).join('')}
    <div class="tiles">${tiles.map(([v, l]) => `<div class="tile"><b>${v}</b><span>${l}</span></div>`).join('')}</div>
    ${r.svm_counts ? `<p>SVM classes (tracks): ${Object.entries(r.svm_counts).map(([k, v]) => `<b>${({ 1: 'mobile', 2: 'intermediate', 3: 'trapped' })[k] || k}</b> ${v}`).join(' · ')} <span class="small mute">(feature transform: ${esc(r.params.feature_transform || '')})</span></p>` : ''}
    <button type="button" id="dl">Download the result table (.csv)</button>
    <div class="stage" style="margin-top:12px"><canvas id="cv" role="img" aria-label="Map of the tracks in the result"></canvas></div>
    ${r.coords_truncated ? '<p class="small mute">Map shows the first 1500 tracks.</p>' : ''}
    <h3>Tracks (first row of each)</h3><div id="tt"></div>
    <p class="small mute">Steps run: ${esc(r.steps.join(', '))}. Coordinates in pixels.</p>`;
  const tr = Object.entries(r.coords).map(([k, [, xs, ys]]) => ({ id: +k, pts: xs.map((x, i) => [x, ys[i]]).filter(p => p[0] !== null && p[1] !== null) }));
  const all = tr.flatMap(t => t.pts).concat(r.unlinked_xy);
  const size = Math.max(8, ...all.map(p => Math.max(p[0], p[1]))) * 1.03;
  drawTracks($('#cv'), { size, dots: r.unlinked_xy.map(([x, y]) => ({ x, y, color: '#6b7896' }),),
    tracks: tr.map(t => ({ ...t, color: `hsl(${(t.id * 137.508) % 360} 85% 62%)` })) });
  table($('#tt'), [{ key: 'track_number', label: 'track' }, { key: 'n_points', label: 'points' },
    ...r.track_columns.map(c => ({ key: c, label: c, fmt: v => esc(fmt(v)) }))], r.tracks);
  $('#dl').onclick = async () => {
    if (r.in_browser) { const { toCsv } = await import('./browser_run.js'); download(r.name.replace(/\.csv$/i, '') + '_analysed.csv', toCsv(r.rows), 'text/csv'); return; }
    location.href = `api/runs/${r.run_id}/result.csv`;
  };
}
