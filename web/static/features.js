// Demo: track features (Step 2, trackFeatures.addFeaturesToTracksDF) of the linked synthetic tracks.
import { $, app, esc, route, slider } from './core.js';
import { fmt, legend, scatter, table } from './plot.js';
import { features, perTrack, settings, trackTruth, update } from './state.js';
import { KIND_COLOR } from './synth.js';

const FEATS = [
  ['radius_gyration', 'Rg (px)'], ['asymmetry', 'asymmetry'], ['skewness', 'skewness'], ['kurtosis', 'kurtosis'],
  ['fracDimension', 'fractal dimension'], ['netDispl', 'net displacement (px)'], ['Straight', 'straightness'],
  ['meanLag', 'mean step (px)'], ['track_length', 'path length (px)'], ['radius_gyration_scaled', 'Rg / mean step'],
  ['n_segments', 'points'], ['track_intensity_mean', 'mean intensity'],
];
const label = k => (FEATS.find(f => f[0] === k) || [k, k])[1];

route('features', async () => {
  app().innerHTML = `<p class="kicker">Demo · Step 2 · <span class="synth">synthetic data</span></p><h1>Track features</h1>
  <p class="lead">For every track longer than the minimum, Step 2 computes shape features that later feed the SVM
    classifier, plus the step to the next point and the distance to the nearest localisation in the same frame.
    These are the tracks from the <a href="#/linking">linking demo</a>.</p>
  <div class="side">
    <div class="panel params">
      <div id="ctl"></div>
      <label for="fx">x axis</label><select id="fx">${FEATS.map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join('')}</select>
      <label for="fy">y axis</label><select id="fy">${FEATS.map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join('')}</select>
      <h3>How they are computed</h3>
      <ul class="small">
        <li><b>Rg</b>: √(λ₁+λ₂) of the gyration tensor (mean outer product of positions about the centroid).</li>
        <li><b>asymmetry</b>: −ln(1 − (λ₁−λ₂)² / 2(λ₁+λ₂)²).</li>
        <li><b>skewness, kurtosis</b>: of the steps projected on the main axis (biased; Fisher kurtosis).</li>
        <li><b>fractal dimension</b>: ln N / ln(N·d / L), with N points, d the largest distance between points, L the path length.</li>
        <li><b>net displacement</b>: first to last point. <b>straightness</b>: mean cosine of the turning angles.</li>
        <li><b>Rg scaled</b>: by mean step, by number of points and by path length.</li>
      </ul>
    </div>
    <div>
      ${legend([['mobile', KIND_COLOR.mobile], ['confined', KIND_COLOR.confined], ['immobile', KIND_COLOR.immobile], ['background', KIND_COLOR.background]])}
      <div class="panel" id="plot"></div>
      <p class="small mute">Colour = simulated motion of most of the track's points (known only for synthetic data).
        The sign of the skewness follows the direction numpy's eigen-solver picks for the main axis, which is
        arbitrary; the JS port reproduces numpy's choice.</p>
      <p id="msg" role="status"></p><div id="tbl"></div>
      <p><a class="btn" href="#/mobility">Diffusion and neighbours →</a></p>
    </div>
  </div>`;
  $('#fx').value = 'radius_gyration'; $('#fy').value = 'asymmetry';
  const draw = () => {
    const { rows, error } = features(), truth = trackTruth();
    $('#msg').innerHTML = error ? `<span class="err">${esc(error)}: Step 2 skips the whole file in this case.</span>` : `<span class="small mute">${perTrack(rows).length} tracks with more than ${settings.minSeg} points.</span>`;
    const tr = perTrack(rows).map(r => ({ ...r, kind: truth.get(r.track_number).kind }));
    const fx = $('#fx').value, fy = $('#fy').value;
    $('#plot').innerHTML = scatter(tr.map(r => ({ x: r[fx], y: r[fy], color: KIND_COLOR[r.kind], title: `track ${r.track_number} (${r.kind}): ${fmt(r[fx])}, ${fmt(r[fy])}` })),
      { xlabel: label(fx), ylabel: label(fy) });
    table($('#tbl'), [{ key: 'track_number', label: 'track' }, { key: 'kind', label: 'simulated', fmt: v => `<span style="color:${KIND_COLOR[v]}">${esc(v)}</span>` },
      ...['n_segments', 'radius_gyration', 'asymmetry', 'skewness', 'kurtosis', 'fracDimension', 'netDispl', 'Straight', 'meanLag'].map(k => ({ key: k, label: label(k) }))], tr);
  };
  slider($('#ctl'), { id: 'minseg', label: 'Keep tracks with more than', min: 0, max: 30, value: settings.minSeg, unit: ' points' }, v => { update({ minSeg: v }); draw(); });
  $('#fx').onchange = draw; $('#fy').onchange = draw;
  draw();
});
