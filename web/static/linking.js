// Demo: linking (Step 2, pointLinking.PointLinker.link_pts and savetracksCSV) on synthetic localisations.
import { $, app, route, slider } from './core.js';
import { drawTracks, fmt, legend } from './plot.js';
import { data, linked, settings, trackTruth, update } from './state.js';
import { KIND_COLOR } from './synth.js';

const hue = i => `hsl(${(i * 137.508) % 360} 85% 62%)`;

route('linking', async () => {
  app().innerHTML = `<p class="kicker">Demo · Step 2 · <span class="synth">synthetic data</span></p><h1>Linking localisations into tracks</h1>
  <p class="lead">Each unassigned localisation starts a track, frame by frame. The track grows to the nearest
    unassigned localisation in the next frame that has one closer than the linking distance, looking up to
    <i>gap</i> + 1 frames ahead. Tracks of one point are dropped when the table is saved.</p>
  <div class="side">
    <div class="panel params">
      <fieldset><legend>Synthetic data</legend><div id="dctl"></div><button type="button" class="alt" id="reseed">New random data</button></fieldset>
      <fieldset><legend>Linking (Step 2)</legend><div id="lctl"></div>
        <p class="hint">Step 2's <code>__main__</code> uses gap 18 frames and 3 px; the function defaults are 1 and 3.</p></fieldset>
    </div>
    <div>
      <div class="seg" role="group" aria-label="Colour tracks by"><button type="button" data-c="track" class="on">Colour by track</button><button type="button" data-c="kind">Colour by simulated motion</button></div>
      <div id="leg"></div>
      <div class="stage"><canvas id="cv" role="img" aria-label="Map of synthetic localisations and linked tracks"></canvas></div>
      <div class="tiles" id="stats"></div>
      <p class="small mute">Synthetic field of 64 × 64 pixels (108 nm), ${settings.frames} frames: mobile, confined and
        immobile particles (each detected in 88% of frames) plus unrelated background localisations. Ground truth is
        known only because the data are simulated.</p>
      <p><a class="btn" href="#/features">Features of these tracks →</a></p>
    </div>
  </div>`;
  let mode = 'track';
  const draw = () => {
    const d = data(), L = linked(), truth = trackTruth();
    const tracks = new Map();
    for (const r of L.table) { if (!tracks.has(r.track_number)) tracks.set(r.track_number, []); tracks.get(r.track_number).push([r.x, r.y]); }
    const linkedIds = new Set(L.table.map(r => r.id));
    drawTracks($('#cv'), {
      size: d.size,
      dots: d.locs.map((l, i) => ({ x: l.xnm / d.pixelSize, y: l.ynm / d.pixelSize,
        color: linkedIds.has(l.id) ? '#c9d6f2' : KIND_COLOR[d.truth[i].kind === 'background' ? 'background' : d.truth[i].kind] })),
      tracks: [...tracks].map(([k, pts]) => ({ id: k, pts, color: mode === 'kind' ? KIND_COLOR[truth.get(k).kind] : hue(k) })),
    });
    $('#leg').innerHTML = mode === 'kind' ? legend([['mobile', KIND_COLOR.mobile], ['confined', KIND_COLOR.confined], ['immobile', KIND_COLOR.immobile], ['background (unlinked dots)', KIND_COLOR.background]])
      : legend([['each track its own colour', '#9fe'], ['linked localisation', '#c9d6f2']]);
    const n = tracks.size, pure = [...truth.values()].filter(t => t.pure).length;
    const parts = new Set(d.truth.filter(t => t.particle >= 0).map(t => t.particle)).size;
    const lens = [...tracks.values()].map(p => p.length);
    $('#stats').innerHTML = [
      [d.locs.length, 'localisations'], [L.tracks.length, 'tracks before saving'], [n, 'tracks with ≥ 2 points'],
      [fmt(100 * linkedIds.size / d.locs.length, 1) + '%', 'localisations in a track'],
      [n ? fmt(lens.reduce((s, v) => s + v, 0) / n, 1) : '–', 'mean points per track'],
      [n ? fmt(100 * pure / n, 1) + '%' : '–', 'tracks from one simulated particle'], [parts, 'simulated particles'],
    ].map(([v, l]) => `<div class="tile"><b>${v}</b><span>${l}</span></div>`).join('') +
      (L.recursiveFailure ? '<p class="warn">Recursion limit reached (Step 2 would skip this file).</p>' : '');
  };
  const dctl = $('#dctl'), lctl = $('#lctl');
  slider(dctl, { id: 'per', label: 'Particles of each kind', min: 2, max: 30, value: settings.perKind }, v => { update({ perKind: v }); draw(); });
  slider(dctl, { id: 'bg', label: 'Background localisations', min: 0, max: 300, step: 10, value: settings.background }, v => { update({ background: v }); draw(); });
  slider(lctl, { id: 'gap', label: 'Gap (frames that may be skipped)', min: 0, max: 20, value: settings.skip }, v => { update({ skip: v }); draw(); });
  slider(lctl, { id: 'dist', label: 'Linking distance', min: 0.5, max: 6, step: 0.1, value: settings.distance, unit: ' px', fmt: v => v.toFixed(1) }, v => { update({ distance: v }); draw(); });
  $('#reseed').onclick = () => { update({ seed: 1 + Math.floor(Math.random() * 1e6) }); draw(); };
  document.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
    mode = b.dataset.c; document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b)); draw();
  });
  draw();
});
