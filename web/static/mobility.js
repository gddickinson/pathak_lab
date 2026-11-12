// Demo: Steps 5, 6, 8 and 11 (diffusion, velocity, neighbour counts, localisation spread) on the synthetic tracks.
import { $, app, esc, route } from './core.js';
import { fmt, legend, lines, scatter, table } from './plot.js';
import { features, mobility, perTrack, settings, trackTruth } from './state.js';
import { KIND_COLOR } from './synth.js';

const nanMean = v => { const f = v.filter(Number.isFinite); return f.length ? f.reduce((s, x) => s + x, 0) / f.length : NaN; };

route('mobility', async () => {
  app().innerHTML = `<p class="kicker">Demo · Steps 5, 6, 8, 11 · <span class="synth">synthetic data</span></p>
  <h1>Diffusion, velocity and neighbours</h1>
  <p class="lead">Step 5 measures each point's distance from the track's first point and the frame lag; Step 6 adds
    the frame interval to the next point, velocity (step / interval) and the direction from the origin; Step 8 counts
    other localisations in the same frame within fixed radii; Step 11 adds each track's mean position and the mean
    distance of its points from it.</p>
  ${legend([['mobile', KIND_COLOR.mobile], ['confined', KIND_COLOR.confined], ['immobile', KIND_COLOR.immobile], ['background', KIND_COLOR.background]])}
  <div class="two">
    <div class="panel"><h3>Squared distance from origin (Step 5)</h3><div id="p1"></div></div>
    <div class="panel"><h3>Mean velocity and localisation spread (Steps 6, 11)</h3><div id="p2"></div></div>
  </div>
  <div class="panel row">
    <div><label for="rad">Neighbour radius (Step 8)</label><select id="rad">${settings.radii.map(r => `<option value="${r}">${r} px</option>`).join('')}</select></div>
    <p class="small mute">Counts are of the other rows of the table in the same frame; in the pipeline Step 7 first adds the unlinked localisations, which this demo leaves out.</p>
  </div>
  <div id="tbl"></div>
  <p class="small mute">Units: pixels and frames, as in the scripts (108 nm pixels). Tracks: those kept on the
    <a href="#/features">features</a> page (more than ${settings.minSeg} points).</p>`;
  const draw = () => {
    if (features().error) { $('#tbl').innerHTML = `<p class="err">${esc(features().error)}</p>`; return; }
    const rows = mobility(), truth = trackTruth(), rad = $('#rad').value;
    const by = new Map();
    for (const r of rows) { if (!by.has(r.track_number)) by.set(r.track_number, []); by.get(r.track_number).push(r); }
    const kind = t => truth.get(t).kind;
    $('#p1').innerHTML = lines([...by].map(([t, rs]) => ({ pts: rs.map(r => [r.lagNumber, r.d_squared]), color: KIND_COLOR[kind(t)], title: `track ${t}` })),
      { xlabel: 'lag (frames from first point)', ylabel: 'distance² (px²)' });
    const per = perTrack(rows).map(r => ({ ...r, kind: kind(r.track_number),
      nn: nanMean(by.get(r.track_number).map(x => x[`nnCountInFrame_within_${rad}_pixels`])),
      vmax: Math.max(...by.get(r.track_number).map(x => x.distanceFromOrigin)) }));
    $('#p2').innerHTML = scatter(per.map(r => ({ x: r.meanLocDistanceFromCenter, y: r.meanVelocity, color: KIND_COLOR[r.kind], title: `track ${r.track_number}` })),
      { xlabel: 'mean distance from track centre (px)', ylabel: 'mean velocity (px / frame)' });
    table($('#tbl'), [{ key: 'track_number', label: 'track' }, { key: 'kind', label: 'simulated', fmt: v => `<span style="color:${KIND_COLOR[v]}">${esc(v)}</span>` },
      { key: 'n_segments', label: 'points' }, { key: 'meanVelocity', label: 'mean velocity' }, { key: 'vmax', label: 'max distance from origin' },
      { key: 'meanLocDistanceFromCenter', label: 'mean distance from centre' }, { key: 'nn', label: `mean neighbours ≤ ${rad} px`, fmt: v => fmt(v, 2) }], per);
  };
  $('#rad').onchange = draw;
  draw();
});
