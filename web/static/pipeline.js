// Pipeline: the preprocessing, analysis and postprocessing scripts in order, with the file patterns each one reads
// and writes (read from the code), so the hand-over from step to step can be followed.
import { api, app, esc, route } from './core.js';

const CLS = { '1_preprocessing': 'pre', '2_analysis': '', '3_postprocessing': 'post' };
const TITLE = { '1_preprocessing': 'Preprocessing', '2_analysis': 'Analysis steps', '3_postprocessing': 'Postprocessing' };
const codes = xs => xs.length ? xs.map(x => `<code>${esc(x)}</code>`).join(' ') : '<span class="mute">–</span>';

route('pipeline', async () => {
  const { stages } = await api('/pipeline');
  let html = `<p class="kicker">Workflow</p><h1>The analysis pipeline</h1>
  <p class="lead">Recordings are prepared, localised with ThunderSTORM (Step 1 writes the ImageJ macro), then each
    numbered step reads the previous step's table and writes a new one with extra columns. Postprocessing compiles
    and filters the results.</p>
  <div class="callout"><p class="small">File patterns are read from each script's own read/write calls; a script's
    <code>__main__</code> block sets the data folder. Steps 9 and 10 need FLIKA and an ROI file; the others run
    without a GUI.</p></div>`;
  let group = null;
  for (const s of stages) {
    if (s.group !== group) { if (group) html += '</ol>'; group = s.group; html += `<h2>${TITLE[group]}</h2><ol class="flow">`; }
    html += `<li class="${CLS[s.group]}"><a href="#/tool/${encodeURIComponent(s.id)}"><b class="mono">${esc(s.name)}</b></a>
      ${s.demos.map(d => `<a class="badge demo" href="#/${d}">demo</a>`).join('')}${s.local_run ? '<span class="badge run">local app</span>' : ''}${s.needs_gui ? '<span class="badge gui">FLIKA/Qt</span>' : ''}
      <div class="small">${esc((s.purpose || '').split('\n')[0])}</div>
      <div class="small mute">reads ${codes(s.inputs)} · writes ${codes(s.outputs)}</div></li>`;
  }
  app().innerHTML = html + '</ol>';
});
