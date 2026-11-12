// Tool catalogue: #/tools (filterable list by folder/plugin) and #/tool/<id> (one file in detail).
// All text is read from the repo's files at build time (web/catalogue.py); nothing here is hand-written per tool.
import { $, api, app, esc, list, route } from './core.js';
import { markdown } from './md.js';

const DEMO_TITLE = { linking: 'Linking demo', features: 'Track-features demo', mobility: 'Diffusion and neighbours demo' };
const badges = t => [
  ...t.demos.map(d => `<span class="badge demo">demo: ${esc(d)}</span>`),
  t.local_run ? '<span class="badge run">runs in local app</span>' : '',
  t.needs_gui ? '<span class="badge gui">needs FLIKA/Qt</span>' : '',
  t.identical_to.length ? `<span class="badge">identical copy ×${t.identical_to.length + 1}</span>` : '',
].join('');

route('tools', async (args, query) => {
  const cat = await api('/catalogue');
  const plugins = Object.fromEntries(cat.plugins.map(p => [p.name, p]));
  app().innerHTML = `<p class="kicker">Catalogue</p><h1>The tools</h1>
  <p class="lead">${cat.tools.length} Python files. Purposes are quoted from each file's docstring or, where it has
    none, from its folder's README (the source is named on each tool page).</p>
  <div class="filters panel">
    <div><label for="q">Search names and purposes</label><input id="q" type="search" placeholder="e.g. diffusion, SVM, overlay"></div>
    <div><label for="grp">Folder</label><select id="grp"><option value="">All folders</option>
      ${list(cat.groups, g => `<option value="${g.id}">${esc(g.title)}</option>`)}</select></div>
    <div><label for="flag">Show</label><select id="flag"><option value="">Everything</option>
      <option value="demo">With an in-browser demo</option><option value="run">Runs in the local app</option>
      <option value="nogui">No FLIKA/Qt needed</option></select></div>
  </div>
  <p id="count" class="small mute" role="status"></p><div id="out"></div>`;
  const q = $('#q'), grp = $('#grp'), flag = $('#flag');
  q.value = query.q || ''; grp.value = query.group || '';
  const draw = () => {
    const s = q.value.trim().toLowerCase();
    const hit = cat.tools.filter(t => (!grp.value || t.group === grp.value)
      && (!s || (t.name + ' ' + t.purpose + ' ' + (t.plugin || '')).toLowerCase().includes(s))
      && (flag.value !== 'demo' || t.demos.length) && (flag.value !== 'run' || t.local_run) && (flag.value !== 'nogui' || !t.needs_gui));
    $('#count').textContent = `${hit.length} of ${cat.tools.length} files`;
    const sections = [];
    for (const g of cat.groups) {
      const ts = hit.filter(t => t.group === g.id);
      if (!ts.length) continue;
      if (g.id !== 'plugin') { sections.push(`<h2>${esc(g.title)}</h2><p class="small mute">${esc(g.about)}</p><div class="grid wide">${ts.map(card).join('')}</div>`); continue; }
      sections.push(`<h2>${esc(g.title)}</h2><p class="small mute">${esc(g.about)}</p>`);
      for (const name of [...new Set(ts.map(t => t.plugin))]) {
        const p = plugins[name];
        sections.push(`<h3>${esc(name)}${p.version ? ` <span class="mute">v${esc(p.version)}</span>` : ''}</h3><p class="small">${esc(p.summary)}</p>
          <div class="grid wide">${ts.filter(t => t.plugin === name).map(t => card(t, true)).join('')}</div>`);
      }
    }
    $('#out').innerHTML = sections.join('') || '<p class="mute">No tool matches.</p>';
  };
  const card = (t, inPlugin = false) => `<a class="card toolcard" href="#/tool/${encodeURIComponent(t.id)}">
    <b>${esc(t.name)}</b><span class="path">${esc(t.path)} · ${t.lines} lines · ${t.n_functions} functions</span>
    ${inPlugin ? '' : `<span>${esc(t.purpose || 'No docstring or README description.')}</span>`}<span>${badges(t)}</span></a>`;
  [q, grp, flag].forEach(el => el.addEventListener('input', draw));
  draw();
});

const files = xs => xs.length ? `<span class="files">${xs.map(x => `<code>${esc(x)}</code>`).join('')}</span>` : '<span class="mute">none found in the code</span>';

function fnBlock(f) {
  const params = f.params.filter(p => p.default !== null);
  return `<div class="fn"><code class="sig">${esc(f.owner ? f.owner + '.' : '')}${esc(f.signature)}</code>
    <span class="small mute"> · line ${f.line}</span>
    ${params.length ? `<div class="small">Defaults: ${params.map(p => `<code>${esc(p.name)}=${esc(p.default)}</code>`).join(' ')}</div>` : ''}
    ${f.doc ? `<pre>${esc(f.doc)}</pre>` : ''}</div>`;
}

route('tool', async ([id]) => {
  const t = await api('/tool/' + encodeURIComponent(id));
  const fns = t.functions.length + t.classes.reduce((s, c) => s + c.methods.length, 0);
  app().innerHTML = `<p class="kicker"><a href="#/tools?group=${esc(t.group)}">${esc(t.group_title)}</a>${t.plugin ? ' · ' + esc(t.plugin) : ''}</p>
  <h1 style="overflow-wrap:anywhere">${esc(t.name)}</h1>
  <p class="mono small mute" style="overflow-wrap:anywhere">${esc(t.path)} · ${t.lines} lines ·
    <a href="${esc(t.source_url)}" target="_blank" rel="noopener">source on GitHub</a></p>
  <p>${badges(t)}</p>
  <h2>Purpose</h2>
  ${t.purpose ? `<div class="quote">${esc(t.purpose)}</div><p class="small mute">From ${esc(t.purpose_source)}.</p>` : '<p class="mute">The file has no docstring and its folder README does not describe it.</p>'}
  ${t.readme_note ? `<p class="small">Folder README: ${esc(t.readme_note)}</p>` : ''}
  ${t.group === 'plugin' && t.docstring ? `<h3>Module docstring</h3><div class="quote">${esc(t.docstring)}</div>` : ''}
  ${t.demos.length ? `<p>${t.demos.map(d => `<a class="btn" href="#/${d}">${DEMO_TITLE[d]}</a>`).join('')}${t.local_run ? '<a class="btn alt" href="#/run">Run on your file</a>' : ''}</p>` : t.local_run ? '<p><a class="btn alt" href="#/run">Run on your file</a></p>' : ''}
  <h2>Inputs and outputs</h2>
  <dl class="kv"><dt>Reads</dt><dd>${files(t.inputs)}</dd><dt>Writes</dt><dd>${files(t.outputs)}</dd>
    <dt>Needs</dt><dd>${t.needs.length ? esc(t.needs.join(', ')) : 'standard library only'}</dd>
    <dt>Run as</dt><dd>${t.has_main ? 'script (edit the settings in its <code>__main__</code> block)' : t.group === 'plugin' ? 'FLIKA plugin module' : 'module / top-level script'}</dd></dl>
  <p class="small mute">File-name patterns found in the file's read/write calls (folder paths left out).
    ${t.ipython_magic ? 'The file contains IPython magics (<code>%gui qt</code>): run it in IPython or Spyder.' : ''}
    ${t.parse_error ? `It could not be parsed: ${esc(t.parse_error)}.` : ''}</p>
  ${t.settings.length ? `<h2>Settings in <code>__main__</code></h2><dl class="kv">${t.settings.map(s => `<dt>${esc(s.name)}</dt><dd>${esc(JSON.stringify(s.value))}</dd>`).join('')}</dl>` : ''}
  <h2>Functions (${fns})</h2>
  ${t.functions.map(fnBlock).join('') || '<p class="mute">No top-level functions.</p>'}
  ${t.classes.map(c => `<h3>class ${esc(c.name)}${c.bases.length ? `(${esc(c.bases.join(', '))})` : ''}</h3>${c.doc ? `<pre>${esc(c.doc)}</pre>` : ''}${c.methods.map(fnBlock).join('')}`).join('')}
  ${t.identical_to.length ? `<h2>Identical copies</h2><p>${t.identical_to.map(i => `<a href="#/tool/${encodeURIComponent(i)}">${esc(i.replaceAll('__', '/'))}</a>`).join('<br>')}</p>` : ''}
  ${t.instructions ? `<h2>Step instructions</h2><div class="panel md">${markdown(t.instructions, { skipTitle: true })}</div>` : ''}`;
});
