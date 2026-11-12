// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py; the Run page's
// analysis runs in the browser (browser_run.js) for the steps that have JS ports. Everything else needs the local app.
import { BROWSER_STEPS, parseCsv, runInBrowser } from './browser_run.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export const NEEDS_SERVER = 'needs the full app running locally (Python server): see the README, "Run the web app"';

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'run') return run(body);
  if (clean.startsWith('runs/')) throw new Error('Server-side downloads ' + NEEDS_SERVER + '.');
  return file(clean);
}

async function run(fd) {
  const steps = BROWSER_STEPS.concat(['svm', 'nn', 'missing']).filter(s => fd.get('step_' + s));
  const off = steps.filter(s => !BROWSER_STEPS.includes(s));
  if (off.length) throw new Error(`Step(s) ${off.join(', ')} ${NEEDS_SERVER}.`);
  if (!steps.length) throw new Error('Choose at least one step.');
  const up = fd.get('file');
  if (!up || !up.name) throw new Error('Upload a CSV file.');
  const num = (k, int = false) => { const v = Number(fd.get(k)); if (!Number.isFinite(v) || (int && !Number.isInteger(v))) throw new Error(`${k} must be a ${int ? 'whole ' : ''}number`); return v; };
  const p = { pixel_size_nm: num('pixel_size_nm'), skip_frames: num('skip_frames', true), link_distance: num('link_distance'),
    min_segments: num('min_segments', true), radii: String(fd.get('radii')).split(',').filter(s => s.trim()).map(Number) };
  if (p.radii.some(r => !Number.isFinite(r))) throw new Error('Radii must be numbers separated by commas.');
  return runInBrowser(parseCsv(await up.text()), p, steps, up.name);
}
