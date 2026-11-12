// Home: what the repo is, real counts read from it, and the ways in (catalogue, pipeline, demos, run).
import { api, app, esc, isStatic, route } from './core.js';

route('home', async () => {
  const info = await api('/info');
  const c = info.counts;
  const card = (href, title, sub, text) => `<a class="card" href="${href}"><b>${title}</b><small>${sub}</small><span>${text}</span></a>`;
  app().innerHTML = `<section class="hero">
    <p class="kicker">PIEZO1 · TIRF · single-particle tracking</p>
    <h1 class="glow">Pathak lab TIRF tools</h1>
    <p class="lead">Python code created for the Pathak lab to analyse TIRF recordings of PIEZO1: FLIKA plugins for
      viewing localisations and tracks, and scripts that link localisations into tracks, measure track shape, diffusion
      and neighbours, and classify tracks as mobile, intermediate or trapped.</p>
    <a class="btn" href="#/tools">Browse the tools</a><a class="btn alt" href="#/linking">Try a demo</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${c.scripts}</b><span>analysis scripts</span></div>
    <div class="tile"><b>${c.plugins}</b><span>FLIKA plugins (${c.plugin_files} files)</span></div>
    <div class="tile"><b>${c.functions}</b><span>functions and methods</span></div>
    <div class="tile"><b>${c.pipeline_steps}</b><span>numbered pipeline steps</span></div>
    <div class="tile"><b>${c.demos}</b><span>in-browser demos</span></div>
  </div>
  <h2>Explore</h2>
  <div class="grid wide">
    ${card('#/tools', 'Tool catalogue', `${c.files} Python files`, 'Every script and plugin module: purpose (from its own docstring or folder README), functions, parameters, files read and written, source.')}
    ${card('#/pipeline', 'The analysis pipeline', 'preprocessing → Steps 1-11 → postprocessing', 'The order the scripts run in and the file each step hands to the next.')}
    ${card('#/linking', 'Demo: linking', 'Step 2 · synthetic data', 'Link localisations into tracks with the same nearest-neighbour rule as the lab code, and see how gap and distance settings change the tracks.')}
    ${card('#/features', 'Demo: track features', 'Step 2 · synthetic data', 'Radius of gyration, asymmetry, skewness, kurtosis, fractal dimension, net displacement and straightness of every track.')}
    ${card('#/mobility', 'Demo: diffusion and neighbours', 'Steps 5, 6, 8, 11 · synthetic data', 'Displacement from origin, velocity, neighbour counts and localisation spread per track.')}
    ${card('#/run', 'Run on your own files', isStatic() ? 'in your browser (JS ports) or the local app' : 'local app: the repo\'s Python', 'Upload a ThunderSTORM localisation file or a tracks table and run the pipeline steps; download the result table.')}
  </div>
  <div class="callout a"><p class="small">The demos use <b>synthetic</b> data generated in your browser (seeded random
    walks), labelled as such on each page. The JS ports are tested against the repo's Python on the same input
    (${esc('web/static/engines.test.mjs')}).</p></div>`;
});
