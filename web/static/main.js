// Entry point: lazy page modules, the mode label, a faint TIRF-like background (blinking puncta), then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = {
  'home.js': ['home'], 'tools.js': ['tools', 'tool'], 'pipeline.js': ['pipeline'], 'linking.js': ['linking'],
  'features.js': ['features'], 'mobility.js': ['mobility'], 'run.js': ['run'], 'about.js': ['about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: catalogue read from the repo at build time; demos run the JS ports in your browser on synthetic data'
  : 'local app: the repo\'s Python runs on this computer';

// Background: a few puncta drifting and blinking on a dark field. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const ps = Array.from({ length: 26 }, () => ({ x: Math.random(), y: Math.random(), on: Math.random() < 0.6, s: 0.6 + Math.random() }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 90) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      p.x += (Math.random() - 0.5) * 0.002 * p.s; p.y += (Math.random() - 0.5) * 0.002 * p.s;
      if (Math.random() < 0.02) p.on = !p.on;
      if (!p.on) continue;
      const x = p.x * canvas.width, y = p.y * canvas.height, r = 9 * p.s;
      const grd = g.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, 'rgba(255,77,109,.55)'); grd.addColorStop(1, 'rgba(255,77,109,0)');
      g.fillStyle = grd; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
