// SYNTHETIC demo data (not lab recordings): seeded 2D random walks of three kinds, sampled every frame with missed
// detections, plus unrelated background localisations. Written as ThunderSTORM-style rows (frame from 1, nm).
import { rng } from './core.js';

export const KINDS = ['mobile', 'confined', 'immobile'];
export const KIND_COLOR = { mobile: '#4fd1ff', confined: '#ffb347', immobile: '#ff4d6d', background: '#6b7896' };

export function synthesize({ seed = 7, perKind = 10, frames = 80, size = 64, pDetect = 0.88, background = 40,
  stepMobile = 0.45, pixelSize = 108 } = {}) {
  const r = rng(seed);
  const gauss = () => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const raw = [];
  let particle = 0;
  for (let k = 0; k < perKind * 3; k++, particle++) {
    const kind = KINDS[k % 3];
    const x0 = 4 + r() * (size - 8), y0 = 4 + r() * (size - 8);
    const start = 1 + Math.floor(r() * frames * 0.4), life = 15 + Math.floor(r() * frames * 0.5);
    let x = x0, y = y0;
    for (let f = start; f <= Math.min(frames, start + life); f++) {
      if (kind === 'mobile') { x += gauss() * stepMobile; y += gauss() * stepMobile; }
      else if (kind === 'confined') { x = x0 + 0.6 * (x - x0) + gauss() * 0.3; y = y0 + 0.6 * (y - y0) + gauss() * 0.3; }
      else { x = x0 + gauss() * 0.06; y = y0 + gauss() * 0.06; }
      if (r() < pDetect && x >= 0 && y >= 0 && x < size && y < size) {
        raw.push({ frame: f, x, y, intensity: (kind === 'immobile' ? 400 : 250) + gauss() * 30, kind, particle });
      }
    }
  }
  for (let i = 0; i < background; i++) {
    raw.push({ frame: 1 + Math.floor(r() * frames), x: r() * size, y: r() * size, intensity: 150 + gauss() * 20, kind: 'background', particle: -1 });
  }
  raw.sort((a, b) => a.frame - b.frame);  // stable: ThunderSTORM writes frame by frame
  const locs = raw.map((p, i) => ({ frame: p.frame, id: i + 1, xnm: p.x * pixelSize, ynm: p.y * pixelSize, intensity: p.intensity }));
  return { locs, truth: raw.map(p => ({ kind: p.kind, particle: p.particle })), size, frames, pixelSize };
}
