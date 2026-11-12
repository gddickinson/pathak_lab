// About: what the repo is, how to run the desktop tools and the web app, data notes and citation (from the repo).
import { api, app, esc, route } from './core.js';

route('about', async () => {
  const info = await api('/info');
  app().innerHTML = `<p class="kicker">About</p><h1>About these tools</h1>
  <p class="lead">This repository holds Python code created for the Pathak lab (UC Irvine) to analyse TIRF recordings of
    PIEZO1-HaloTag using <a href="https://flika-org.github.io/" target="_blank" rel="noopener">FLIKA</a>. The work is a
    group effort; this site describes what the tools do.</p>
  <h2>What is in the repository</h2>
  <ul>
    <li><b>flika_plugins/</b>: ${info.plugins.length} FLIKA plugins (${info.plugins.map(p => esc(p.name)).join(', ')}).</li>
    <li><b>flika_scripts/piezo1_analysis/</b>: preprocessing, the numbered analysis steps, postprocessing and
      development scripts.</li>
    <li><b>minimal_model/</b>: an example recording with its ThunderSTORM settings, ROI file and results table.</li>
  </ul>
  <p>Localisation is done beforehand with ThunderSTORM (ImageJ/FIJI; Step 1 writes the batch macro) or FLIKA's pynsight
    plugin; the scripts then link, measure and classify the localisations, and the plugins display them.</p>
  <h2>Run the desktop tools</h2>
  <p>Install FLIKA, copy the folders in <code>flika_plugins/</code> to <code>~/.FLIKA/plugins/</code> and restart FLIKA.
    The pipeline scripts are run one by one after setting the data <code>path</code> in their
    <code>__main__</code> block (see each tool's page). <code>pip install -r requirements.text</code> installs the
    Python packages.</p>
  <h2>Run this web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8202</pre>
  <p>The local app runs the repo's own Python on your files (Run page), including the SVM classification with the
    training file kept in the repository. The static site runs the JS ports in your browser instead.</p>
  <h2>Data in the repository</h2>
  <p>The example recording and results table in <code>minimal_model/</code> and the SVM training-features file in
    <code>flika_plugins/locsAndTracksPlotter/</code> are lab data. They are used by the local app only and are not
    copied to this static site; every demo here uses synthetic data.</p>
  <h2>Citation</h2>
  <p>The analysis is described in: ${esc(info.paper.title)}. <i>${esc(info.paper.journal)}</i>.
    <a href="https://doi.org/${esc(info.paper.doi)}" target="_blank" rel="noopener">doi:${esc(info.paper.doi)}</a>
    (<code>CITATION.cff</code> in the repository has the full reference). Licence: MIT.</p>`;
});
