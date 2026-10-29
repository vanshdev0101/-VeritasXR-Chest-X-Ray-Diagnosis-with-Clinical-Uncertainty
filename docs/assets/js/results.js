/* Renders docs/data/summary.json, produced by scripts/export_web_data.py.
 *
 * Two rules the chart guidance is emphatic about and that this page follows:
 * series are distinguished by dash pattern and point shape as well as hue, and
 * every chart carries a table fallback so the data is reachable without
 * perceiving colour at all.
 */
const $ = (id) => document.getElementById(id);
const DASH = [[], [6, 3], [2, 3], [9, 3, 2, 3], [1, 3]];
const POINT = ['circle', 'triangle', 'rect', 'rectRot', 'cross'];

(function theme() {
  const saved = (() => { try { return localStorage.getItem('vxr-theme'); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $('theme').addEventListener('click', () => {
    const dark = matchMedia('(prefers-color-scheme: dark)').matches;
    const cur = document.documentElement.dataset.theme || (dark ? 'dark' : 'light');
    document.documentElement.dataset.theme = cur === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('vxr-theme', document.documentElement.dataset.theme); } catch {}
    location.reload();   // simplest way to re-theme the canvases
  });
})();

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const f4 = (v) => v.toFixed(4);

function table(el, head, rows, caption) {
  el.innerHTML =
    (caption ? `<caption>${caption}</caption>` : '') +
    `<thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${
      (r.cells || r).map((c, i) =>
        i === 0 ? `<th scope="row">${c}</th>` : `<td class="num">${c}</td>`).join('')
    }</tr>`).join('')}</tbody>`;
}

function baseOpts(xTitle, yTitle, extra = {}) {
  const grid = css('--color-border'), text = css('--color-muted-foreground');
  return {
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: { labels: { color: text, usePointStyle: true, boxWidth: 10, font: { size: 11 } } },
      tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${Number(c.parsed.y).toFixed(4)}` } },
    },
    scales: {
      x: { type: 'linear', title: { display: true, text: xTitle, color: text },
           ticks: { color: text }, grid: { color: grid } },
      y: { title: { display: true, text: yTitle, color: text },
           ticks: { color: text }, grid: { color: grid } },
    },
    ...extra,
  };
}

function series(models, key, i) {
  const colors = [css('--series-1'), css('--series-2'), css('--series-3'),
                  css('--series-4'), css('--series-5')];
  const m = models[i];
  return {
    label: m.label,
    data: (m[key] || []).map(([x, y]) => ({ x, y })),
    borderColor: colors[i % 5],
    backgroundColor: colors[i % 5],
    borderDash: DASH[i % DASH.length],
    pointStyle: POINT[i % POINT.length],
    borderWidth: m.id === 'veritasxr' ? 3 : 2,
    pointRadius: 0, pointHitRadius: 8, tension: 0,
  };
}

fetch('data/summary.json').then((r) => {
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}).then(render).catch((err) => {
  $('headline').innerHTML =
    `<div class="card"><h3>Results unavailable</h3><p class="hint">Could not load
     <code>data/summary.json</code> (${err.message}). Regenerate it with
     <code>python scripts/export_web_data.py</code>.</p></div>`;
});

function render(d) {
  const models = d.models;
  const ours = models.find((m) => m.id === 'veritasxr');
  const best = models[0];
  const floor = models[models.length - 1];

  /* ---------- headline tiles ---------- */
  $('headline').innerHTML = [
    ['Best AUPRC', f4(best.metrics.auprc.mean), `${best.label} · ${best.params.toLocaleString()} params`],
    ['VeritasXR AUPRC', f4(ours.metrics.auprc.mean),
     `95% CI ${ours.ci ? `[${ours.ci.auprc[0]}, ${ours.ci.auprc[1]}]` : 'n/a'} · ${ours.params.toLocaleString()} params`],
    ['Classical control', f4(floor.metrics.auprc.mean), `${floor.label} — the floor deep learning must clear`],
  ].map(([l, v, s]) => `<div class="card stat"><span class="label">${l}</span>
      <span class="value">${v}</span><span class="sub">${s}</span></div>`).join('');

  /* ---------- prevalence note ---------- */
  const prev = d.dataset.prevalence, neg = d.baselines.always_negative_accuracy;
  const acc = ours.metrics.accuracy.mean;
  $('prevalence-note').innerHTML =
    `Positive cases are ${pct(prev)} of the test set, so a model that always
     predicts <em>negative</em> scores ${pct(neg)} accuracy while detecting nothing.
     VeritasXR's accuracy is ${pct(acc)} — only
     ${((acc - neg) * 100).toFixed(1)} points above that useless baseline — while its
     AUPRC of ${f4(ours.metrics.auprc.mean)} is ${(ours.metrics.auprc.mean / prev).toFixed(1)}×
     the no-skill floor of ${f4(prev)}. Accuracy is recorded for completeness and
     never used to select a checkpoint.`;

  /* ---------- main table ---------- */
  table($('main-table'),
    ['Model', 'Params', 'AUPRC', '95% CI', 'AUROC', 'F1', 'Bal. acc', 'Sens', 'Spec', 'Seeds', 'Train min'],
    models.map((m) => ({
      cls: m.id === 'veritasxr' ? 'is-ours' : '',
      cells: [
        `${m.label}<br><span class="ci">${m.note}</span>`,
        m.params.toLocaleString(),
        `${f4(m.metrics.auprc.mean)}${m.metrics.auprc.n > 1 ? ` <span class="ci">± ${f4(m.metrics.auprc.std)}</span>` : ''}`,
        m.ci ? `<span class="ci">[${m.ci.auprc[0]}, ${m.ci.auprc[1]}]</span>` : '—',
        f4(m.metrics.auroc.mean), f4(m.metrics.f1.mean),
        f4(m.metrics.balanced_accuracy.mean), f4(m.metrics.sensitivity.mean),
        f4(m.metrics.specificity.mean), m.seeds.length, m.train_minutes,
      ],
    })),
    `Test set: ${d.dataset.split.test.toLocaleString()} images, ${pct(prev)} positive.`);

  /* ---------- paired significance ---------- */
  const sig = models.filter((m) => m.vs_veritasxr);
  table($('sig-table'),
    ['Comparison', 'Δ AUPRC', '95% CI', 'p', 'Verdict'],
    sig.map((m) => {
      const v = m.vs_veritasxr;
      const sign = v.diff >= 0 ? '+' : '';
      return [
        `${m.label} − VeritasXR`,
        `${sign}${v.diff.toFixed(4)}`,
        `<span class="ci">[${v.ci[0]}, ${v.ci[1]}]</span>`,
        v.p < 0.0005 ? '&lt; 0.001' : v.p.toFixed(3),
        v.separated
          ? `<span class="badge">separated</span>`
          : `<span class="badge badge-pending">not separated</span>`,
      ];
    }));

  /* ---------- PR ---------- */
  new Chart($('pr-chart'), {
    type: 'line',
    data: {
      datasets: [
        ...models.map((_, i) => series(models, 'pr', i)),
        { label: `No-skill (${f4(prev)})`, data: [{ x: 0, y: prev }, { x: 1, y: prev }],
          borderColor: css('--color-muted-foreground'), borderDash: [3, 3],
          borderWidth: 1, pointRadius: 0 },
      ],
    },
    options: baseOpts('Recall', 'Precision', { scales: undefined,
      ...{ scales: { x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall', color: css('--color-muted-foreground') }, ticks: { color: css('--color-muted-foreground') }, grid: { color: css('--color-border') } },
                     y: { min: 0, max: 1, title: { display: true, text: 'Precision', color: css('--color-muted-foreground') }, ticks: { color: css('--color-muted-foreground') }, grid: { color: css('--color-border') } } } } }),
  });
  table($('pr-table'), ['Model', 'AUPRC', 'Precision @ 50% recall', 'Precision @ 80% recall'],
    models.map((m) => {
      const at = (r) => {
        const pt = (m.pr || []).filter(([rec]) => rec >= r).pop();
        return pt ? f4(pt[1]) : '—';
      };
      return [m.label, f4(m.metrics.auprc.mean), at(0.5), at(0.8)];
    }));

  /* ---------- ROC ---------- */
  new Chart($('roc-chart'), {
    type: 'line',
    data: {
      datasets: [
        ...models.map((_, i) => series(models, 'roc', i)),
        { label: 'Chance', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: css('--color-muted-foreground'), borderDash: [3, 3],
          borderWidth: 1, pointRadius: 0 },
      ],
    },
    options: baseOpts('False positive rate', 'True positive rate'),
  });
  table($('roc-table'), ['Model', 'AUROC', 'TPR @ 10% FPR', 'TPR @ 20% FPR'],
    models.map((m) => {
      const at = (f) => {
        const pt = (m.roc || []).filter(([fpr]) => fpr <= f).pop();
        return pt ? f4(pt[1]) : '—';
      };
      return [m.label, f4(m.metrics.auroc.mean), at(0.1), at(0.2)];
    }));

  /* ---------- training curves ---------- */
  const deep = models.filter((m) => m.history && m.history.length);
  new Chart($('curve-chart'), {
    type: 'line',
    data: {
      datasets: deep.map((m, i) => ({
        label: m.label,
        data: m.history.map((h) => ({ x: h.epoch, y: h.val_auprc })),
        borderColor: [css('--series-1'), css('--series-2'), css('--series-3'), css('--series-4')][i % 4],
        borderDash: DASH[i % DASH.length],
        pointStyle: POINT[i % POINT.length],
        borderWidth: m.id === 'veritasxr' ? 3 : 2,
        pointRadius: 2, tension: .15,
      })),
    },
    options: baseOpts('Epoch', 'Validation AUPRC'),
  });
  table($('curve-table'), ['Model', 'Epochs run', 'Best val AUPRC', 'Final val AUPRC'],
    deep.map((m) => [m.label, m.history.length,
      f4(Math.max(...m.history.map((h) => h.val_auprc))),
      f4(m.history[m.history.length - 1].val_auprc)]));

  /* ---------- risk-coverage ---------- */
  const rc = models.filter((m) => m.risk_coverage);
  new Chart($('rc-chart'), {
    type: 'line',
    data: {
      datasets: rc.map((m, i) => ({
        label: m.label,
        data: m.risk_coverage.map(([c, a]) => ({ x: c, y: a })),
        borderColor: [css('--series-1'), css('--series-2'), css('--series-3'), css('--series-4'), css('--series-5')][i % 5],
        borderDash: DASH[i % DASH.length],
        pointStyle: POINT[i % POINT.length],
        borderWidth: m.id === 'veritasxr' ? 3 : 2,
        pointRadius: 2, tension: .15,
      })),
    },
    options: baseOpts('Coverage — fraction of cases retained', 'Accuracy on retained cases'),
  });
  table($('rc-table'), ['Model', 'Acc @ 25% coverage', 'Acc @ 50%', 'Acc @ 100%'],
    rc.map((m) => {
      const at = (c) => {
        const pt = m.risk_coverage.find(([cov]) => Math.abs(cov - c) < 0.03);
        return pt ? f4(pt[1]) : '—';
      };
      return [m.label, at(0.25), at(0.5), at(1.0)];
    }));

  /* ---------- ablation ---------- */
  if (d.ablations && d.ablations.length) {
    table($('abl-table'), ['Variant', 'Params', 'AUPRC', 'AUROC', 'F1', 'Δ AUPRC vs full'],
      (() => {
        const full = d.ablations.find((a) => a.ablation === 'full');
        const ref = full ? full.metrics.auprc.mean : null;
        return d.ablations.map((a) => ({
          cls: a.ablation === 'full' ? 'is-ours' : '',
          cells: [a.ablation, a.params.toLocaleString(), f4(a.metrics.auprc.mean),
            f4(a.metrics.auroc.mean), f4(a.metrics.f1.mean),
            ref === null ? '—' : (a.metrics.auprc.mean - ref >= 0 ? '+' : '') + (a.metrics.auprc.mean - ref).toFixed(4)],
        }));
      })());
    $('abl-pending').textContent = '';
  } else {
    $('abl-table').innerHTML = '';
    $('abl-pending').innerHTML =
      'Ablation runs are still training. This table populates automatically once ' +
      '<code>results/veritasxr_&lt;variant&gt;_seed0/</code> exists and the export is re-run.';
  }

  /* ---------- dataset note ---------- */
  $('dataset-note').innerHTML =
    `<strong>${d.dataset.name}</strong> — ${d.dataset.images.toLocaleString()} labelled
     radiographs of ${d.dataset.population}. Labels from ${d.dataset.labels}.
     Split by patient into ${d.dataset.split.train.toLocaleString()} train /
     ${d.dataset.split.val.toLocaleString()} validation /
     ${d.dataset.split.test.toLocaleString()} test, stratified on the three-class
     label so the hard "abnormal but not pneumonia" group stays proportional. A
     test asserts no patient appears in more than one split.`;

  document.querySelectorAll('section').forEach((s) => s.classList.add('reveal'));
  requestAnimationFrame(() => document.querySelectorAll('.reveal').forEach((s) => s.classList.add('in')));
}
