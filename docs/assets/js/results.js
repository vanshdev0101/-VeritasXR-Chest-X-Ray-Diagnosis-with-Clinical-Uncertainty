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
    ['Comparison', 'Δ AUPRC', '95% CI', 'Worst-seed p (Holm)', 'Verdict'],
    sig.map((m) => {
      const v = m.vs_veritasxr;
      const sign = v.diff >= 0 ? '+' : '';
      const n = v.seeds || 1;
      return [
        `${m.label} − VeritasXR`,
        `${sign}${v.diff.toFixed(4)}`,
        `<span class="ci">[${v.ci[0]}, ${v.ci[1]}]</span>`,
        (v.p_holm ?? v.p) < 0.0005 ? '&lt; 0.001' : (v.p_holm ?? v.p).toFixed(3),
        (v.separated_holm ?? v.separated)
          ? `<span class="badge">separated on ${n}/${n} seeds</span>`
          : `<span class="badge badge-pending">not separated</span>`,
      ];
    }));
  const worst = sig.length ? sig.reduce((a, b) => (a.vs_veritasxr.p > b.vs_veritasxr.p ? a : b)) : null;
  if (worst && $('sig-note')) {
    $('sig-note').innerHTML =
      `Run independently on each seed, not once on seed 0 — the earlier single-seed version
       of this table did not reconcile with the mean AUPRC column above, which is what gave
       it away. Δ is the mean over seeds and the p column reports the <em>weakest</em> seed.
       The weakest of all ${sig.length * (sig[0].vs_veritasxr.seeds || 1)} tests is
       ${worst.label} at p = ${(worst.vs_veritasxr.p_holm ?? worst.vs_veritasxr.p).toFixed(3)};
       the rest are below 0.0001. p values are Holm-corrected over the whole family of
       ${worst.vs_veritasxr.family_size || sig.length} tests, and every pair still
       separates on every seed after that correction.`;
  }

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

  /* ---------- operating point ---------- */
  fetch('data/thresholds.json').then((r) => r.ok ? r.json() : null).then((T) => {
    if (!T) return;
    const nm = { efficientnet_b0: 'EfficientNet-B0', densenet121: 'DenseNet121',
                 resnet50: 'ResNet50', veritasxr: 'VeritasXR' };
    const byTuned = Object.keys(T).sort(
      (a, b) => T[b].agg.f1_at_tuned.mean - T[a].agg.f1_at_tuned.mean);
    const byHalf = Object.keys(T).sort(
      (a, b) => T[b].agg.f1_at_half.mean - T[a].agg.f1_at_half.mean);
    table($('thr-table'),
      ['Model', 'Tuned t', 'F1 @ 0.5', 'F1 @ tuned', 'Δ'],
      byTuned.map((k) => {
        const a = T[k].agg, d = a.f1_at_tuned.mean - a.f1_at_half.mean;
        return {
          cls: k === 'veritasxr' ? 'is-ours' : '',
          cells: [nm[k] || k, T[k].threshold_mean.toFixed(2),
                  f4(a.f1_at_half.mean), f4(a.f1_at_tuned.mean),
                  (d >= 0 ? '+' : '') + d.toFixed(4)],
        };
      }));
    const changed = byHalf.join() !== byTuned.join();
    $('thr-note').innerHTML =
      `The F1 ranking ${changed ? '<strong>changes</strong>' : 'does not change'} when the
       operating point is tuned: ${byHalf.map((k) => nm[k] || k).join(' &gt; ')} at 0.5
       becomes ${byTuned.map((k) => nm[k] || k).join(' &gt; ')} tuned. The first ordering
       contradicts AUPRC and AUROC, which are threshold-free and both rank
       ${nm[byTuned[0]]} first; the second agrees with them. ResNet50's own optimum is
       ${T.resnet50 ? T.resnet50.threshold_mean.toFixed(2) : '—'}, essentially the default,
       so it was already being read at its best point while the others were not. VeritasXR
       is last under either convention, so this correction does not flatter it.`;
  });

  /* ---------- calibration ---------- */
  fetch('data/calibration.json').then((r) => r.ok ? r.json() : null).then((C) => {
    if (!C) return;
    const order = Object.keys(C).sort((a, b) => C[a].ece - C[b].ece);
    const name = { veritasxr: 'VeritasXR', efficientnet_b0: 'EfficientNet-B0',
                   resnet50: 'ResNet50', densenet121: 'DenseNet121',
                   hog_svm: 'HOG + linear SVM' };
    table($('cal-table'),
      ['Model', 'ECE', 'MCE', 'Brier', 'Mean confidence', 'Accuracy', 'Gap'],
      order.map((k) => {
        const v = C[k], gap = v.mean_confidence - v.accuracy;
        return {
          cls: k === 'veritasxr' ? 'is-ours' : '',
          cells: [name[k] || k, f4(v.ece), f4(v.mce), f4(v.brier),
                  f4(v.mean_confidence), f4(v.accuracy),
                  (gap >= 0 ? '+' : '') + gap.toFixed(4)],
        };
      }));
    const vx = C.veritasxr;
    if (vx) {
      const ps = vx.ece_per_seed || [vx.ece];
      const rank = order.indexOf('veritasxr') + 1;
      $('cal-note').innerHTML =
        `This table was single-seed until an audit caught it, and it read as a 12&times;
         win for VeritasXR. Across three seeds its ECE is
         ${ps.map((x) => x.toFixed(4)).join(', ')} — seed 0 was its best draw by a wide
         margin. Averaged, it ranks ${rank} of ${order.length} with the largest seed
         spread of the four deep models (&plusmn; ${f4(vx.ece_std || 0)}), so it is the
         least predictable of them on the metric it was built to own. Every model here is
         overconfident: the gap column is positive for all five. ResNet50 still has the
         best Brier score, which combines calibration with refinement — different metrics
         favour different models. The HOG + SVM figure is an artifact of squashing an
         unbounded SVM margin through a logistic, not a property of the classifier.`;
    }
  });

  /* ---------- temperature scaling ---------- */
  fetch('data/temperature.json').then((r) => r.ok ? r.json() : null).then((T) => {
    const sect = $('temperature');
    if (!T || !Object.keys(T).length) { if (sect) sect.hidden = true; return; }
    const name = { veritasxr: 'VeritasXR', efficientnet_b0: 'EfficientNet-B0',
                   resnet50: 'ResNet50', densenet121: 'DenseNet121' };
    const order = Object.keys(T).sort((a, b) => T[a].agg.after.ece - T[b].agg.after.ece);
    table($('temp-table'),
      ['Model', 'Fitted T', 'ECE before', 'ECE after', 'Brier before', 'Brier after'],
      order.map((k) => {
        const a = T[k].agg;
        return {
          cls: k === 'veritasxr' ? 'is-ours' : '',
          cells: [name[k] || k, a.temperature.toFixed(3), f4(a.before.ece),
                  f4(a.after.ece), f4(a.before.brier), f4(a.after.brier)],
        };
      }));
    const last = order[order.length - 1];
    $('temp-note').innerHTML =
      `Ranking is invariant under this transform — AUPRC and AUROC are identical before
       and after, which the script asserts — so the only thing that moves is how
       believable the numbers are. ${name[order[0]]} ends up best calibrated of the four,
       and <strong>${name[last]} ends up last</strong>. ResNet50's miscalibration is
       almost entirely a scale error, which one scalar corrects; what is left of
       VeritasXR's after scaling is not, so scaling cannot remove it. Calibration out of
       the box is still worth having, but it is not an advantage any of these models can
       be denied.`;
  });

  /* ---------- localisation ---------- */
  fetch('data/localisation.json').then((r) => r.ok ? r.json() : null).then((L) => {
    if (!L) return;
    const v = L.veritasxr, chance = L._chance.random_point_in_box;
    const rows = [
      ['EfficientNet-B0', L.efficientnet_b0 && L.efficientnet_b0.point,
       L.efficientnet_b0 && L.efficientnet_b0.mass, L.efficientnet_b0 && L.efficientnet_b0.iou],
      ['ResNet50', L.resnet50 && L.resnet50.point, L.resnet50 && L.resnet50.mass,
       L.resnet50 && L.resnet50.iou],
      ['VeritasXR — local 3x3', v.local_point, v.local_mass, v.local_iou],
      ['VeritasXR — merged', v.merged_point, v.merged_mass, v.merged_iou],
      ['DenseNet121', L.densenet121 && L.densenet121.point,
       L.densenet121 && L.densenet121.mass, L.densenet121 && L.densenet121.iou],
      ['VeritasXR — global 7x7', v.global_point, v.global_mass, v.global_iou],
    ].filter((r) => r[1]).sort((a, b) => b[1].mean - a[1].mean);

    table($('loc-table'), ['Model / pathway', 'Pointing game', 'Mass in box', 'IoU @ 0.5'],
      rows.map((r) => ({
        cls: r[0].startsWith('VeritasXR') ? 'is-ours' : '',
        cells: [r[0], f4(r[1].mean), f4(r[2].mean), f4(r[3].mean)],
      })).concat([{ cls: '', cells: [
        '<em>chance — random point</em>', `<em>${f4(chance)}</em>`,
        `<em class="ci">boxes cover ${pct(L._chance.mean_box_area_fraction)}</em>`, '—'] }]));

    const rs = L.resnet50 && L.resnet50.point;
    const sd = (m) => m && m.seed_std !== undefined ? ` ±${f4(m.seed_std)}` : '';
    $('loc-note').innerHTML =
      `The metrics disagree about the two pathways, and that is the result. The local 3x3
       branch places its peak inside a box more often (${f4(v.local_point.mean)}${sd(v.local_point)}
       against ${f4(v.global_point.mean)}${sd(v.global_point)}), while the global 7x7 branch
       puts more activation mass inside the boxes (${f4(v.global_mass.mean)} against
       ${f4(v.local_mass.mean)}) and overlaps them better. Neither dominates. Their peaks
       coincide on just ${pct(v.pathway_peak_agreement.mean)} of images, so the branches are
       genuinely not duplicates.` +
      (rs && rs.per_seed ? ` Note also that ResNet50's localisation swings from
       ${f4(Math.min(...rs.per_seed))} to ${f4(Math.max(...rs.per_seed))} on seed alone
       (±${f4(rs.seed_std)}), so a single-seed Grad-CAM figure from it would describe the
       seed rather than the architecture.` : '');
  });

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
    $('abl-pending').innerHTML =
      'Significance is a paired bootstrap per seed, not a rule of thumb over seed means. ' +
      'Only <code>local_only</code> — the variant keeping just the 3&times;3 branch — is worse ' +
      'than the full model on all three seeds, so removing the 7&times;7 global pathway ' +
      'reliably costs accuracy. Every other variant changes sign depending on the seed, ' +
      'including <code>global_only</code>, which separates on seed 0 and reverses on seed 1. ' +
      'None of those components is doing measurable work here, and each would have looked ' +
      'meaningful from a single run.';
  } else {
    $('abl-table').innerHTML = '';
    $('abl-pending').innerHTML =
      'Ablation runs are still training. This table populates automatically once ' +
      '<code>results/veritasxr_&lt;variant&gt;_seed0/</code> exists and the export is re-run.';
  }

  /* ---------- learning-rate probe ---------- */
  const LP = d.lr_probe;
  if (LP && LP.models && LP.models.length) {
    const rates = [...new Set(LP.models.flatMap((m) => m.points.map((p) => p[0])))]
      .sort((a, b) => a - b);
    const fmt = (x) => x.toExponential(0).replace('e-', 'e&minus;');
    table($('lr-table'),
      ['Model', ...rates.map((r) => fmt(r) + (r === LP.shared_lr ? ' (shared)' : '')),
       'Best &minus; shared'],
      LP.models.map((m) => {
        const at = new Map(m.points);
        return {
          cls: m.model === 'veritasxr' ? 'is-ours' : '',
          cells: [m.label,
            ...rates.map((r) => {
              const v = at.get(r);
              if (v === undefined) return '—';
              return r === m.best_lr ? `<strong>${f4(v)}</strong>` : f4(v);
            }),
            (m.gain >= 0 ? '+' : '') + m.gain.toFixed(4)],
        };
      }));
    const vx = LP.models.find((m) => m.model === 'veritasxr');
    const top = LP.models[0];
    const gapLo = (() => {
      const r = (d.models || []).filter((m) => m.vs_veritasxr)
        .map((m) => m.vs_veritasxr.diff)
        .filter((x) => typeof x === 'number' && x > 0);
      return r.length ? Math.min(...r) : null;
    })();
    $('lr-note').innerHTML =
      `Seed 0, the same grid for every model, AUPRC on the test set. The confound is real
       for one model and it points the wrong way: ${top.label} was genuinely handicapped by
       the shared rate and gains <strong>${top.gain.toFixed(4)}</strong> at its own${
         vx ? `, while VeritasXR gains ${vx.gain.toFixed(4)}` : ''}${
         gapLo !== null ? ` against a gap to the nearest baseline of ${gapLo.toFixed(4)}`
                        : ''}. Tuning each model individually therefore <em>widens</em> the
       gap to the custom model rather than closing it, because the baselines have more to
       gain from it than it does. ResNet50 at 1e&minus;3 is the other half of the picture —
       it collapses to 0.6257. One shared rate sits at no model's optimum, but it sits
       inside the stable region for all four, and that is the honest reason to keep it.
       This is a confound that was tested, not one assumed away.`;
  } else if ($('lr-section')) {
    $('lr-section').hidden = true;
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
