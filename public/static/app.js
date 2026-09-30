/* TomatoLeafAI v21 web client -- no framework, no external requests. */
(() => {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const CFG = JSON.parse($('#cfg').textContent);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (x, d = 0) => (x == null || isNaN(x) ? '–' : (100 * x).toFixed(d) + '%');
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } },
  };
  const MODEL_LABEL = Object.fromEntries(CFG.models.map((m) => [m.name, m.label]));

  /* ---------------------------------------------------------------- theme */
  const applyTheme = (t) => { if (t) document.documentElement.dataset.theme = t; };
  applyTheme(store.get('tla_theme'));
  $('#theme').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme === 'dark' ||
      (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    const t = dark ? 'light' : 'dark'; applyTheme(t); store.set('tla_theme', t);
  });

  /* ----------------------------------------------------------------- tabs */
  const loaded = {};
  function showTab(name) {
    $$('.tab').forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle('active', on); b.setAttribute('aria-selected', on); });
    $$('.panel').forEach((p) => { const on = p.id === 'tab-' + name; p.hidden = !on; p.classList.toggle('active', on); });
    if (!loaded[name]) { loaded[name] = true; ({ history: renderHistory, models: loadPerformance, diseases: loadDiseases }[name] || (() => {}))(); }
    if (name === 'history') renderHistory();
  }
  $$('.tab').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

  /* ------------------------------------------------------------ lightbox */
  const lb = $('#lightbox');
  document.addEventListener('click', (e) => {
    const img = e.target.closest('[data-zoom]');
    if (img) { $('#lb-img').src = img.dataset.zoom || img.src; $('#lb-cap').textContent = img.dataset.cap || ''; lb.showModal(); }
    if (e.target.matches('[data-close]')) e.target.closest('dialog').close();
  });

  /* --------------------------------------------------------------- input */
  let photo = null, photoThumb = null;
  const fileInput = $('#file'), drop = $('#drop');
  async function setPhoto(file) {
    if (!file || !file.type.startsWith('image/')) return toast('Please choose an image file.');
    try {
      const { blob, thumb, w, h } = await shrink(file);
      photo = new File([blob], (file.name || 'photo').replace(/\.\w+$/, '') + '.jpg', { type: 'image/jpeg' });
      photoThumb = thumb;
      const url = URL.createObjectURL(blob);
      $('#preview').src = url; $('#preview').hidden = false; $('#drop-empty').hidden = true; $('#clear-btn').hidden = false;
      $('#file-info').hidden = false;
      $('#file-info').textContent = `${w}×${h}px · ${(blob.size / 1024).toFixed(0)} KB (resized on your device before upload)`;
    } catch (err) { toast('Could not read that image: ' + err.message); }
  }
  async function shrink(file, maxSide = 1280) {
    let bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }); }
    catch { bmp = await new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = URL.createObjectURL(file); }); }
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
    const c = document.createElement('canvas'); c.width = w; c.height = h; c.getContext('2d').drawImage(bmp, 0, 0, w, h);
    let q = 0.9, blob;
    do { blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', q)); q -= 0.15; } while (blob.size > 3.5e6 && q > 0.3);
    const t = document.createElement('canvas'); const ts = 96 / Math.max(w, h); t.width = Math.round(w * ts); t.height = Math.round(h * ts);
    t.getContext('2d').drawImage(c, 0, 0, t.width, t.height);
    return { blob, thumb: t.toDataURL('image/jpeg', 0.7), w, h };
  }
  fileInput.addEventListener('change', () => setPhoto(fileInput.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => setPhoto(e.dataTransfer.files[0]));
  document.addEventListener('paste', (e) => { const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/')); if (f) setPhoto(f); });
  $('#clear-btn').addEventListener('click', () => {
    photo = null; fileInput.value = ''; $('#preview').hidden = true; $('#drop-empty').hidden = false; $('#clear-btn').hidden = true; $('#file-info').hidden = true;
  });

  /* camera: live preview when available, otherwise the phone's camera picker */
  const camDlg = $('#cam-dialog'); let stream = null;
  $('#camera-btn').addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) return $('#camera-file').click();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 } } });
      $('#video').srcObject = stream; $('#cam-err').hidden = true; camDlg.showModal();
    } catch { $('#camera-file').click(); }
  });
  $('#camera-file').addEventListener('change', (e) => setPhoto(e.target.files[0]));
  camDlg.addEventListener('close', () => { stream?.getTracks().forEach((t) => t.stop()); stream = null; });
  $('#snap').addEventListener('click', () => {
    const v = $('#video'); const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob((b) => { setPhoto(new File([b], 'camera.jpg', { type: 'image/jpeg' })); camDlg.close(); }, 'image/jpeg', 0.92);
  });

  /* model select */
  const modeRadios = $$('input[name="mode"]');
  const describe = () => {
    const m = CFG.models.find((x) => x.name === $('#model').value);
    $('#model-desc').textContent = m ? m.desc : '';
    const compare = modeRadios.find((r) => r.checked)?.value === 'compare';
    $('#model-field').hidden = compare; $('#run').textContent = compare ? 'Compare all models' : 'Diagnose';
  };
  $('#model').addEventListener('change', describe); modeRadios.forEach((r) => r.addEventListener('change', describe)); describe();

  /* ------------------------------------------------------------- analyse */
  const STAGES = { decode: 'Reading photo', segment: 'Background removal & photo check', leafcheck: 'Tomato-leaf check',
    predict: 'Diagnosis + Grad-CAM', advice: 'Treatment advice' };
  $('#form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!photo) return toast('Choose a leaf photo first.');
    const mode = modeRadios.find((r) => r.checked).value;
    const fd = new FormData(); fd.append('image', photo); fd.append('mode', mode); fd.append('model', $('#model').value);
    $('#run').disabled = true; $('#empty').hidden = true; $('#result').innerHTML = '';
    $('#progress').hidden = false; $('#bar').style.width = '2%';
    $('#stages').innerHTML = Object.entries(STAGES).map(([k, v]) => `<li data-s="${k}">${v}</li>`).join('');
    let result = null;
    try {
      const resp = await fetch('/api/analyze', { method: 'POST', body: fd });
      if (!resp.ok || !resp.body) { const j = await resp.json().catch(() => ({})); throw new Error(j.error || `Server error ${resp.status}`); }
      const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf = '';
      for (;;) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i; while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === 'progress') progress(ev);
          else if (ev.type === 'partial') progressDetail(`${MODEL_LABEL[ev.model.model] || ev.model.model}: ${ev.model.display_name} (${pct(ev.model.confidence)})`);
          else if (ev.type === 'result') result = ev.result;
          else if (ev.type === 'error') throw new Error(ev.error);
        }
      }
      if (!result) throw new Error('No result received (the server may have timed out).');
      $('#bar').style.width = '100%'; $$('#stages li').forEach((li) => li.classList.add('done'));
      renderResult(result);
      saveHistory(result);
    } catch (err) {
      $('#result').innerHTML = `<div class="verdict bad"><h2>Something went wrong</h2><p>${esc(err.message)}</p></div>`;
    } finally {
      $('#run').disabled = false; setTimeout(() => { $('#progress').hidden = true; }, 600);
    }
  });
  function progress(ev) {
    $('#bar').style.width = ev.pct + '%';
    let passed = true;
    $$('#stages li').forEach((li) => {
      if (li.dataset.s === ev.stage) { passed = false; li.className = 'now'; li.textContent = ev.detail || STAGES[ev.stage]; }
      else if (passed) li.className = 'done'; else li.className = '';
    });
  }
  function progressDetail(t) { const li = $('#stages li.now'); if (li) li.textContent = t; }

  /* -------------------------------------------------------------- render */
  const fig = (src, cap) => src ? `<figure><img src="${src}" data-zoom="${src}" data-cap="${esc(cap)}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>` : '';
  const list = (arr) => arr && arr.length ? `<ul>${arr.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '';
  const CATEGORY = { chemical_may_be_needed: 'Chemical control may be needed', biological_available: 'Biological / cultural control first',
    vector_control_for_viral: 'Viral — control the insect vector, remove infected plants', none_healthy: 'No treatment needed' };
  const ADVISORY = { high_confidence_guidance: 'High-confidence guidance', moderate_confidence_review_recommended: 'Moderate confidence — review recommended',
    low_confidence_seek_expert_opinion: 'Low confidence — seek expert opinion' };

  function renderResult(r) {
    const out = $('#result');
    if (!r.accepted) {
      out.innerHTML = `<div class="verdict bad"><h2>${r.stage === 'setup' ? 'Setup needed' : 'Not accepted'}</h2><p>${esc(r.message)}</p>
        ${r.reasons?.length ? `<details class="acc"><summary>Why?</summary><div>${list(r.reasons)}</div></details>` : ''}</div>
        ${r.preview ? `<div class="imgs">${fig(r.preview.leaf_mask, 'What the app found (green = leaf)')}${fig(r.preview.background_removed, 'After background removal')}</div>` : ''}`;
      return;
    }
    const p = r.prediction;
    const conf = p.confidence;
    const cls = p.confident ? 'ok' : 'unsure';
    const f = p.focus || {};
    let html = `
      <div class="verdict ${p.confident ? cls : 'unsure'}">
        <div class="badges"><span class="badge">${esc(MODEL_LABEL[p.model] || p.model)}</span>${p.calibrated ? '<span class="badge">calibrated</span>' : ''}
          ${r.compare && r.consensus ? `<span class="badge">${r.consensus.agree}/${r.consensus.total} models agree</span>` : ''}</div>
        <h2>${p.healthy ? '🌿 ' : ''}${esc(p.display_name)}</h2>
        <div class="meter"><span style="width:${(conf * 100).toFixed(1)}%"></span></div>
        <div class="sub">Confidence ${pct(conf, 1)} · ${esc(r.message)}</div>
      </div>
      <div class="imgs">
        ${fig(r.preview.background_removed, 'Leaf after background removal')}
        ${fig(p.images.gradcam, 'Grad-CAM: where the model looked')}
        ${p.healthy ? '' : fig(p.images.disease_regions, 'Disease regions (CAM × lesion colour)')}
        ${fig(p.images.lesion_attention, 'Hybrid lesion attention (steers the ViT)')}
        ${fig(r.preview.leaf_mask, 'Segmentation (green leaf, red lesions)')}
      </div>
      <div class="kpis">
        <div class="kpi"><b>${pct(f.lfs)}</b><small>Leaf-Focus Score (attention on the leaf)</small></div>
        ${!p.healthy && f.lefs != null ? `<div class="kpi"><b>${pct(f.lefs)}</b><small>Attention on lesion tissue</small></div>` : ''}
        ${r.affected_area_estimate != null ? `<div class="kpi"><b>${pct(r.affected_area_estimate)}</b><small>Leaf area with lesion colour (estimate)</small></div>` : ''}
        <div class="kpi"><b>${(r.timing_ms / 1000).toFixed(1)} s</b><small>Processing time</small></div>
      </div>
      <h3 class="sec">Other possibilities</h3>
      <ul class="tops">${p.top.map((t) => `<li><span>${esc(t.display_name)}</span><span class="t"><span style="width:${(t.prob * 100).toFixed(1)}%"></span></span><span>${pct(t.prob)}</span></li>`).join('')}</ul>`;
    if (r.compare) html += compareTable(r);
    html += treatmentBlock(r.treatment, p);
    out.innerHTML = html;
  }

  function compareTable(r) {
    return `<h3 class="sec">All models</h3><div class="table-wrap"><table><thead><tr><th>Model</th><th>Prediction</th><th>Confidence</th><th>Leaf-Focus</th><th>Grad-CAM</th><th>Time</th></tr></thead><tbody>
      ${r.results.map((m) => `<tr class="${m.model === r.prediction.model ? 'best' : ''}"><td>${esc(MODEL_LABEL[m.model] || m.model)}</td><td>${esc(m.display_name)}</td><td>${pct(m.confidence, 1)}</td>
        <td>${pct(m.focus?.lfs)}</td><td><img class="mini" src="${m.images.gradcam}" data-zoom="${m.images.gradcam}" data-cap="${esc(m.label)} Grad-CAM" alt=""></td><td>${m.time_ms} ms</td></tr>`).join('')}
      </tbody></table></div>`;
  }

  function treatmentBlock(t, p) {
    if (!t) return '';
    const lowConf = !p.confident;
    return `<h3 class="sec">Treatment &amp; management — ${esc(t.disease_name)}</h3>
      <div class="badges"><span class="badge">${esc(CATEGORY[t.control_category] || t.control_category)}</span><span class="badge">${esc(t.pathogen_type)}</span>
        <span class="badge">${esc(ADVISORY[t.advisory_level] || t.advisory_level)}</span></div>
      ${lowConf ? '<div class="warnbox">Confidence is low. Retake the photo (one leaf, in focus, daylight) or confirm with an agronomist before acting.</div>' : ''}
      ${acc('What to look for', t.symptoms_observed, !p.healthy)}
      ${acc('Cultural practices', t.cultural_practices, true)}
      ${acc('Sanitation', t.sanitation)}
      ${acc('Irrigation & water', t.irrigation_water_management)}
      ${acc('Monitoring', t.monitoring_advice)}
      ${acc('Prevention', t.prevention_advice)}
      <p class="disclaimer">${esc(t.disclaimer)}</p>`;
  }
  const acc = (title, items, open) => items && items.length ? `<details class="acc" ${open ? 'open' : ''}><summary>${esc(title)}</summary><div>${list(items)}</div></details>` : '';

  /* ------------------------------------------------------------- history */
  function trimForHistory(r) {
    const c = JSON.parse(JSON.stringify(r));
    if (c.results) c.results.forEach((m) => { m.images = { gradcam: m.images.gradcam }; });
    if (c.prediction) { const keep = c.prediction.images; c.prediction.images = { gradcam: keep.gradcam, disease_regions: keep.disease_regions }; }
    return c;
  }
  function saveHistory(r) {
    const h = store.get('tla_history', []);
    const item = { id: Date.now(), time: new Date().toISOString(), thumb: photoThumb, file: photo?.name,
      accepted: r.accepted, name: r.accepted ? r.prediction.display_name : 'Not accepted',
      conf: r.accepted ? r.prediction.confidence : null, confident: r.accepted && !!r.prediction.confident, model: r.accepted ? r.prediction.model : '', compare: !!r.compare, result: trimForHistory(r) };
    h.unshift(item);
    while (h.length && !store.set('tla_history', h.slice(0, 25))) h.pop();
    updateHistCount();
  }
  function updateHistCount() { const n = store.get('tla_history', []).length; $('#hist-count').hidden = !n; $('#hist-count').textContent = n; }
  function renderHistory() {
    const q = ($('#hist-search').value || '').toLowerCase();
    const h = store.get('tla_history', []).filter((x) => !q || `${x.name} ${x.model} ${x.file}`.toLowerCase().includes(q));
    $('#hist-list').innerHTML = h.length ? h.map((x) => `<article class="card hitem" data-id="${x.id}">
        <img src="${x.thumb || ''}" alt=""><div><b>${esc(x.name)}</b><div class="muted small">${[esc(MODEL_LABEL[x.model] || x.model || ''), x.compare ? 'compare' : '', new Date(x.time).toLocaleString()].filter(Boolean).join(' · ')}</div></div>
        <span class="pill ${x.confident ? 'ok' : 'warn'}">${x.conf != null ? pct(x.conf) : 'rejected'}</span></article>`).join('')
      : '<p class="muted">No diagnoses yet.</p>';
  }
  $('#hist-list').addEventListener('click', (e) => {
    const it = e.target.closest('.hitem'); if (!it) return;
    const x = store.get('tla_history', []).find((y) => String(y.id) === it.dataset.id); if (!x) return;
    showTab('diagnose'); $('#empty').hidden = true; renderResult(x.result);
  });
  $('#hist-search').addEventListener('input', renderHistory);
  $('#hist-clear').addEventListener('click', () => { if (confirm('Delete all history stored in this browser?')) { store.set('tla_history', []); renderHistory(); updateHistCount(); } });
  updateHistCount();

  /* ------------------------------------------------------ models / perf */
  function csvTable(csv, maxCols = 9) {
    if (!csv) return '';
    const rows = csv.trim().split(/\r?\n/).map((l) => l.split(','));
    const head = rows.shift().slice(0, maxCols);
    return `<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>
      ${rows.map((r) => `<tr>${r.slice(0, maxCols).map((c) => { const n = Number(c); return `<td>${isNaN(n) || c === '' ? esc(c) : (Math.abs(n) <= 1 && c.includes('.') ? (n * 100).toFixed(2) + '%' : esc(c))}</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }
  async function loadPerformance() {
    const el = $('#perf'); el.innerHTML = '<p class="muted">Loading evaluation results…</p>';
    try {
      const p = await (await fetch('/api/performance')).json();
      let html = '';
      if (p.comparison_csv) html += `<h3 class="sec">Test-set comparison (from the notebook)</h3>${csvTable(p.comparison_csv)}`;
      if (p.gate?.heldout) {
        const g = p.gate.heldout;
        html += `<h3 class="sec">Tomato-leaf gate (held-out)</h3><div class="kpis"><div class="kpi"><b>${pct(g.tomato_accept_rate, 1)}</b><small>real tomato leaves accepted</small></div>
          <div class="kpi"><b>${pct(g.non_tomato_reject_rate, 1)}</b><small>non-tomato images rejected</small></div>
          ${Object.entries(g.per_category_reject_rate || {}).map(([k, v]) => `<div class="kpi"><b>${pct(v, 1)}</b><small>${esc(k)} rejected</small></div>`).join('')}</div>`;
      }
      if (p.hypotheses) html += `<h3 class="sec">Proposal hypotheses (statistical tests)</h3><div class="kpis">${['H1', 'H2', 'H3'].filter((k) => p.hypotheses[k]).map((k) => `<div class="kpi"><b>${k}</b><small>${esc(p.hypotheses[k].verdict)}</small></div>`).join('')}</div>`;
      if (p.xai_csv) html += `<h3 class="sec">Explainability (Leaf-Focus Score)</h3>${csvTable(p.xai_csv, 11)}`;
      if (p.latency_csv) html += `<h3 class="sec">Speed &amp; size</h3>${csvTable(p.latency_csv)}`;
      el.innerHTML = html || '<p class="muted">No evaluation files were exported with the models (see README: copy the notebook results when exporting).</p>';
    } catch { el.innerHTML = '<p class="muted">Could not load evaluation results.</p>'; }
  }

  /* ------------------------------------------------------------ diseases */
  async function loadDiseases() {
    const el = $('#dis-list'); el.innerHTML = '<p class="muted">Loading…</p>';
    const d = await (await fetch('/api/diseases')).json();
    el.innerHTML = d.classes.map((c) => `<article class="card dcard" data-s="${esc((c.name + ' ' + (c.disease_name || '')).toLowerCase())}">
      <span class="cat">${esc(CATEGORY[c.control_category] || '')}</span><h3>${esc(c.disease_name || c.name)}</h3>
      <div class="muted small">${esc(c.pathogen_type || '')}</div>
      ${acc('Symptoms', c.symptoms_observed, true)}${acc('Cultural practices', c.cultural_practices)}${acc('Sanitation', c.sanitation)}
      ${acc('Irrigation & water', c.irrigation_water_management)}${acc('Monitoring', c.monitoring_advice)}${acc('Prevention', c.prevention_advice)}</article>`).join('');
    $('#dis-disclaimer').textContent = d.disclaimer;
  }
  $('#dis-search').addEventListener('input', (e) => { const q = e.target.value.toLowerCase(); $$('.dcard').forEach((c) => { c.hidden = q && !c.dataset.s.includes(q); }); });

  /* --------------------------------------------------------------- toast */
  function toast(msg) { const r = $('#result'); $('#empty').hidden = true; r.innerHTML = `<div class="banner warn">${esc(msg)}</div>`; }
})();
