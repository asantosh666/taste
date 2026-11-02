let items = [];
let selected = 0;
let groupsVisible = false;
let groupsData = [];
let q1Sharp = 0;
let q3Sharp = 0;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

function visionLabel(v) {
  if (v.status === "live") return `composition ${v.composition.toFixed(2)} · moment ${v.moment.toFixed(2)}`;
  return "vision: pending";
}

function fmt(v, d) {
  return (v === null || v === undefined) ? "—" : v.toFixed(d);
}

function computeQuartiles() {
  const s = items.map((x) => x.sharpness).sort((a, b) => a - b);
  if (!s.length) { q1Sharp = 0; q3Sharp = 0; return; }
  q1Sharp = s[Math.floor(s.length * 0.25)];
  q3Sharp = s[Math.floor(s.length * 0.75)];
}

// "Contested": Muse's composition score and the deterministic features
// strongly disagree. Purely rule-based — no new model calls.
function contested(it) {
  const comp = it.vision.composition;
  if (comp === null || comp === undefined) return null;
  const subjSoft = it.vision.sam_mask_returned &&
    it.vision.subject_sharpness !== null && it.vision.subject_sharpness < 0.4;
  if (comp >= 0.8 && (it.sharpness <= q1Sharp || subjSoft)) {
    return "Muse loves the composition but the frame is soft";
  }
  if (comp <= 0.2 && it.sharpness >= q3Sharp) {
    return "Muse rates the composition low but the frame is technically strong";
  }
  return null;
}

// Reason cards from MEASURED deltas only — never free-text praise from
// the model. members = the frame's duplicate-group members (ranked dicts).
function reasonCards(it, members) {
  const cards = [];
  if (it.pairwise && it.pairwise.is_winner) cards.push("Group winner (pairwise)");
  if (!members || members.length < 2) {
    if (!members) cards.push("Standalone frame (no near-duplicates)");
    return cards;
  }
  const mx = (f) => {
    const vals = members.map(f).filter((v) => v !== null && v !== undefined);
    return vals.length ? Math.max(...vals) : null;
  };
  if (it.sharpness === mx((m) => m.sharpness)) cards.push("Sharpest in group");
  if (it.exposure_quality === mx((m) => m.exposure_quality)) cards.push("Best exposed in group");
  if (it.vision.moment !== null && it.vision.moment !== undefined &&
      it.vision.moment === mx((m) => m.vision.moment)) cards.push("Strongest moment in group");
  if (it.vision.composition !== null && it.vision.composition !== undefined &&
      it.vision.composition === mx((m) => m.vision.composition)) cards.push("Best composed in group");
  const subjMax = mx((m) => (m.vision.sam_mask_returned ? m.vision.subject_sharpness : null));
  if (it.vision.sam_mask_returned && it.vision.subject_sharpness === subjMax) cards.push("Sharpest subject in group");
  if (it.score === mx((m) => m.score)) cards.push("Top ranked in group");
  return cards;
}

function groupMembers(it) {
  if (it.group === null || it.group === undefined) return null;
  return groupsData[it.group] || null;
}

async function refresh(keepSelectionId) {
  const [st, rank, grp] = await Promise.all([api("/api/state"), api("/api/rank"), api("/api/groups")]);
  items = rank.items;
  groupsData = grp.groups;
  computeQuartiles();
  const banner = $("vision-banner");
  banner.className = "banner " + (st.vision_status === "live" ? "live" : "pending");
  banner.hidden = false;
  if (st.vision_status === "live") {
    banner.textContent = "Vision pass live — composition / moment scores are real model output.";
  } else if (st.vision_status === "pending_configuration") {
    banner.textContent = "Vision pass failed — " + (st.vision_note || "scores are neutral placeholders, not model output.");
  } else {
    banner.textContent = "Vision pass pending API key — composition / moment scores are neutral placeholders, not model output.";
  }
  $("state").textContent =
    `${st.set_name} · ${st.total} frames · ${st.keeps} kept · ${st.rejects} rejected · ${st.maybes} maybe · ` +
    `model: ${st.model_trained ? `trained on ${st.model_n_train} labels` : "untrained (neutral ranking)"} · ` +
    `weights: ${JSON.stringify(st.model_weights)} · ` +
    `${st.dupe_groups} near-duplicate groups (${st.dupe_frames} frames) · vision: ${st.vision_status}`;
  renderGrid();
  if (keepSelectionId) {
    const i = items.findIndex((x) => x.id === keepSelectionId);
    if (i >= 0) selected = i;
  }
  selected = Math.max(0, Math.min(selected, items.length - 1));
  renderDetail();
  if (!$("eval-result").hidden) renderEval();
}

function renderGrid() {
  const grid = $("grid");
  grid.innerHTML = "";
  items.forEach((it, i) => {
    const card = document.createElement("div");
    card.className = "card" + (i === selected ? " selected" : "") + (it.user_label ? " " + it.user_label : "");
    card.innerHTML =
      `<img src="/api/thumb/${it.id}" alt="" loading="lazy">` +
      `<div class="meta"><span class="score">${it.score.toFixed(2)}</span> ${it.filename}` +
      (it.group !== null && it.group !== undefined ? `<span class="dup-badge" title="Near-duplicate group">dup ${it.group + 1}</span>` : "") +
      (it.pairwise && it.pairwise.is_winner ? `<span class="label-badge winner">★ group winner</span>` : "") +
      (it.user_label ? `<span class="label-badge ${it.user_label}">${it.user_label}</span>` : "") +
      (it.calib_label ? `<span class="label-badge calib">calib ${it.calib_label}</span>` : "") +
      (contested(it) ? `<span class="badge-contested" title="${contested(it)}">⚠ contested</span>` : "") +
      `<br>` +
      (it.pairwise && it.pairwise.status === "live" ? `pairwise #${it.pairwise.rank_in_group}/${it.pairwise.group_size} · ` :
        it.pairwise && it.pairwise.status === "pending" ? `pairwise pending · ` : "") +
      `sharp ${it.sharpness} · expo ${it.exposure_quality} · ${visionLabel(it.vision)}` +
      (it.vision.sam_mask_returned ? ` · subj ${it.vision.subject_sharpness.toFixed(2)}` :
        it.vision.sam_done ? ` · subj —` : "") + `</div>`;
    card.onclick = (e) => {
      if (e.target.classList.contains("dup-badge")) { showGroup(it.group); return; }
      selected = i; renderGrid(); renderDetail();
    };
    grid.appendChild(card);
  });
}

function renderDetail() {
  const d = $("detail");
  const it = items[selected];
  if (!it) { d.hidden = true; return; }
  d.hidden = false;
  d.innerHTML =
    `<img src="/api/image/${it.id}" alt="">` +
    `<div class="info"><b>${it.filename}</b><br>` +
    `Rank score: <code>${it.score.toFixed(3)}</code> (P(keep) from taste model)<br>` +
    (it.blended_score !== undefined ? `Pairwise-blended (experimental, not used for sort): <code>${it.blended_score.toFixed(3)}</code><br>` : "") +
    (it.pairwise && it.pairwise.status === "live"
      ? `Pairwise in group: <code>#${it.pairwise.rank_in_group}/${it.pairwise.group_size}</code> · margin <code>${it.pairwise.margin}</code> · deciding: <i>${it.pairwise.deciding}</i><br>`
      : it.pairwise && it.pairwise.status === "pending"
      ? `Pairwise in group: <code>pending</code> (model score used)<br>`
      : "") +
    `Sharpness (log Laplacian var): <code>${it.sharpness}</code><br>` +
    `Exposure quality: <code>${it.exposure_quality}</code> · clipped: <code>${it.exposure_clipped}</code><br>` +
    `Vision: <code>${visionLabel(it.vision)}</code><br>` +
    (it.vision.sam_done
      ? (it.vision.sam_mask_returned
        ? `Subject sharpness (SAM 3.1): <code>${it.vision.subject_sharpness.toFixed(3)}</code> · masks: <code>${it.vision.sam_mask_count}</code><br>`
        : `Subject sharpness (SAM 3.1): <code>no subject</code><br>`)
      : `Subject sharpness (SAM 3.1): <code>pending</code><br>`) +
    `<span style="color:#999">${it.vision.note}</span><br>` +
    `Set: ${it.set} · calibration: ${it.calib_label || "—"} · your label: ${it.user_label || "—"}<br>` +
    (() => {
      const cards = reasonCards(it, groupMembers(it));
      const c = contested(it);
      return (cards.length ? `Why it stands out (measured, not praise): <span class="reasons">${cards.map((x) => `<span class="reason">${x}</span>`).join(" ")}</span><br>` : "") +
        (c ? `⚠ Contested: ${c}<br>` : "");
    })() + `<br>` +
    `<button onclick="labelSelected('keep')">Keep (L)</button> ` +
    `<button onclick="labelSelected('reject')">Reject (J)</button> ` +
    `<button onclick="labelSelected('maybe')">Maybe (M)</button> ` +
    `<button onclick="labelSelected(null)">Clear (K)</button></div>`;
}

async function labelSelected(label, advance) {
  const it = items[selected];
  if (!it) return;
  await api("/api/label", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: it.id, label }) });
  await refresh(it.id); // refit + re-rank; selection follows the same frame
  if (advance && items.length) {
    // Keyboard flagging auto-advances to the next frame (mouse clicks
    // in buttons do not — the user is still looking at that frame).
    const i = items.findIndex((x) => x.id === it.id);
    selected = Math.min((i >= 0 ? i : selected) + 1, items.length - 1);
    renderGrid(); renderDetail();
  }
}

let compareZoom = "fit";
function setCompareZoom(mode) { compareZoom = mode; applyCompareZoom(); }
function applyCompareZoom() {
  document.querySelectorAll(".sync-frame img").forEach((img) => {
    if (compareZoom === "fit") img.style.width = "100%";
    else if (compareZoom === "100") img.style.width = (img.naturalWidth || 800) + "px";
    else if (compareZoom === "200") img.style.width = ((img.naturalWidth || 800) * 2) + "px";
  });
}

async function showGroup(gi) {
  const data = await api("/api/groups");
  const grp = data.groups[gi];
  if (!grp) return;
  const view = $("groups-view");
  view.hidden = false;
  const factors = [
    { label: "Rank score", get: (m) => m.score, d: 3 },
    { label: "Sharpness (log)", get: (m) => m.sharpness, d: 2 },
    { label: "Exposure", get: (m) => m.exposure_quality, d: 3 },
    { label: "Composition (Muse)", get: (m) => m.vision.composition, d: 2 },
    { label: "Moment (Muse)", get: (m) => m.vision.moment, d: 2 },
    { label: "Subject sharpness (SAM)", get: (m) => (m.vision.sam_mask_returned ? m.vision.subject_sharpness : null), d: 3 },
  ];
  const best = factors.map((f) => {
    const vals = grp.map(f.get).filter((v) => v !== null && v !== undefined);
    return vals.length ? Math.max(...vals) : null;
  });
  const deltaRows = factors.map((f, fi) => `<tr><td>${f.label}</td>` +
    grp.map((m) => {
      const v = f.get(m);
      const isBest = v !== null && v !== undefined && best[fi] !== null && v === best[fi];
      return `<td class="${isBest ? "best" : ""}">${fmt(v, f.d)}</td>`;
    }).join("") + `</tr>`).join("");
  view.innerHTML = `<div class="group"><h3>Near-duplicate group ${gi + 1} — compare side by side</h3>` +
    `<div class="zoombar">Zoom: <button onclick="setCompareZoom('fit')">Fit</button> <button onclick="setCompareZoom('100')">100%</button> <button onclick="setCompareZoom('200')">200%</button> <span class="hint">scrolling one frame scrolls all (synced pan)</span></div>` +
    `<div class="compare">` +
    grp.map((m) => `<figure><div class="sync-frame"><img src="/api/image/${m.id}" data-sync onload="applyCompareZoom()"></div><figcaption>${m.filename} · score ${m.score.toFixed(2)} · sharp ${m.sharpness}` +
      (m.pairwise && m.pairwise.status === "live" ? ` · pairwise #${m.pairwise.rank_in_group}/${m.pairwise.group_size}${m.pairwise.is_winner ? " ★ winner" : ""} — ${m.pairwise.deciding}` : "") +
      `<br><span class="reasons">${reasonCards(m, grp).map((c) => `<span class="reason">${c}</span>`).join(" ")}</span>` +
      `</figcaption>` +
      `<button onclick="labelId('${m.id}','keep')">Keep</button><button onclick="labelId('${m.id}','reject')">Reject</button><button onclick="labelId('${m.id}','maybe')">Maybe</button></figure>`).join("") +
    `</div>` +
    `<table class="delta-table"><tr><th>Factor</th>` + grp.map((m) => `<th>${m.filename}</th>`).join("") + `</tr>${deltaRows}</table>` +
    `<p class="hint">Subject strip — zoomed crop of the largest SAM “person” box (one SAM call per frame on first view, cached after; frames with no subject show a note):</p>` +
    `<div class="strip">` + grp.map((m) => `<div class="strip-tile"><img src="/api/subject-crop/${m.id}" alt="" loading="lazy" onerror="this.replaceWith(document.createTextNode('no subject crop'))"><div>${m.filename}</div></div>`).join("") + `</div></div>`;
  applyCompareZoom();
  const frames = [...view.querySelectorAll(".sync-frame")];
  let syncing = false;
  frames.forEach((fr) => fr.addEventListener("scroll", () => {
    if (syncing) return;
    syncing = true;
    frames.forEach((o) => { if (o !== fr) { o.scrollLeft = fr.scrollLeft; o.scrollTop = fr.scrollTop; } });
    syncing = false;
  }));
  const headerH = document.querySelector("header") ? document.querySelector("header").offsetHeight : 0;
  window.scrollTo({ top: view.getBoundingClientRect().top + window.scrollY - headerH - 12, behavior: "smooth" });
}

async function labelId(id, label) {
  await api("/api/label", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, label }) });
  await refresh(id);
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (!items.length) return;
  if (e.key === "ArrowRight") { selected = Math.min(selected + 1, items.length - 1); renderGrid(); renderDetail(); }
  else if (e.key === "ArrowLeft") { selected = Math.max(selected - 1, 0); renderGrid(); renderDetail(); }
  else if (e.key === "l" || e.key === "L") labelSelected("keep", true);
  else if (e.key === "j" || e.key === "J") labelSelected("reject", true);
  else if (e.key === "k" || e.key === "K") labelSelected(null, true);
  else if (e.key === "p" || e.key === "P") labelSelected("keep", true);
  else if (e.key === "x" || e.key === "X") labelSelected("reject", true);
  else if (e.key === "u" || e.key === "U") labelSelected(null, true);
  else if (e.key === "m" || e.key === "M") labelSelected("maybe", true);
  else if (e.key === "z" || e.key === "Z") { const d = $("detail"); if (d) d.classList.toggle("zoomed"); }
});

$("btn-demo").onclick = async () => { await api("/api/load-demo", { method: "POST" }); selected = 0; await refresh(); };

async function renderEval() {
  const el = $("eval-result");
  el.hidden = false;
  el.innerHTML = "Computing…";
  try {
    const ev = await api("/api/evaluate");
    const bar = (v) => `<div class="bar"><div class="fill" style="width:${Math.max(0, Math.min(100, v * 100))}%"></div></div>`;
    const ins = ev.in_sample, ho = ev.held_out, snap = ev.pairwise_learned_snapshot;
    el.innerHTML =
      `<p><b>IN-SAMPLE — fit on the same labels the model trained on. NOT generalization.</b><br>` +
      `Labels: ${ins.n_labels} · keeper mean ${fmt(ins.keeper_mean, 3)} · reject mean ${fmt(ins.reject_mean, 3)} · separation <b>${fmt(ins.separation, 3)}</b> · top-25 keepers ${ins.top25_keepers} · top-50 keepers ${ins.top50_keepers} · label agreement ${fmt(ins.label_agreement, 3)}</p>` +
      (ho ? `<p><b>HELD-OUT — unseen frames (fresh splits, fit on train only).</b><br>Pair accuracy ${ho.pair_accuracy} ± ${ho.sem} over ${ho.splits} splits. ${ho.note}</p>${bar(ho.pair_accuracy)}` :
        `<p><b>HELD-OUT:</b> not enough labeled frames for splits yet.</p>`) +
      (snap ? `<p>Offline pairwise-learned snapshot (2026-10-28 experiment): pairwise ${snap.held_out_pairwise_accuracy} vs pointwise ${snap.held_out_pointwise_accuracy} on held-out pairs — ${snap.verdict}</p>` : "") +
      `<p class="hint">${ev.honesty_note}</p>`;
  } catch (err) { el.innerHTML = "Evaluation failed: " + err.message; }
}

$("btn-eval").onclick = renderEval;
$("btn-keepset").onclick = async () => {
  const n = parseInt($("keep-n").value, 10) || 20;
  const res = await api("/api/keep-set", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n }) });
  const el = $("keepset-result");
  el.hidden = false;
  el.innerHTML = `<b>${res.count} keeps, ${res.groups_covered} groups covered, ${res.scenes} scenes</b> (a scene = a duplicate group or a standalone frame; one winner per group first, then fill by score skipping already-represented groups)<br>` +
    `<div class="strip">` + res.members.map((m) => `<div class="strip-tile"><img src="/api/thumb/${m.id}" loading="lazy"><div>${m.filename}</div></div>`).join("") + `</div>` +
    `<button id="btn-export-keepset">Export this keep-set</button>`;
  $("btn-export-keepset").onclick = async () => {
    const ex = await api("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: res.ids }) });
    const eel = $("export-result");
    eel.hidden = false;
    eel.innerHTML = `Exported <b>${ex.count}</b> frames (${ex.mode}) to <code>${ex.export_dir}</code><br><a href="${ex.contact_sheet}" target="_blank">Open contact sheet</a>`;
  };
};

$("btn-groups").onclick = async () => {
  const data = await api("/api/groups");
  const view = $("groups-view");
  groupsVisible = !groupsVisible;
  view.hidden = !groupsVisible;
  if (groupsVisible) {
    view.innerHTML = data.groups.map((grp, gi) => `<div class="group"><h3>Group ${gi + 1}</h3><div class="compare">` +
      grp.map((m) => `<figure><img src="/api/thumb/${m.id}"><figcaption>${m.filename} · score ${m.score.toFixed(2)}</figcaption>` +
      `<button onclick="labelId('${m.id}','keep')">Keep</button><button onclick="labelId('${m.id}','reject')">Reject</button><button onclick="showGroup(${gi})">Compare full</button></figure>`).join("") +
      `</div></div>`).join("");
  }
};
$("btn-export").onclick = async () => {
  const res = await api("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  const el = $("export-result");
  el.hidden = false;
  el.innerHTML = `Exported <b>${res.count}</b> frames (${res.mode}) to <code>${res.export_dir}</code><br>` +
    `<a href="${res.contact_sheet}" target="_blank">Open contact sheet</a> · files: ${res.files.slice(0, 8).join(", ")}${res.files.length > 8 ? "…" : ""}`;
};
$("file-input").onchange = async (e) => {
  const fd = new FormData();
  for (const f of e.target.files) fd.append("files", f);
  await api("/api/upload", { method: "POST", body: fd });
  selected = 0; await refresh();
};
const dz = $("dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", async (e) => {
  e.preventDefault(); dz.classList.remove("over");
  const fd = new FormData();
  for (const f of e.dataTransfer.files) fd.append("files", f);
  await api("/api/upload", { method: "POST", body: fd });
  selected = 0; await refresh();
});

refresh();
