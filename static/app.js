let items = [];
let selected = 0;
let groupsVisible = false;
let groupsData = [];
let q1Sharp = 0;
let q3Sharp = 0;
// People-split state: flags keyed by item id, filter, last job status.
let peopleById = {};
let peopleFilter = "all";
let lastPeople = null;
let peoplePoll = null;

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(body.error || `${path} -> ${r.status}`);
    err.status = r.status;
    err.body = body;
    throw err;
  }
  return body;
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

// People filter: the grid/detail/keyboard all work on this visible list,
// so "With people" shows exactly the flagged frames and nothing else.
function visibleItems() {
  if (peopleFilter === "with") return items.filter((it) => peopleById[it.id] && peopleById[it.id].has_person === true);
  if (peopleFilter === "without") return items.filter((it) => peopleById[it.id] && peopleById[it.id].has_person === false);
  return items;
}

function updatePeopleBar(p) {
  if (!p) return;
  const all = $("btn-people-all"), w = $("btn-people-with"), wo = $("btn-people-without");
  all.textContent = `All (${p.total})`;
  w.textContent = `With people (${p.with_people})`;
  wo.textContent = `Without people (${p.without_people})`;
  [all, w, wo].forEach((b) => b.classList.remove("active"));
  ({ all, with: w, without: wo })[peopleFilter].classList.add("active");
  const st = $("people-status");
  if (p.status === "running") st.textContent = `Detecting people… ${p.done}/${p.total} (offline, ~0.6s/photo)`;
  else if (p.status === "done") st.textContent = `People detection done — ${p.with_people} with people, ${p.without_people} without${p.seconds ? ` in ${p.seconds}s` : " (cached)"}`;
  else if (p.status === "unavailable") st.textContent = "People detection unavailable — " + (p.note || "mediapipe/model missing");
  else st.textContent = p.done > 0 ? `${p.done}/${p.total} detected — run detection to finish` : "People not detected yet — run detection (offline, ~0.6s/photo)";
  // Separate into folders needs a complete detection pass first.
  const sep = $("btn-people-separate");
  if (sep) sep.disabled = p.status !== "done";
}

async function refresh(keepSelectionId) {
  const [st, rank, grp, ppl, src] = await Promise.all([api("/api/state"), api("/api/rank"), api("/api/groups"), api("/api/people"), api("/api/source")]);
  items = rank.items;
  groupsData = grp.groups;
  computeQuartiles();
  peopleById = {};
  (ppl.items || []).forEach((p) => { peopleById[p.id] = p; });
  lastPeople = ppl;
  updatePeopleBar(ppl);
  $("source-current").textContent = `${src.demo_dir} · ${src.visible}/${src.total} frames${src.persisted ? "" : " (built-in set)"}`;
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
    const vis = visibleItems();
    const i = vis.findIndex((x) => x.id === keepSelectionId);
    if (i >= 0) selected = i;
  }
  selected = Math.max(0, Math.min(selected, Math.max(0, visibleItems().length - 1)));
  renderDetail();
  if (!$("eval-result").hidden) renderEval();
}

function renderGrid() {
  const grid = $("grid");
  grid.innerHTML = "";
  const vis = visibleItems();
  vis.forEach((it, i) => {
    const card = document.createElement("div");
    card.className = "card" + (i === selected ? " selected" : "") + (it.user_label ? " " + it.user_label : "");
    const pb = peopleById[it.id];
    card.innerHTML =
      `<img src="/api/thumb/${it.id}" alt="" loading="lazy">` +
      `<div class="meta"><span class="score">${it.score.toFixed(2)}</span> ${it.filename}` +
      (it.group !== null && it.group !== undefined ? `<span class="dup-badge" title="Near-duplicate group">dup ${it.group + 1}</span>` : "") +
      (pb && pb.has_person === true ? `<span class="label-badge people" title="Offline tiled EfficientDet-Lite0, conf 0.3">👤 people${pb.person_count ? " ×" + pb.person_count : ""}</span>` : "") +
      (pb && pb.has_person === false ? `<span class="label-badge nopeople" title="Offline tiled EfficientDet-Lite0, conf 0.3">no people</span>` : "") +
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
  const it = visibleItems()[selected];
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
      const pb = peopleById[it.id];
      if (!pb || pb.has_person === null || pb.has_person === undefined) return `People: <i>not detected yet</i> — run Detect people (offline)<br>`;
      const line = pb.has_person
        ? `People: <b>yes</b> — ${pb.person_count} detection${pb.person_count === 1 ? "" : "s"}${pb.max_conf !== null && pb.max_conf !== undefined ? `, best confidence ${pb.max_conf}` : ""} (offline tiled EfficientDet-Lite0)`
        : `People: <b>none detected</b> (offline tiled EfficientDet-Lite0)`;
      const flipTo = pb.has_person ? "no people" : "people";
      return `${line} <button onclick="flipPeople('${it.id}', ${!pb.has_person})">Flip to ${flipTo}</button><br>`;
    })() +
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
  const it = visibleItems()[selected];
  if (!it) return;
  await api("/api/label", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: it.id, label }) });
  await refresh(it.id); // refit + re-rank; selection follows the same frame
  if (advance) {
    // Keyboard flagging auto-advances to the next frame (mouse clicks
    // in buttons do not — the user is still looking at that frame).
    const vis = visibleItems();
    const i = vis.findIndex((x) => x.id === it.id);
    selected = Math.min((i >= 0 ? i : selected) + 1, Math.max(0, vis.length - 1));
    renderGrid(); renderDetail();
  }
}

// Flip one photo's people verdict (manual correction). The cache is
// updated server-side, so badges, filter counts and the dry-run
// reflect the change after this refresh.
async function flipPeople(id, hasPerson) {
  await api("/api/people/override", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, has_person: hasPerson }) });
  await refresh(id);
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
      (m.is_suggested_keeper ? ` <span class="label-badge winner">★ suggested keeper</span>` : "") +
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
  const vis = visibleItems();
  if (!vis.length) return;
  if (e.key === "ArrowRight") { selected = Math.min(selected + 1, vis.length - 1); renderGrid(); renderDetail(); }
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

function setPeopleFilter(f) {
  peopleFilter = f;
  selected = 0;
  renderGrid(); renderDetail();
  updatePeopleBar(lastPeople);
}

$("btn-people-all").onclick = () => setPeopleFilter("all");
$("btn-people-with").onclick = () => setPeopleFilter("with");
$("btn-people-without").onclick = () => setPeopleFilter("without");
$("btn-people-run").onclick = async () => {
  const res = await api("/api/people/start", { method: "POST" });
  lastPeople = res;
  updatePeopleBar(res);
  if (res.status === "unavailable") return;
  if (peoplePoll) clearInterval(peoplePoll);
  peoplePoll = setInterval(async () => {
    try {
      const p = await api("/api/people");
      lastPeople = p;
      updatePeopleBar(p);
      if (p.status === "done" || p.status === "unavailable") {
        clearInterval(peoplePoll); peoplePoll = null;
        await refresh();
      }
    } catch (err) { clearInterval(peoplePoll); peoplePoll = null; }
  }, 2000);
};

function escHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Separate into folders: dry-run first (moves nothing), confirm moves.
$("btn-people-separate").onclick = async () => {
  const el = $("separate-result");
  el.hidden = false;
  el.innerHTML = "Checking what would move…";
  let dry;
  try {
    dry = await api("/api/people/separate/dry-run");
  } catch (err) {
    // api() throws on non-2xx without the body; re-fetch for the message.
    let msg = err.message;
    try {
      const r = await fetch("/api/people/separate/dry-run");
      const body = await r.json();
      msg = body.error || msg;
      if (body.no_result && body.no_result.length) {
        el.innerHTML = `<b>People detection is not complete.</b><br>No result yet for: ${body.no_result.map(escHtml).join(", ")}<br>Run Detect people first.`;
        return;
      }
    } catch (_) { /* keep err.message */ }
    el.innerHTML = `<b>Separate into folders failed:</b> ${escHtml(msg)}`;
    return;
  }
  const list = (arr) => arr.length ? `<ul>${arr.map((x) => `<li>${escHtml(x)}</li>`).join("")}</ul>` : "<i>none</i>";
  el.innerHTML =
    `<b>Separate into folders — dry run</b><br>` +
    `${dry.counts.with_people} RAW file(s) with people → <code>${escHtml(dry.dest_with)}</code><br>` +
    `${dry.counts.without_people} RAW file(s) without people → <code>${escHtml(dry.dest_without)}</code><br>` +
    `No matching RAW (${dry.unmatched.length}): ${list(dry.unmatched)}` +
    `No detection result (${dry.no_result.length}): ${list(dry.no_result)}` +
    `<p class="hint">Only the RAW originals move — the previews stay where they are. A move log is written next to the moves and in the app's data folder, so every move can be undone by hand.</p>` +
    `<button id="btn-separate-confirm">Move the RAW files</button> <button id="btn-separate-cancel">Cancel</button>`;
  $("btn-separate-cancel").onclick = () => { el.hidden = true; el.innerHTML = ""; };
  $("btn-separate-confirm").onclick = async () => {
    el.innerHTML = "Moving files…";
    try {
      const res = await api("/api/people/separate/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed: true }) });
      el.innerHTML =
        `<b>Moved ${res.moved_with} RAW file(s) with people</b> → <code>${escHtml(res.dest_with)}</code><br>` +
        `<b>Moved ${res.moved_without} RAW file(s) without people</b> → <code>${escHtml(res.dest_without)}</code><br>` +
        `Skipped (no detection): ${res.skipped_no_result.length ? escHtml(res.skipped_no_result.join(", ")) : "none"}<br>` +
        `Skipped (no matching RAW): ${res.skipped_unmatched.length ? escHtml(res.skipped_unmatched.join(", ")) : "none"}<br>` +
        `Move log: <code>${escHtml(res.log)}</code> (a copy is in each destination folder too)`;
      await refresh();
    } catch (err) {
      el.innerHTML = `<b>Move failed:</b> ${escHtml(err.message)} — files moved before the failure stay moved; the move log in the app's data folder lists them.`;
    }
  };
};

// Best shots: one best per similar group, user rejects, then move.
// Rejected picks are excluded client-side and sent as `excluded`.
let bestExcluded = new Set();
let bestTopN = 10;

function renderBest(dry) {
  const el = $("best-result");
  const kept = dry.picks.filter((p) => !bestExcluded.has(p.id));
  el.innerHTML =
    `<b>Best shots — one best per similar group, top ${dry.top_n} by rank</b><br>` +
    `${kept.length} pick(s) → <code>${escHtml(dry.dest)}</code> · ${dry.skipped_dupes.length} duplicate(s) held back · ` +
    (dry.unmatched.length ? `no RAW for: ${dry.unmatched.map(escHtml).join(", ")}` : "every pick has a matching RAW") + `<br>` +
    `<div class="best-list">` + dry.picks.map((p) => {
      const out = bestExcluded.has(p.id);
      return `<div class="best-pick${out ? " excluded" : ""}"><img src="/api/thumb/${p.id}" alt="" loading="lazy">` +
        `<div><b>${escHtml(p.filename)}</b> · score ${p.score.toFixed(3)} · sharp ${p.sharpness} · expo ${p.exposure_quality}<br>` +
        (p.group_size > 1 ? `best of ${p.group_size} similar` : "no duplicates") +
        ` <button data-best-reject="${p.id}">${out ? "Restore" : "Reject ✕"}</button></div></div>`;
    }).join("") + `</div>` +
    `<p class="hint">Rejected picks are excluded from the move. Only the RAW originals move — the previews stay where they are. A move log is written next to the moves and in the app's data folder.</p>` +
    `<button id="btn-best-confirm">Move best shots (${kept.length})</button> <button id="btn-best-cancel">Cancel</button>`;
  el.querySelectorAll("[data-best-reject]").forEach((b) => {
    b.onclick = () => {
      const id = b.getAttribute("data-best-reject");
      if (bestExcluded.has(id)) bestExcluded.delete(id); else bestExcluded.add(id);
      renderBest(dry);
    };
  });
  $("btn-best-cancel").onclick = () => { el.hidden = true; el.innerHTML = ""; };
  $("btn-best-confirm").onclick = async () => {
    el.innerHTML = "Moving files…";
    try {
      const res = await api("/api/best/separate/run", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmed: true, top_n: bestTopN, excluded: [...bestExcluded] }) });
      el.innerHTML =
        `<b>Moved ${res.moved} best-shot RAW file(s)</b> → <code>${escHtml(res.dest)}</code><br>` +
        `Excluded by you: ${res.excluded.length} · Skipped (no matching RAW): ${res.skipped_unmatched.length ? escHtml(res.skipped_unmatched.join(", ")) : "none"}<br>` +
        `Move log: <code>${escHtml(res.log)}</code> (a copy is in the destination folder too)`;
      await refresh();
    } catch (err) {
      el.innerHTML = `<b>Move failed:</b> ${escHtml(err.message)}`;
    }
  };
}

$("btn-best-preview").onclick = async () => {
  bestTopN = parseInt($("best-n").value, 10) || 10;
  bestExcluded = new Set();
  const el = $("best-result");
  el.hidden = false;
  el.innerHTML = "Ranking best shots…";
  try {
    const dry = await api(`/api/best/dry-run?top_n=${bestTopN}`);
    renderBest(dry);
  } catch (err) {
    el.innerHTML = `<b>Best shots failed:</b> ${escHtml(err.message)}`;
  }
};

// Feed any folder: switch to an already-ingested working set
// (tools/ingest.py). No restart; the choice persists.
$("btn-source-set").onclick = async () => {
  const dir = $("source-input").value.trim();
  const st = $("source-status");
  if (!dir) { st.textContent = "Type a folder path first."; return; }
  st.textContent = "Switching…";
  try {
    const res = await api("/api/source/set", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo_dir: dir }) });
    st.textContent = `Switched to ${res.set_name}`;
    selected = 0;
    await refresh();
  } catch (err) {
    st.textContent = err.message;
  }
};

// Near-duplicate groups, with the suggested keeper, per-photo Delete
// (to the OS Trash, never permanent), and per-group "Delete the rest".
async function renderGroupsView() {
  const data = await api("/api/groups");
  const view = $("groups-view");
  if (!data.groups.length) {
    view.innerHTML = "<p>No near-duplicate groups in this set.</p>";
    return;
  }
  view.innerHTML = data.groups.map((grp, gi) => {
    const keeper = grp.find((m) => m.is_suggested_keeper);
    return `<div class="group"><h3>Group ${gi + 1} — ${grp.length} similar frames${keeper ? ` · suggested keeper: ${escHtml(keeper.filename)}` : ""}</h3><div class="compare">` +
      grp.map((m) => `<figure><img src="/api/thumb/${m.id}"><figcaption>${escHtml(m.filename)} · score ${m.score.toFixed(2)}` +
      (m.is_suggested_keeper ? ` <span class="label-badge winner">★ suggested keeper</span>` : "") + `</figcaption>` +
      `<button onclick="labelId('${m.id}','keep')">Keep</button><button onclick="labelId('${m.id}','reject')">Reject</button><button onclick="showGroup(${gi})">Compare full</button>` +
      `<button class="danger" data-delete-photo="${m.id}" data-filename="${escHtml(m.filename)}">Delete</button><span class="del-confirm" data-del-confirm="${m.id}"></span></figure>`).join("") +
      `</div>` +
      (grp.length > 1 && keeper ? `<button class="danger" data-delete-rest="${gi}" data-keep-id="${keeper.id}">Delete the rest (keep ${escHtml(keeper.filename)})</button><span class="del-confirm" data-del-rest-confirm="${gi}"></span>` : "") +
      `</div>`;
  }).join("");
  view.querySelectorAll("[data-delete-photo]").forEach((b) => {
    b.onclick = () => {
      const id = b.getAttribute("data-delete-photo");
      const fn = b.getAttribute("data-filename");
      const slot = view.querySelector(`[data-del-confirm="${id}"]`);
      slot.innerHTML = `Move <b>${escHtml(fn)}</b> to Trash? <button data-del-yes="${id}">Yes, delete</button> <button data-del-no="${id}">Cancel</button>`;
      slot.querySelector("[data-del-yes]").onclick = async () => {
        slot.textContent = "Deleting…";
        try {
          await api("/api/dupes/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }) });
          await refresh();
          await renderGroupsView();
        } catch (err) { slot.textContent = "Delete failed: " + err.message; }
      };
      slot.querySelector("[data-del-no]").onclick = () => { slot.innerHTML = ""; };
    };
  });
  view.querySelectorAll("[data-delete-rest]").forEach((b) => {
    b.onclick = () => {
      const gi = parseInt(b.getAttribute("data-delete-rest"), 10);
      const keepId = b.getAttribute("data-keep-id");
      const grp = data.groups[gi];
      const rest = grp.filter((m) => m.id !== keepId);
      const keeper = grp.find((m) => m.id === keepId);
      const slot = view.querySelector(`[data-del-rest-confirm="${gi}"]`);
      slot.innerHTML = `<b>Delete ${rest.length} photo(s) to Trash, keeping ${escHtml(keeper.filename)}?</b><br>` +
        rest.map((m) => escHtml(m.filename)).join(", ") +
        ` <button data-del-rest-yes="${gi}">Yes, delete the rest</button> <button data-del-rest-no="${gi}">Cancel</button>`;
      slot.querySelector("[data-del-rest-yes]").onclick = async () => {
        slot.textContent = "Deleting…";
        try {
          await api("/api/dupes/delete-group", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ group_id: gi, keep_id: keepId, confirmed: true }) });
          await refresh();
          await renderGroupsView();
        } catch (err) { slot.textContent = "Delete failed: " + err.message; }
      };
      slot.querySelector("[data-del-rest-no]").onclick = () => { slot.innerHTML = ""; };
    };
  });
}

$("btn-groups").onclick = async () => {
  const view = $("groups-view");
  groupsVisible = !groupsVisible;
  view.hidden = !groupsVisible;
  if (groupsVisible) await renderGroupsView();
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
