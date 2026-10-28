let items = [];
let selected = 0;
let groupsVisible = false;

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

async function refresh(keepSelectionId) {
  const [st, rank] = await Promise.all([api("/api/state"), api("/api/rank")]);
  items = rank.items;
  $("vision-banner").hidden = st.vision_status === "live";
  $("vision-banner").className = "banner " + (st.vision_status === "live" ? "live" : "pending");
  if (st.vision_status === "live") {
    $("vision-banner").textContent = "Vision pass live — composition / moment scores are real model output.";
    $("vision-banner").hidden = false;
  }
  $("state").textContent =
    `${st.set_name} · ${st.total} frames · ${st.keeps} kept · ${st.rejects} rejected · ` +
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
      (it.user_label ? `<span class="label-badge ${it.user_label}">${it.user_label}</span>` : "") +
      (it.calib_label ? `<span class="label-badge calib">calib ${it.calib_label}</span>` : "") +
      `<br>sharp ${it.sharpness} · expo ${it.exposure_quality} · ${visionLabel(it.vision)}</div>`;
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
    `Sharpness (log Laplacian var): <code>${it.sharpness}</code><br>` +
    `Exposure quality: <code>${it.exposure_quality}</code> · clipped: <code>${it.exposure_clipped}</code><br>` +
    `Vision: <code>${visionLabel(it.vision)}</code><br>` +
    `<span style="color:#999">${it.vision.note}</span><br>` +
    `Set: ${it.set} · calibration: ${it.calib_label || "—"} · your label: ${it.user_label || "—"}<br><br>` +
    `<button onclick="labelSelected('keep')">Keep (L)</button> ` +
    `<button onclick="labelSelected('reject')">Reject (J)</button> ` +
    `<button onclick="labelSelected(null)">Clear (K)</button></div>`;
}

async function labelSelected(label) {
  const it = items[selected];
  if (!it) return;
  await api("/api/label", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: it.id, label }) });
  await refresh(it.id); // refit + re-rank; selection follows the same frame
}

async function showGroup(gi) {
  const data = await api("/api/groups");
  const grp = data.groups[gi];
  const view = $("groups-view");
  view.hidden = false;
  view.innerHTML = `<div class="group"><h3>Near-duplicate group ${gi + 1} — compare side by side</h3><div class="compare">` +
    grp.map((m) => `<figure><img src="/api/image/${m.id}"><figcaption>${m.filename} · score ${m.score.toFixed(2)} · sharp ${m.sharpness}</figcaption>` +
      `<button onclick="labelId('${m.id}','keep')">Keep</button><button onclick="labelId('${m.id}','reject')">Reject</button></figure>`).join("") +
    `</div></div>`;
  view.scrollIntoView({ behavior: "smooth" });
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
  else if (e.key === "l" || e.key === "L") labelSelected("keep");
  else if (e.key === "j" || e.key === "J") labelSelected("reject");
  else if (e.key === "k" || e.key === "K") labelSelected(null);
});

$("btn-demo").onclick = async () => { await api("/api/load-demo", { method: "POST" }); selected = 0; await refresh(); };
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
