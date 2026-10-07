// The session id and the last scan summary live only in this page's memory.
// Nothing is stored in localStorage, sessionStorage or cookies.
let sessionId = null;
let summaryText = null;
const $ = (id) => document.getElementById(id);
const auth = () => ({ Authorization: "Bearer " + $("token").value });

function addLine(who, text) {
  const p = document.createElement("p");
  const b = document.createElement("strong");
  b.textContent = who + ": ";
  p.appendChild(b);
  p.appendChild(document.createTextNode(text)); // plain text only, never HTML
  p.style.whiteSpace = "pre-wrap";
  $("log").appendChild(p);
}

$("send").addEventListener("click", async () => {
  const f = $("file").files[0];
  if (!f) return;
  const r = await fetch("segment", { method: "POST", headers: { ...auth(), "X-Filename": f.name }, body: f });
  const data = await r.json();
  if (!r.ok) { $("summary").textContent = "Error: " + (data.detail || r.status); return; }
  const s = data.summary;
  summaryText = `liver region ${s.liver_volume_ml} mL, lesion candidates ${s.lesion_count}, lesion volume ${s.tumor_volume_ml} mL`;
  $("summary").textContent = JSON.stringify(s, null, 2);
  $("overlay").src = data.overlay_png_base64 ? "data:image/png;base64," + data.overlay_png_base64 : "";
});

$("ask").addEventListener("click", async () => {
  const q = $("question").value.trim();
  if (!q) return;
  addLine("You", q);
  $("question").value = "";
  const r = await fetch("chat", {
    method: "POST",
    headers: { ...auth(), "Content-Type": "application/json" },
    body: JSON.stringify({ question: q, session_id: sessionId, summary_text: summaryText }),
  });
  const data = await r.json();
  if (!r.ok) { addLine("Error", data.detail || String(r.status)); return; }
  sessionId = data.session_id;
  addLine("Assistant", data.answer);
});

$("end").addEventListener("click", async () => {
  if (sessionId) await fetch("chat/" + encodeURIComponent(sessionId), { method: "DELETE", headers: auth() });
  sessionId = null;
  $("log").textContent = "";
});
