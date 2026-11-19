/* TeamDocs AI frontend — talks to the real FastAPI backend. No framework, no build step.
   All dynamic text is inserted with textContent / createTextNode (no innerHTML), so
   document content or LLM output can never inject markup. */
(() => {
  "use strict";
  const API = ((window.TEAMDOCS_CONFIG && window.TEAMDOCS_CONFIG.apiBaseUrl) || "").replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);
  const state = { documents: [], llm: false, ready: false, busy: false, lastSources: [] };

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) node.append(c);
    return node;
  }

  async function api(path, options = {}) {
    let response;
    try { response = await fetch(API + path, options); }
    catch { throw new Error("Cannot reach the server. Is the backend running?"); }
    let body = null;
    try { body = await response.json(); } catch { /* non-JSON */ }
    if (!response.ok) {
      const message = body && body.error ? body.error.message : `Request failed (${response.status}).`;
      const err = new Error(message); err.code = body && body.error ? body.error.code : "http_error"; err.status = response.status;
      throw err;
    }
    return body;
  }

  /* ---------------------------------------------------------------- banner / status */
  function banner(message, kind = "error") {
    const b = $("banner");
    if (!message) { b.hidden = true; return; }
    b.textContent = message; b.className = "banner" + (kind === "info" ? " info" : ""); b.hidden = false;
  }
  const fmtSize = (n) => n == null ? "" : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`;
  const fmtDate = (s) => { const d = s ? new Date(s) : null; return d && !isNaN(d) ? d.toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : ""; };

  async function refreshStatus() {
    const dotApi = $("dot-api"), dotLlm = $("dot-llm");
    try {
      const res = await fetch(API + "/ready");
      const body = await res.json();
      state.ready = !!body.ready;
      dotApi.className = "dot " + (body.ready ? "ok" : "err");
      $("status-api").textContent = body.ready ? "Backend ready" : "Backend not ready";
      state.llm = !!(body.checks && body.checks.llm_configured);
      dotLlm.className = "dot " + (state.llm ? "ok" : "warn");
      $("status-llm").textContent = state.llm ? "OpenRouter key configured" : "No OpenRouter key";
      $("status-meta").textContent = body.ready ? `${body.documents ?? 0} documents · ${body.chunks ?? 0} chunks` : "";
      if (!body.ready) banner((body.checks && body.checks.error) || "The backend is still starting or failed to initialise.");
      else banner("");
      refreshProviders();
    } catch {
      state.ready = false; dotApi.className = "dot err"; $("status-api").textContent = "Backend unreachable";
      banner("Cannot reach the backend" + (API ? ` at ${API}` : "") + ".");
    }
  }

  async function refreshProviders() {
    try {
      const st = await api("/api/v1/status");
      state.status = st;
      const rag = st.llm.agents.rag || {};
      const dotLlm = $("dot-llm");
      if (rag.provider) { dotLlm.className = "dot " + (rag.provider === "openrouter" ? "ok" : "warn"); $("status-llm").textContent = `LLM: ${rag.provider === "openrouter" ? "OpenRouter" : "Ollama (local)"}`; }
      else { dotLlm.className = "dot err"; $("status-llm").textContent = "LLM: not available"; }
      $("dot-gh").className = "dot " + (st.github.configured ? "ok" : "warn");
      $("status-gh").textContent = st.github.configured ? "GitHub token set" : "GitHub: no token";
    } catch { /* status panel is informational */ }
  }

  /* ---------------------------------------------------------------- navigation */
  function show(view) {
    for (const name of ["chat", "ask", "docs", "settings"]) {
      $("view-" + name).hidden = name !== view;
      const tab = $("tab-" + name);
      tab.classList.toggle("active", name === view); tab.setAttribute("aria-selected", String(name === view));
    }
    if (view === "docs") loadDocuments();
    if (view === "settings") loadSettings(false);
  }
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
  document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => show(b.dataset.goto)));

  /* ---------------------------------------------------------------- documents */
  async function loadDocuments() {
    try {
      const body = await api("/api/v1/documents");
      state.documents = body.documents;
    } catch (e) { banner(e.message); return; }
    renderDocuments();
  }

  function renderDocuments() {
    const docs = state.documents;
    const badge = $("doc-count"); badge.hidden = docs.length === 0; badge.textContent = docs.length;
    $("empty-docs").hidden = docs.length > 0; $("examples").hidden = docs.length === 0;
    const select = $("opt-doc"), current = select.value;
    select.replaceChildren(el("option", { value: "", text: "All documents" }),
      ...docs.map((d) => el("option", { value: d.filename, text: d.filename })));
    if (docs.some((d) => d.filename === current)) select.value = current;

    const list = $("docs-list");
    if (!docs.length) { list.replaceChildren(el("div", { class: "empty" }, el("h2", { text: "Nothing indexed yet" }), el("p", { text: "Upload a file above to build your knowledge base." }))); return; }
    list.replaceChildren(...docs.map((d) => el("div", { class: "doc" },
      el("span", { class: "ext", text: d.file_type || "file" }),
      el("div", { class: "info" },
        el("div", { class: "fname", text: d.filename }),
        el("div", { class: "sub", text: [`${d.chunk_count} chunks`, d.pages ? `${d.pages} pages` : null, fmtSize(d.size_bytes), fmtDate(d.ingested_at)].filter(Boolean).join(" · ") })),
      el("button", { class: "btn small", type: "button", text: "Re-index", onclick: (e) => docAction(e.currentTarget, d, "reindex") }),
      el("button", { class: "btn small danger", type: "button", text: "Delete", onclick: (e) => docAction(e.currentTarget, d, "delete") }))));
  }

  async function docAction(button, doc, action) {
    if (action === "delete" && !confirm(`Remove “${doc.filename}” from the index? This cannot be undone.`)) return;
    button.disabled = true;
    try {
      if (action === "delete") await api(`/api/v1/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
      else await api(`/api/v1/documents/${encodeURIComponent(doc.doc_id)}/reindex`, { method: "POST" });
      banner("");
    } catch (e) { banner(e.message); }
    await Promise.all([loadDocuments(), refreshStatus()]);
  }

  const dz = $("dropzone"), fileInput = $("file-input");
  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); uploadFiles([...e.dataTransfer.files]); });
  fileInput.addEventListener("change", () => { uploadFiles([...fileInput.files]); fileInput.value = ""; });

  async function uploadFiles(files) {
    const log = $("upload-log");
    for (const file of files) {
      const status = el("span", { class: "wait", text: "Indexing…" });
      const row = el("li", {}, el("strong", { text: file.name }), status);
      log.prepend(row);
      const form = new FormData(); form.append("file", file);
      try {
        const r = await api("/api/v1/documents", { method: "POST", body: form });
        status.className = "ok";
        status.textContent = r.status === "indexed" ? `Indexed (${r.chunk_count} chunks)`
          : r.status === "replaced" ? `Updated (${r.chunk_count} chunks; old version removed)`
          : r.status === "duplicate" ? `Skipped — identical to “${r.filename}”` : "Already indexed — unchanged";
      } catch (e) { status.className = "bad"; status.textContent = e.message; }
    }
    await Promise.all([loadDocuments(), refreshStatus()]);
  }

  /* ---------------------------------------------------------------- answer rendering */
  function inline(text, onCite) {
    const out = [];
    const re = /(\*\*[^*]+\*\*|`[^`]+`|\[\d{1,3}\])/g;
    let last = 0, m;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(document.createTextNode(text.slice(last, m.index)));
      const tok = m[0];
      if (tok.startsWith("**")) out.push(el("strong", { text: tok.slice(2, -2) }));
      else if (tok.startsWith("`")) out.push(el("code", { text: tok.slice(1, -1) }));
      else { const n = Number(tok.slice(1, -1)); out.push(el("button", { class: "cite", type: "button", title: `Show source ${n}`, "aria-label": `Show source ${n}`, text: String(n), onclick: () => onCite(n) })); }
      last = m.index + tok.length;
    }
    if (last < text.length) out.push(document.createTextNode(text.slice(last)));
    return out;
  }

  function renderMarkdown(text, onCite) {
    const root = el("div", { class: "answer" });
    let list = null, para = [];
    const flush = () => { if (para.length) { root.append(el("p", {}, inline(para.join(" "), onCite))); para = []; } };
    const endList = () => { list = null; };
    for (const raw of text.split("\n")) {
      const line = raw.trimEnd();
      const bullet = line.match(/^\s*[-*•]\s+(.*)$/), number = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (bullet || number) {
        flush();
        const type = bullet ? "ul" : "ol";
        if (!list || list.tagName.toLowerCase() !== type) { list = el(type); root.append(list); }
        list.append(el("li", {}, inline((bullet || number)[1], onCite)));
      } else if (!line.trim()) { flush(); endList(); } else { endList(); para.push(line.trim()); }
    }
    flush();
    return root;
  }

  function sourceCard(s, prefix = "source") {
    const loc = [s.page ? `page ${s.page}` : null, s.section].filter(Boolean).join(" · ");
    const scores = [];
    if (s.dense_score != null) scores.push(`Semantic similarity ${(s.dense_score * 100).toFixed(0)}%`);
    if (s.bm25_score != null) scores.push(`Keyword score ${s.bm25_score.toFixed(2)}`);
    scores.push(`Rank score ${s.score.toFixed(4)}`);
    return el("details", { class: "source", id: `${prefix}-${s.index}` },
      el("summary", {}, el("span", { class: "num", text: String(s.index) }),
        el("div", {}, el("div", { class: "name", text: s.filename }), loc ? el("div", { class: "loc", text: loc }) : null),
        el("span", { class: "tag" + (s.cited ? " cited" : ""), text: s.cited ? "cited in answer" : "retrieved" })),
      el("div", { class: "excerpt", text: s.excerpt }),
      safeLink(s.source_url, "Open original page", "source-link"),
      el("div", { class: "scores" }, scores.map((t) => el("span", { text: t }))));
  }

  function safeLink(url, text, cls) {
    return url && /^https?:\/\//i.test(url) ? el("a", { href: url, class: cls, target: "_blank", rel: "noopener noreferrer", text }) : null;
  }

  function openSource(n, prefix = "source") {
    const card = $(`${prefix}-${n}`); if (!card) return;
    card.open = true; card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 1600);
  }

  function renderResult(r) {
    const box = $("result"); const parts = [];
    if (r.generated) {
      const body = r.insufficient_evidence
        ? el("div", { class: "notice", role: "note" }, el("span", { text: r.answer }))
        : renderMarkdown(r.answer, openSource);
      parts.push(el("section", { class: "answer-card" }, el("h2", { text: "Answer" }), body,
        el("div", { class: "meta" }, el("span", { text: `Mode: ${r.mode}` }),
          r.model ? el("span", { text: `Model: ${r.model}` }) : null,
          r.timings_ms && r.timings_ms.total != null ? el("span", { text: `${(r.timings_ms.total / 1000).toFixed(1)} s` }) : null)));
    } else if (!r.sources.length) {
      parts.push(el("div", { class: "notice" }, el("span", { text: "No matching passages were found in the indexed documents." })));
    }
    if (r.sources.length) {
      parts.push(el("section", { class: "sources" },
        el("h2", { text: `Sources (${r.sources.length})` }), ...r.sources.map((x) => sourceCard(x))));
    }
    box.replaceChildren(...parts);
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------------------------------------------------------------- ask */
  function setBusy(busy) {
    state.busy = busy;
    $("ask-btn").disabled = busy; $("question").disabled = busy;
    $("ask-btn").classList.toggle("busy", busy);
    $("ask-btn").querySelector(".btn-label").textContent = busy ? "Thinking…" : "Ask";
  }

  $("ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state.busy) return;
    const question = $("question").value.trim();
    if (!question) { $("question").focus(); banner("Type a question first."); return; }
    banner(""); setBusy(true);
    $("result").replaceChildren(el("div", { class: "skeleton", "aria-label": "Loading answer" },
      el("div", { class: "line", style: "width:35%" }), el("div", { class: "line" }), el("div", { class: "line", style: "width:80%" })));
    const payload = { question, generate: !$("opt-sources-only").checked };
    if ($("opt-mode").value) payload.mode = $("opt-mode").value;
    if ($("opt-doc").value) payload.filename = $("opt-doc").value;
    try {
      renderResult(await api("/api/v1/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }));
    } catch (err) {
      $("result").replaceChildren();
      banner(err.code === "llm_rate_limited" || err.code === "llm_timeout"
        ? err.message + " You can retry, or tick “Sources only”."
        : err.message);
    } finally { setBusy(false); }
  });
  $("question").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("ask-form").requestSubmit(); });
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { $("question").value = c.textContent; $("question").focus(); }));

  /* ---------------------------------------------------------------- assistant (Main Agent) */
  const chat = { history: [], busy: false, n: 0 };

  function agentChip(a) {
    const name = a.agent === "rag" ? "Documents (RAG)" : "GitHub";
    const model = a.model ? ` · ${a.model.provider}:${a.model.model}` : "";
    const text = a.ok ? `${name} ✓ ${(a.duration_ms / 1000).toFixed(1)} s${model}` : `${name} ✗ ${a.error ? a.error.message : "failed"}`;
    return el("span", { class: "agent-chip " + (a.ok ? "ok" : "bad"), text });
  }

  function renderAssistantMessage(r, prefix) {
    const parts = [r.agents_used.length && r.agents_used.every((a) => a.insufficient_evidence)
      ? el("div", { class: "notice", role: "note" }, el("span", { text: r.answer }))
      : renderMarkdown(r.answer, (n) => openSource(n, prefix))];
    if (r.agents_used.length) parts.push(el("div", { class: "agent-row" }, r.agents_used.map(agentChip)));
    else parts.push(el("div", { class: "msg-meta", text: "Answered directly — no sub-agent was needed." }));
    if (r.warnings.length) parts.push(el("div", { class: "warn-list", role: "alert" }, r.warnings.map((w) => el("div", { text: w }))));
    if (r.github.length) parts.push(el("div", { class: "gh-refs" }, el("strong", { text: "GitHub objects read" }),
      r.github.map((g) => safeLink(g.url, g.label, "source-link") || el("span", { text: g.label }))));
    if (r.sources.length) parts.push(el("section", { class: "sources" }, el("h2", { text: `Sources (${r.sources.length})` }), ...r.sources.map((x) => sourceCard(x, prefix))));
    parts.push(el("div", { class: "msg-meta", text: `${r.model.provider}:${r.model.model} · ${(r.duration_ms / 1000).toFixed(1)} s` }));
    return el("div", { class: "msg assistant" }, ...parts);
  }

  function setChatBusy(busy) {
    chat.busy = busy; $("chat-btn").disabled = busy; $("chat-input").disabled = busy;
    $("chat-btn").classList.toggle("busy", busy); $("chat-btn").querySelector(".btn-label").textContent = busy ? "Working…" : "Send";
  }

  $("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (chat.busy) return;
    const message = $("chat-input").value.trim();
    if (!message) { $("chat-input").focus(); return; }
    banner(""); $("chat-input").value = "";
    const log = $("chat-log");
    log.append(el("div", { class: "msg user", text: message }));
    const pending = el("div", { class: "msg assistant skeleton", "aria-label": "The assistant is working" },
      el("div", { class: "line", style: "width:40%" }), el("div", { class: "line" }), el("div", { class: "line", style: "width:70%" }));
    log.append(pending); pending.scrollIntoView({ behavior: "smooth", block: "end" });
    setChatBusy(true);
    try {
      const r = await api("/api/v1/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history: chat.history.slice(-10) }) });
      const node = renderAssistantMessage(r, `chat${++chat.n}`);
      pending.replaceWith(node); node.scrollIntoView({ behavior: "smooth", block: "start" });
      chat.history.push({ role: "user", content: message }, { role: "assistant", content: r.answer });
    } catch (err) {
      pending.replaceWith(el("div", { class: "msg assistant" }, el("div", { class: "warn-list", role: "alert", text: err.message })));
      if (err.code === "llm_not_configured" || err.code === "llm_unreachable" || err.code === "llm_auth") banner(err.message + " Open Settings to check the connections.");
    } finally { setChatBusy(false); $("chat-input").focus(); }
  });
  $("chat-input").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") $("chat-form").requestSubmit(); });
  $("chat-clear").addEventListener("click", () => { chat.history = []; $("chat-log").replaceChildren(); $("chat-input").focus(); });

  /* ---------------------------------------------------------------- add a web page */
  $("url-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = $("url-input").value.trim();
    if (!url) { $("url-input").focus(); return; }
    const btn = $("url-btn"); btn.disabled = true; btn.classList.add("busy");
    const status = el("span", { class: "wait", text: "Downloading and indexing…" });
    $("upload-log").prepend(el("li", {}, el("strong", { text: url }), status));
    try {
      const r = await api("/api/v1/documents/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
      status.className = "ok";
      status.textContent = r.status === "indexed" ? `Indexed (${r.chunk_count} chunks)` : r.status === "replaced" ? `Updated (${r.chunk_count} chunks; old version removed)` : "Already indexed — unchanged";
      $("url-input").value = "";
    } catch (err) { status.className = "bad"; status.textContent = err.message; }
    finally { btn.disabled = false; btn.classList.remove("busy"); }
    await Promise.all([loadDocuments(), refreshStatus()]);
  });

  /* ---------------------------------------------------------------- settings & connections */
  const STATE_LABEL = { ok: "OK", not_configured: "Not configured", not_checked: "Not checked yet", invalid_key: "Key rejected", invalid_token: "Token rejected",
    unreachable: "Unreachable", model_missing: "Model not installed", rate_limited: "Rate limited", error: "Error" };

  function settingsCard(title, rows, state, detail) {
    const cls = state === "ok" ? "ok" : state === "not_checked" ? "warn" : state === "not_configured" ? "warn" : "err";
    return el("div", { class: "s-card" }, el("h3", {}, el("span", { class: "dot " + cls }), title),
      el("dl", {}, rows.flatMap(([k, v]) => [el("dt", { text: k }), el("dd", { text: String(v) })])),
      detail ? el("div", { class: "detail", text: detail }) : null);
  }

  async function loadSettings(check) {
    const btn = $("btn-check");
    if (check) { btn.disabled = true; btn.classList.add("busy"); }
    try {
      const st = await api("/api/v1/status" + (check ? "?check=true" : ""));
      state.status = st;
      const o = st.llm.openrouter, l = st.llm.ollama, g = st.github, a = st.llm.agents;
      const agentRows = ["main", "rag", "github"].map((k) => [`${k} agent`, a[k].provider ? `${a[k].provider}:${a[k].model}` : "unavailable"]);
      $("settings-cards").replaceChildren(
        settingsCard("OpenRouter", [["Key", o.configured ? `configured (${o.key_source})` : "not set"], ["Model", o.model], ["Status", STATE_LABEL[o.state] || o.state]], o.state, o.detail),
        settingsCard("Ollama (local)", [["Server", l.base_url], ["Model", l.model], ["Status", STATE_LABEL[l.state] || l.state], ["Installed", l.installed_models.length ? l.installed_models.join(", ") : "—"]], l.state, l.detail),
        settingsCard("GitHub", [["Token", g.configured ? `configured (${g.key_source})` : "not set"], ["MCP runtime", `${g.mcp_runtime}${g.mcp_runtime_found ? "" : " (not found)"}`], ["Status", STATE_LABEL[g.state] || g.state]], g.state,
          g.detail || (!g.mcp_runtime_found ? "The GitHub MCP server cannot start: install Docker or set GITHUB_MCP_COMMAND." : "")),
        settingsCard("Agents (current choice)", [["Policy", st.llm.policy], ...agentRows], "ok", st.llm.env_files_found.length ? "" : "No .env file was found; only environment variables are used."));
      $("sel-provider").value = st.llm.override.provider || "auto";
      $("sel-or-model").value = st.llm.override.openrouter_model || ""; $("sel-ol-model").value = st.llm.override.ollama_model || "";
      $("ollama-models").replaceChildren(...l.installed_models.map((m) => el("option", { value: m })));
      refreshProviders();
    } catch (e) { banner(e.message); }
    finally { btn.disabled = false; btn.classList.remove("busy"); }
  }
  $("btn-check").addEventListener("click", () => loadSettings(true));
  $("btn-reload").addEventListener("click", async () => {
    try { await api("/api/v1/settings/reload", { method: "POST" }); banner("Settings reloaded from .env / environment.", "info"); await loadSettings(false); await refreshStatus(); }
    catch (e) { banner(e.message); }
  });
  $("llm-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = { provider: $("sel-provider").value, openrouter_model: $("sel-or-model").value.trim() || "default", ollama_model: $("sel-ol-model").value.trim() || "default" };
    try { await api("/api/v1/settings/llm", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); $("llm-form-msg").textContent = "Applied."; banner(""); await loadSettings(false); }
    catch (err) { $("llm-form-msg").textContent = ""; banner(err.message); }
  });

  /* ---------------------------------------------------------------- boot */
  refreshStatus().then(loadDocuments);
})();
