"use strict";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);

/** Create an element; text is always set with textContent (no HTML injection). */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function setStatus(text, isError = false) {
  const status = $("#status");
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function table(headers, rows, numeric = []) {
  return el("div", { class: "table-wrap" },
    el("table", { class: "data" },
      el("thead", {}, el("tr", {}, headers.map((h, i) =>
        el("th", { class: numeric.includes(i) ? "num" : null, text: h })))),
      el("tbody", {}, rows.map((row) => el("tr", {}, row.map((cell, i) =>
        el("td", { class: numeric.includes(i) ? "num" : null },
          cell instanceof Node ? cell : String(cell)))))),
    ));
}

// ---------------------------------------------------------------------------
// tabs
// ---------------------------------------------------------------------------
let labLoaded = false;
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t === tab);
    t.setAttribute("aria-selected", t === tab ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = panel.id !== `tab-${tab.dataset.tab}`;
  });
  if (tab.dataset.tab === "lab" && !labLoaded) {
    labLoaded = true;
    loadEvaluation();
    $("#pre-form").requestSubmit();
    $("#pat-form").requestSubmit();
  }
}));

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------
const state = { query: "", results: [], marks: new Map() };   // marks: doc_id -> "rel" | "nonrel"

const MODE_HINTS = {
  ranked: "Ranked: describe symptoms in plain words; the top 5 conditions are ranked by TF-IDF cosine similarity.",
  boolean: "Boolean: exact matching with AND, OR, NOT, parentheses and \"quoted phrases\". Words without an operator are combined with AND. Results are an unranked set.",
};
const MODE_PLACEHOLDERS = {
  ranked: "e.g. high fever with chills, sweating and a headache",
  boolean: "e.g. fever AND rash NOT cough",
};

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function setMode(mode) {
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  applyMode();
}

/** Update hint, placeholder and options for the selected answer mode. */
function applyMode() {
  const mode = currentMode();
  $("#mode-hint").textContent = MODE_HINTS[mode];
  $("#query").placeholder = MODE_PLACEHOLDERS[mode];
  const severity = $("#opt-severity");
  severity.disabled = mode === "boolean";  // severity weights only affect ranking
  severity.closest("label").classList.toggle("disabled", severity.disabled);
  severity.closest("label").title = severity.disabled ? "Only used in ranked mode" : "";
}

function searchOptions() {
  return {
    mode: currentMode(),
    severity: $("#opt-severity").checked,
    method: $("#opt-method").value,
    autocorrect: $("#opt-autocorrect").checked,
  };
}

function correctionsNotice(data, typedQuery) {
  const fixes = Object.entries(data.corrections);
  if (!fixes.length) return null;
  return el("div", { class: "notice" },
    "Spelling corrected: ",
    el("b", { text: fixes.map(([a, b]) => `${a} → ${b}`).join(", ") }),
    ". ",
    el("button", {
      class: "linklike", type: "button", text: `Search for "${typedQuery}" instead`,
      onclick: () => runSearch(typedQuery, { autocorrect: false }),
    }));
}

async function runSearch(query, overrides = {}) {
  query = query.trim();
  if (!query) { setStatus("Type some symptoms first."); return; }
  const button = $("#search-btn");
  button.disabled = true;
  setStatus("Searching ...");
  try {
    const data = await api("/api/search", { query, ...searchOptions(), ...overrides });
    renderResults(data, query);
    const modeParam = data.mode === "boolean" ? "&mode=boolean" : "";
    history.replaceState(null, "", `?q=${encodeURIComponent(query)}${modeParam}`);  // bookmarkable
    if (data.mode === "boolean") {
      setStatus(data.total_matches
        ? `${data.total_matches} condition(s) satisfy the query (unranked set, listed alphabetically).`
        : "");
    } else {
      setStatus(data.total_matches
        ? `${data.total_matches} conditions matched; top ${data.results.length} shown.`
        : "");
    }
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
}

function renderResults(data, typedQuery) {
  state.query = data.query;
  state.results = data.mode === "boolean" ? [] : data.results;
  state.marks = new Map();
  $("#results-area").hidden = false;
  $("#fb-output").replaceChildren();
  if (data.mode === "boolean") return renderBoolean(data, typedQuery);
  $("#explain").hidden = false;

  // spelling corrections + recognised phrases
  const info = $("#query-info");
  info.replaceChildren();
  const notice = correctionsNotice(data, typedQuery);
  if (notice) info.append(notice);
  if (data.noun_groups.length) {
    info.append(el("div", { class: "meta" }, "Recognised symptom phrases: ",
      data.noun_groups.map((g) => el("span", { class: "pill", text: g }))));
  }

  // result cards
  const list = $("#results");
  list.replaceChildren();
  if (!data.results.length) {
    list.append(el("li", { class: "result" },
      el("p", { class: "desc", text: "No condition matches these symptoms. Try other words, or start typing a symptom to see suggestions." })));
  }
  const top = data.results.length ? data.results[0].score : 1;
  data.results.forEach((r) => list.append(resultCard(r, top)));

  // explanation
  $("#explain-body").replaceChildren(
    el("p", { class: "meta" }, "Query terms after preprocessing: ",
      data.terms.length ? data.terms.map((t) => el("span", { class: "pill", text: t })) : "(none)"),
    data.query_vector.length
      ? table(["Term", "df", "idf = log10(N/df)", "Query weight"],
        data.query_vector.map((q) => [q.term, q.df, q.idf.toFixed(3), q.weight.toFixed(3)]), [1, 2, 3])
      : el("p", { class: "muted", text: "None of the query terms occur in the collection." }),
    el("p", { class: "muted", text: "Score = cosine similarity between the query vector and each document's TF-IDF vector. The bars on each result show how much each term contributed; the contributions add up to the score." }),
  );
  $("#feedback-card").hidden = !data.results.length;
  $("#disclaimer").textContent = `Note: ${data.disclaimer}`;
}

function resultCard(r, topScore) {
  const card = el("li", { class: "result", "data-doc": r.doc_id });
  const maxContribution = r.contributions.length ? r.contributions[0][1] : 1;
  const partial = r.partial_symptoms.map((s) => el("span", { class: "pill", text: s }));
  const full = r.matched_symptoms.map((s) => el("span", { class: "pill full", text: s }));

  const toggle = (kind) => () => {
    const current = state.marks.get(r.doc_id);
    if (current === kind) state.marks.delete(r.doc_id); else state.marks.set(r.doc_id, kind);
    const mark = state.marks.get(r.doc_id);
    card.classList.toggle("is-rel", mark === "rel");
    card.classList.toggle("is-nonrel", mark === "nonrel");
    relBtn.classList.toggle("on-rel", mark === "rel");
    nonBtn.classList.toggle("on-nonrel", mark === "nonrel");
    relBtn.setAttribute("aria-pressed", mark === "rel" ? "true" : "false");
    nonBtn.setAttribute("aria-pressed", mark === "nonrel" ? "true" : "false");
  };
  const relBtn = el("button", { type: "button", "aria-pressed": "false", text: "✓ Relevant", onclick: toggle("rel") });
  const nonBtn = el("button", { type: "button", "aria-pressed": "false", text: "✗ Not relevant", onclick: toggle("nonrel") });

  // build through el() so null parts (e.g. no alternative names) are skipped
  card.append(...el("div", {},
    el("div", { class: "result-head" },
      el("span", { class: "rank", text: r.rank }),
      el("h3", { text: r.disease }),
      el("span", { class: "score", title: "cosine similarity" },
        el("span", { class: "scorebar" }, el("span", { style: `width:${Math.max(4, (r.score / topScore) * 100)}%` })),
        r.score.toFixed(4))),
    r.also_called.length ? el("div", { class: "aka", text: `Also known as: ${r.also_called.join(", ")}` }) : null,
    el("div", { class: "result-body" },
      (full.length || partial.length) ? el("div", {},
        el("div", { class: "row-label", text: "Matched symptoms" }), full, partial) : null,
      el("div", {},
        el("div", { class: "row-label", text: "Matched terms (contribution to score)" }),
        el("div", { class: "terms" }, r.contributions.map(([term, value]) => [
          el("span", { text: term }),
          el("span", { class: "bar" }, el("span", { style: `width:${(value / maxContribution) * 100}%` })),
          el("span", { class: "val", text: value.toFixed(3) }),
        ]))),
      r.description ? el("p", { class: "desc", text: r.description }) : null,
      r.precautions.length ? el("div", {},
        el("div", { class: "row-label", text: "Precautions" }),
        el("ul", { class: "prec" }, r.precautions.map((p) => el("li", { text: p })))) : null),
    el("div", { class: "fb-toggle" }, relBtn, nonBtn),
  ).childNodes);
  return card;
}

/** Boolean mode: an unranked set, so no rank numbers, scores or feedback. */
function renderBoolean(data, typedQuery) {
  $("#explain").hidden = true;
  $("#feedback-card").hidden = true;
  const info = $("#query-info");
  info.replaceChildren();
  const notice = correctionsNotice(data, typedQuery);
  if (notice) info.append(notice);
  info.append(el("div", { class: "meta" }, "Parsed query: ", el("span", { class: "parsed", text: data.parsed })));
  if (data.unknown_terms.length) {
    info.append(el("div", { class: "meta" }, "Not found in any document: ",
      data.unknown_terms.map((t) => el("span", { class: "pill", text: t }))));
  }

  const list = $("#results");
  list.replaceChildren();
  if (!data.results.length) {
    list.append(el("li", { class: "result" }, el("p", { class: "desc",
      text: "No condition satisfies this Boolean query. Boolean retrieval is exact matching: try OR instead of AND, remove a term, or switch to Ranked mode." })));
  }
  data.results.forEach((r) => list.append(el("li", { class: "result" },
    el("div", { class: "result-head" },
      el("span", { class: "rank set", title: "matches the query", text: "✓" }),
      el("h3", { text: r.disease })),
    r.also_called.length ? el("div", { class: "aka", text: `Also known as: ${r.also_called.join(", ")}` }) : null,
    el("div", { class: "result-body" },
      el("div", {},
        el("div", { class: "row-label", text: "Matched terms" }),
        r.matched_terms.length ? r.matched_terms.map((t) => el("span", { class: "pill full", text: t }))
          : el("span", { class: "muted", text: "(matched through NOT only)" })),
      r.description ? el("p", { class: "desc", text: r.description }) : null,
      r.precautions.length ? el("div", {},
        el("div", { class: "row-label", text: "Precautions" }),
        el("ul", { class: "prec" }, r.precautions.map((p) => el("li", { text: p })))) : null),
  )));
  $("#disclaimer").textContent = `Note: ${data.disclaimer}`;
}

document.querySelectorAll('input[name="mode"]').forEach((radio) => radio.addEventListener("change", () => {
  applyMode();
  if ($("#query").value.trim()) runSearch($("#query").value);
}));

$("#search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch($("#query").value);
});
$("#query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#search-form").requestSubmit();
  }
});
document.querySelectorAll(".example").forEach((chip) => chip.addEventListener("click", () => {
  setMode(chip.dataset.mode || "ranked");  // Boolean examples switch to Boolean mode
  $("#query").value = chip.textContent;
  runSearch(chip.textContent);
}));
["#opt-severity", "#opt-method"].forEach((sel) => $(sel).addEventListener("change", () => {
  if (state.query) runSearch($("#query").value);
}));

// symptom suggestions for the word being typed (prefix matching)
let suggestTimer = null;
$("#query").addEventListener("input", () => {
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(async () => {
    const box = $("#suggestions");
    const text = $("#query").value;
    const match = text.match(/([A-Za-z]{2,})$/);
    if (!match) { box.replaceChildren(); return; }
    try {
      const data = await api(`/api/suggest?q=${encodeURIComponent(match[1])}`);
      box.replaceChildren(...data.suggestions.map((s) => el("button", {
        type: "button", class: "chip", text: s,
        onclick: () => {
          $("#query").value = text.slice(0, text.length - match[1].length) + s + ", ";
          box.replaceChildren();
          $("#query").focus();
        },
      })));
    } catch { box.replaceChildren(); }
  }, 180);
});

// relevance feedback
$("#fb-btn").addEventListener("click", async () => {
  const relevant = [...state.marks].filter(([, m]) => m === "rel").map(([id]) => id);
  const output = $("#fb-output");
  if (!relevant.length) {
    output.replaceChildren(el("p", { class: "status error", text: "Mark at least one result as relevant first." }));
    return;
  }
  // unmarked results shown to the user count as non-relevant (as in the CLI)
  const nonrelevant = state.results.map((r) => r.doc_id).filter((id) => state.marks.get(id) !== "rel");
  const button = $("#fb-btn");
  button.disabled = true;
  try {
    const data = await api("/api/feedback", {
      query: state.query, ...searchOptions(), autocorrect: false,
      relevant, nonrelevant, feedback_method: $("#fb-method").value,
    });
    const rows = [];
    for (let i = 0; i < Math.max(data.before.length, data.after.length); i++) {
      const b = data.before[i], a = data.after[i];
      let move = "";
      if (a) {
        move = a.move === null ? el("span", { class: "move-up", text: "new" })
          : a.move > 0 ? el("span", { class: "move-up", text: `▲ ${a.move}` })
          : a.move < 0 ? el("span", { class: "move-down", text: `▼ ${-a.move}` }) : "=";
      }
      rows.push([i + 1, b ? b.disease : "", b ? b.score.toFixed(3) : "",
        a ? a.disease : "", a ? a.score.toFixed(3) : "", move]);
    }
    output.replaceChildren(
      table(["Rank", "Before", "Score", `After ${data.method}`, "Score", "Move"], rows, [0, 2, 4]),
      data.added_terms.length ? el("p", { class: "meta" }, "Terms added to the query: ",
        data.added_terms.map(([t, w]) => el("span", { class: "pill", text: `${t} (${w.toFixed(3)})` }))) : null,
    );
  } catch (err) {
    output.replaceChildren(el("p", { class: "status error", text: err.message }));
  } finally {
    button.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// lab experiments
// ---------------------------------------------------------------------------
function formatValue(value) {
  if (Array.isArray(value)) {
    return value.map((v) => (Array.isArray(v) ? `${v[0]}/${v[1]}` : v)).join(Array.isArray(value[0]) ? "  " : ", ");
  }
  return String(value);
}

$("#pre-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("#pre-out");
  try {
    const data = await api("/api/preprocess", { text: $("#pre-text").value, method: $("#pre-method").value });
    out.replaceChildren(el("dl", { class: "steps" },
      data.steps.map((s) => [el("dt", { text: s.step }), el("dd", { text: formatValue(s.value) || "(none)" })])));
  } catch (err) {
    out.replaceChildren(el("p", { class: "status error", text: err.message }));
  }
});

const PATTERN_EXAMPLES = {
  wildcard: "head*", prefix: "fev", suffix: "ache", substring: "rrhoe", exact: "fever",
  regex: "^dia.*", fuzzy: "feaver", range: "fatigue", suggest: "he",
};
$("#pat-type").addEventListener("change", () => {
  const type = $("#pat-type").value;
  $("#pat-text").value = PATTERN_EXAMPLES[type];
  $("#pat-high").hidden = type !== "range";
  if (type === "range") $("#pat-high").value = "fever";
  $("#pat-form").requestSubmit();
});

$("#pat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const out = $("#pat-out");
  const type = $("#pat-type").value;
  const body = type === "range"
    ? { type, low: $("#pat-text").value, high: $("#pat-high").value }
    : { type, pattern: $("#pat-text").value };
  try {
    const data = await api("/api/pattern", body);
    const parts = [el("p", { class: "meta", text: `${data.matches.length} match(es)` +
      (data.allowed_distance ? ` within edit distance ${data.allowed_distance}` : "") }),
    el("div", { class: "chips" }, data.matches.slice(0, 80).map((m) => el("span", { class: "pill", text: m })))];
    if (data.dp) parts.push(dpTable(data.dp));
    out.replaceChildren(...parts);
  } catch (err) {
    out.replaceChildren(el("p", { class: "status error", text: err.message }));
  }
});

function dpTable({ source, target, table: d }) {
  const header = el("tr", {}, el("th"), el("th", { text: "ε" }), [...target].map((c) => el("th", { text: c })));
  const rows = d.map((row, i) => el("tr", {},
    el("th", { text: i === 0 ? "ε" : source[i - 1] }),
    row.map((v, j) => el("td", { class: i === d.length - 1 && j === row.length - 1 ? "final" : null, text: v }))));
  return el("div", {},
    el("p", { class: "meta", text: `Levenshtein DP table for "${source}" → "${target}" (distance ${d[d.length - 1][d[0].length - 1]})` }),
    el("div", { class: "table-wrap" }, el("table", { class: "dp" }, header, rows)));
}

$("#idx-btn").addEventListener("click", async () => {
  const out = $("#idx-out");
  const button = $("#idx-btn");
  button.disabled = true;
  out.replaceChildren(el("p", { class: "muted", text: "Timing 100 lookups per pattern ..." }));
  try {
    const data = await api("/api/index-compare");
    const s = data.stats;
    out.replaceChildren(
      el("p", { class: "meta", text: `Vocabulary ${s.vocabulary_size} terms · suffix array ${s.suffix_array_length} entries (built in ${s.suffix_array_build_ms} ms) · suffix trie ${s.suffix_trie_nodes} nodes (built in ${s.suffix_trie_build_ms} ms)` }),
      table(["Pattern", "Matching terms", "Docs", "Same result", "Scan µs", "Suffix array µs", "Suffix trie µs"],
        data.rows.map((r) => [r.pattern, r.matching_terms || "—", r.num_docs, r.same_result ? "yes" : "NO",
          r.inverted_scan_us, r.suffix_array_us, r.suffix_trie_us]), [2, 4, 5, 6]),
    );
  } catch (err) {
    out.replaceChildren(el("p", { class: "status error", text: err.message }));
  } finally {
    button.disabled = false;
  }
});

async function loadEvaluation() {
  const out = $("#eval-out");
  try {
    const data = await api("/api/evaluation");
    if (!data.results.length) {
      out.replaceChildren(el("p", { class: "muted", text: "No saved results yet. Run python eval/run_evaluation.py first." }));
      return;
    }
    const f = (v) => Number(v).toFixed(3);
    const parts = [];
    if (data.data_sources.length) {
      parts.push(el("h3", { class: "row-label", text: "Effect of the data sources (VSM + stemming)" }),
        table(["Documents", "Test set", "P@1", "MAP", "MRR", "NDCG@5"],
          data.data_sources.map((r) => [r.data, r.test_set, f(r["P@1"]), f(r.MAP), f(r.MRR), f(r["NDCG@5"])]), [2, 3, 4, 5]));
    }
    parts.push(el("h3", { class: "row-label", style: "margin-top:16px", text: "Configurations" }),
      table(["Test set", "Configuration", "P@1", "P@5", "MAP", "MRR", "NDCG@5"],
        data.results.map((r) => [r.test_set || "", r.configuration, f(r["P@1"]), f(r["P@5"]), f(r.MAP), f(r.MRR), f(r["NDCG@5"])]),
        [2, 3, 4, 5, 6]));
    parts.push(el("div", { class: "charts" },
      el("img", { src: "/eval/map_ndcg_by_config.png", alt: "Bar chart of MAP and NDCG@5 for each configuration" }),
      el("img", { src: "/eval/precision_recall_curve.png", alt: "Interpolated precision-recall curves" })));
    out.replaceChildren(...parts);
  } catch (err) {
    out.replaceChildren(el("p", { class: "status error", text: err.message }));
  }
}

// ---------------------------------------------------------------------------
// startup
// ---------------------------------------------------------------------------
api("/api/info").then((info) => {
  const e = info.enrichment;
  $("#footer-info").textContent =
    `${info.documents} conditions · ${info.symptoms} symptoms · ${info.vocabulary} index terms` +
    (e.expanded_docs ? ` · ${e.expansion_texts} user descriptions (Symptom2Disease) · alternative names from MedlinePlus.gov` : "");
}).catch(() => {
  $("#footer-info").textContent = "Server not reachable - start it with: python web/server.py";
});

// ?q=... runs a search on load, &mode=boolean selects Boolean mode, ?tab=lab opens the lab tab
const params = new URLSearchParams(location.search);
setMode(params.get("mode") === "boolean" ? "boolean" : "ranked");
if (params.get("tab") === "lab") document.querySelector('.tab[data-tab="lab"]').click();
if (params.get("q")) {
  $("#query").value = params.get("q");
  runSearch(params.get("q"));
}
