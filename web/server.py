"""
Web version of the Symptom-to-Condition Retrieval System (localhost only).

Uses only the Python standard library (http.server), so no extra packages are needed.
The page in web/static/ calls the JSON API below; all retrieval is done by the same
modules as the command-line demo (ir/*).

    python web/server.py [--port 8000] [--no-external]
    then open http://127.0.0.1:8000

API
  GET  /api/info                       collection statistics
  GET  /api/suggest?q=he               symptom suggestions (prefix matching, Exp 5)
  POST /api/search     {query, mode, severity, autocorrect, method}
                       mode "ranked": vector space results (Exp 4)
                       mode "boolean": AND / OR / NOT exact-match set (Exp 3)
  POST /api/feedback   {query, severity, method, relevant, nonrelevant, feedback_method}
                                                                     Rocchio / Ide (Exp 6)
  POST /api/preprocess {text, method}                               pipeline steps (Exp 2)
  POST /api/pattern    {pattern, type, low, high}                    pattern matching (Exp 5)
  GET  /api/index-compare                                            index structures (Exp 7)
  GET  /api/evaluation                                               saved results (Exp 8)
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WEB_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ir.boolean_model import BooleanQueryError, ast_to_string, parse_query  # noqa: E402
from ir.indexing import InvertedIndex, compare_indexes  # noqa: E402
from ir.pattern_matching import levenshtein_matrix, max_edits_for  # noqa: E402
from ir.preprocessing import extract_noun_groups, pipeline_steps, preprocess  # noqa: E402
from ir.query_expansion import METHODS, added_terms, relevance_feedback  # noqa: E402
from ir.search_engine import DEFAULT_SEVERITY_BOOST, SearchEngine  # noqa: E402

DISCLAIMER = ("This is an academic IR project and not a medical diagnosis. "
              "Consult a doctor for health concerns.")
TOP_K = 5
MAX_BODY = 20_000          # bytes; queries are short
STATIC_FILES = {           # whitelist: URL path -> (file, content type)
    "/": ("static/index.html", "text/html; charset=utf-8"),
    "/static/style.css": ("static/style.css", "text/css; charset=utf-8"),
    "/static/app.js": ("static/app.js", "application/javascript; charset=utf-8"),
    "/eval/map_ndcg_by_config.png": ("../eval/map_ndcg_by_config.png", "image/png"),
    "/eval/precision_recall_curve.png": ("../eval/precision_recall_curve.png", "image/png"),
}
METHOD_NAMES = {"rocchio": "Rocchio", "ide_regular": "Ide Regular", "ide_dec_hi": "Ide Dec-Hi"}


class ApiError(Exception):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# API logic (independent of HTTP, so it is easy to test)
# ---------------------------------------------------------------------------
class SymptomApi:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()  # the models build caches lazily; serialise access
        self._index_compare = None
        # warm up lazily-loaded NLTK resources (WordNet, POS tagger) once
        preprocess("warm up", "lemma")
        extract_noun_groups("warm up text")
        engine.correct("warm up")

    # -- helpers ------------------------------------------------------------
    def _model(self, body):
        method = body.get("method", "stem")
        if method not in ("stem", "lemma"):
            raise ApiError("method must be 'stem' or 'lemma'")
        boost = DEFAULT_SEVERITY_BOOST if body.get("severity") else 0.0
        return self.engine.vsm(method, boost)

    @staticmethod
    def _text(body, key, limit=1000):
        value = body.get(key, "")
        if not isinstance(value, str) or not value.strip():
            raise ApiError(f"'{key}' must be a non-empty string")
        return value.strip()[:limit]

    def _result(self, model, rank, result, query_terms):
        doc = self.engine.documents[result.doc_id]
        full, partial = model.matched_symptoms(result.doc_id, query_terms)
        return {
            "rank": rank, "doc_id": result.doc_id, "disease": doc.disease,
            "score": round(result.score, 4), "also_called": doc.also_called[:6],
            "matched_symptoms": full, "partial_symptoms": partial,
            "contributions": [[t, round(c, 4)] for t, c in result.contributions[:8]],
            "description": doc.description, "precautions": doc.precautions,
            "symptoms": doc.symptoms,
        }

    def _corrected_query(self, body):
        query = self._text(body, "query")
        corrections = {}
        if body.get("autocorrect", True):
            query, corrections = self.engine.correct(query)
        return query, corrections

    # -- endpoints ----------------------------------------------------------
    def info(self):
        e = self.engine
        return {
            "documents": len(e.documents), "symptoms": len(e.symptoms),
            "vocabulary": len(e.vsm().idf), "enrichment": e.enrichment,
            "feedback_methods": METHOD_NAMES, "disclaimer": DISCLAIMER,
        }

    def suggest(self, prefix):
        prefix = prefix.strip().lower()[:40]
        return {"suggestions": self.engine.suggest_symptoms(prefix, 8) if prefix else []}

    def search(self, body):
        mode = body.get("mode", "ranked")
        if mode == "boolean":
            return self.boolean_search(body)
        if mode != "ranked":
            raise ApiError("mode must be 'ranked' or 'boolean'")
        model = self._model(body)
        original = self._text(body, "query")
        query, corrections = self._corrected_query(body)
        terms = preprocess(query, model.method)
        ranking = model.search(query)
        qvec = model.query_vector(query)
        return {
            "mode": "ranked", "query": query, "original_query": original, "corrections": corrections,
            "terms": terms, "noun_groups": extract_noun_groups(query),
            "query_vector": [{"term": t, "df": model.index.document_frequency(t),
                              "idf": round(model.idf[t], 4), "weight": round(w, 4)}
                             for t, w in sorted(qvec.items(), key=lambda x: -x[1])],
            "total_matches": len(ranking),
            "results": [self._result(model, i, r, terms)
                        for i, r in enumerate(ranking[:TOP_K], start=1)],
            "disclaimer": DISCLAIMER,
        }

    def boolean_search(self, body):
        """Exact-match Boolean retrieval (AND, OR, NOT, parentheses, "phrases").
        Returns an unranked set, listed in document order (alphabetical)."""
        method = body.get("method", "stem")
        if method not in ("stem", "lemma"):
            raise ApiError("method must be 'stem' or 'lemma'")
        model = self.engine.boolean(method)
        original = self._text(body, "query")
        query, corrections = self._corrected_query(body)
        try:
            tree = parse_query(query, method)
            hits = model.search(query)
        except BooleanQueryError as err:
            raise ApiError(f"Boolean query error: {err}")
        terms = set(preprocess(query, method))
        results = []
        for doc_id in hits:
            doc = self.engine.documents[doc_id]
            results.append({
                "doc_id": doc_id, "disease": doc.disease, "also_called": doc.also_called[:6],
                "matched_terms": sorted(terms & set(model.doc_tokens[doc_id])),
                "description": doc.description, "precautions": doc.precautions,
            })
        return {
            "mode": "boolean", "query": query, "original_query": original,
            "corrections": corrections, "parsed": ast_to_string(tree),
            "unknown_terms": sorted(set(model.unknown_terms)),
            "total_matches": len(hits), "results": results, "disclaimer": DISCLAIMER,
        }

    def feedback(self, body):
        model = self._model(body)
        query, _ = self._corrected_query(body)
        method = body.get("feedback_method", "rocchio")
        if method not in METHODS:
            raise ApiError(f"feedback_method must be one of {list(METHODS)}")
        valid = set(range(len(self.engine.documents)))
        relevant = [int(i) for i in body.get("relevant", []) if int(i) in valid]
        nonrelevant = [int(i) for i in body.get("nonrelevant", [])
                       if int(i) in valid and int(i) not in relevant]
        if not relevant:
            raise ApiError("mark at least one result as relevant")
        before = model.search(query)
        rank_of = {r.doc_id: i for i, r in enumerate(before, start=1)}
        nonrelevant.sort(key=lambda d: rank_of.get(d, 10 ** 6))  # Dec-Hi needs rank order
        qvec = model.query_vector(query)
        new_q = relevance_feedback(model, qvec, relevant, nonrelevant, method)
        after = model.rank_vector(new_q)

        def row(rank, r):
            old = rank_of.get(r.doc_id)
            return {"rank": rank, "doc_id": r.doc_id,
                    "disease": self.engine.documents[r.doc_id].disease,
                    "score": round(r.score, 4),
                    "move": None if old is None else old - rank}

        return {
            "method": METHOD_NAMES[method],
            "before": [row(i, r) for i, r in enumerate(before[:TOP_K], start=1)],
            "after": [row(i, r) for i, r in enumerate(after[:TOP_K], start=1)],
            "added_terms": [[t, round(w, 4)] for t, w in added_terms(qvec, new_q, 8)],
        }

    def preprocess_steps(self, body):
        text = self._text(body, "text", 500)
        method = body.get("method", "stem")
        steps = pipeline_steps(text, method if method in ("stem", "lemma") else "stem")
        out = []
        for name, value in steps.items():
            if isinstance(value, list):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            out.append({"step": name, "value": value})
        return {"steps": out}

    def pattern(self, body):
        pm = self.engine.matcher
        kind = body.get("type", "wildcard")
        if kind == "range":
            low = self._text(body, "low", 40).lower()
            high = self._text(body, "high", 40).lower()
            return {"type": kind, "matches": pm.range_search(low, high)}
        pattern = self._text(body, "pattern", 60).lower()
        runs = {
            "exact": pm.exact_match, "prefix": pm.prefix_match, "suffix": pm.suffix_match,
            "substring": pm.substring_match, "wildcard": pm.wildcard_search,
            "regex": pm.regex_search,
        }
        if kind == "fuzzy":
            hits = pm.fuzzy_search(pattern)
            result = {"type": kind, "allowed_distance": max_edits_for(pattern),
                      "matches": [f"{w} (d={d})" for w, d in hits[:15]]}
            if hits:
                best = hits[0][0]
                result["dp"] = {"source": pattern, "target": best,
                                "table": levenshtein_matrix(pattern, best)}
            return result
        if kind == "suggest":
            return {"type": kind, "matches": self.engine.suggest_symptoms(pattern, 15)}
        if kind not in runs:
            raise ApiError(f"unknown pattern type '{kind}'")
        try:
            matches = runs[kind](pattern)
        except re.error as err:
            raise ApiError(f"invalid regular expression: {err}")
        return {"type": kind, "matches": matches}

    def index_compare(self):
        if self._index_compare is None:
            index = InvertedIndex([preprocess(d.text, "none") for d in self.engine.documents])
            table, stats = compare_indexes(
                index, ["fever", "head", "itis", "rrhoe", "pain", "skin", "ache", "xyz"], repeat=100)
            self._index_compare = {
                "stats": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()},
                "rows": table.to_dict(orient="records"),
            }
        return self._index_compare

    @staticmethod
    def evaluation():
        def read(name):
            path = os.path.join(PROJECT_ROOT, "eval", name)
            if not os.path.exists(path):
                return []
            with open(path, encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        return {"results": read("results.csv"), "data_sources": read("results_data_sources.csv"),
                "severity_sweep": read("severity_sweep.csv")}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def make_handler(api):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SymptomIR/1.0"

        def log_message(self, fmt, *args):  # quieter console: only API calls
            if self.path.startswith("/api/"):
                sys.stderr.write(f"  {self.command} {self.path.split('?')[0]}\n")

        def _send(self, status, payload, content_type="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ApiError("request too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise ApiError("body must be JSON")
            if not isinstance(body, dict):
                raise ApiError("body must be a JSON object")
            return body

        def _api(self, fn, *args):
            try:
                with api.lock:
                    self._send(HTTPStatus.OK, fn(*args))
            except ApiError as err:
                self._send(err.status, {"error": str(err)})
            except (ValueError, TypeError) as err:
                self._send(HTTPStatus.BAD_REQUEST, {"error": f"bad request: {err}"})
            except Exception as err:  # keep the server alive, report the problem
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"server error: {err}"})

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in STATIC_FILES:
                rel, ctype = STATIC_FILES[url.path]
                path = os.path.normpath(os.path.join(WEB_DIR, rel))
                if not os.path.exists(path):
                    return self._send(HTTPStatus.NOT_FOUND, {"error": "file not found"})
                with open(path, "rb") as fh:
                    return self._send(HTTPStatus.OK, fh.read(), ctype)
            routes = {
                "/api/info": lambda: self._api(api.info),
                "/api/suggest": lambda: self._api(api.suggest, parse_qs(url.query).get("q", [""])[0]),
                "/api/index-compare": lambda: self._api(api.index_compare),
                "/api/evaluation": lambda: self._api(api.evaluation),
            }
            if url.path in routes:
                return routes[url.path]()
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            routes = {
                "/api/search": api.search, "/api/feedback": api.feedback,
                "/api/preprocess": api.preprocess_steps, "/api/pattern": api.pattern,
            }
            fn = routes.get(urlparse(self.path).path)
            if fn is None:
                return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            try:
                body = self._body()
            except ApiError as err:
                return self._send(err.status, {"error": str(err)})
            self._api(fn, body)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Web version of the retrieval system")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-external", action="store_true",
                        help="use only the original Kaggle documents")
    parser.add_argument("--open", action="store_true", help="open the browser automatically")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    print("Loading documents and building the index ...")
    engine = SearchEngine(medlineplus=not args.no_external, symptom2disease=not args.no_external)
    api = SymptomApi(engine)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(api))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Symptom-to-Condition Retrieval System running at {url}  (Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
