"""
Symptom-to-Condition Retrieval System - command-line demo.

    python main.py                              interactive search loop
    python main.py --search "fever and chills"  one-shot ranked search
    python main.py --preprocess "text"          Exp 2: show every preprocessing step
    python main.py --boolean "fever AND rash NOT cough"   Exp 3
    python main.py --export-matrix out.csv      Exp 3: incidence matrix to CSV
    python main.py --tfidf "chest pain"         Exp 4: score breakdown
    python main.py --pattern "head*"            Exp 5 (also --pattern-type, --range, --fuzzy)
    python main.py --feedback "yellow skin" --relevant 3 --nonrelevant 1 --method rocchio  Exp 6
    python main.py --index-compare              Exp 7
    python main.py --evaluate                   Exp 8
    python main.py --no-external ...            use only the original Kaggle dataset
"""

import argparse
import os
import re
import sys
import textwrap

from ir.boolean_model import BooleanQueryError, ast_to_string, parse_query
from ir.indexing import InvertedIndex, compare_indexes
from ir.pattern_matching import levenshtein_matrix, max_edits_for
from ir.preprocessing import extract_noun_groups, pipeline_steps, preprocess
from ir.query_expansion import METHODS, added_terms, relevance_feedback
from ir.search_engine import DEFAULT_SEVERITY_BOOST, SearchEngine

DISCLAIMER = ("This is an academic IR project and not a medical diagnosis. "
              "Consult a doctor for health concerns.")
TOP_K = 5
METHOD_NAMES = {"rocchio": "Rocchio", "ide_regular": "Ide Regular", "ide_dec_hi": "Ide Dec-Hi"}
WIDTH = 92


def banner():
    print("=" * WIDTH)
    print(" Symptom-to-Condition Retrieval System (Information Retrieval lab project)")
    print("=" * WIDTH)


def wrap(label, text, indent=4, label_width=17):
    prefix = " " * indent + f"{label:<{label_width}}: "
    return textwrap.fill(text, WIDTH, initial_indent=prefix,
                         subsequent_indent=" " * len(prefix))


def ask(prompt, default=""):
    try:
        answer = input(prompt).strip()
    except EOFError:
        raise KeyboardInterrupt
    return answer or default


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_vsm_results(engine, model, query_text, results):
    terms = preprocess(query_text, model.method)
    if not results:
        print("  No condition matches these symptoms. Try other words or 'suggest <prefix>'.")
        return
    for rank, r in enumerate(results, start=1):
        doc = engine.documents[r.doc_id]
        full, partial = model.matched_symptoms(r.doc_id, terms)
        print(f"\n  #{rank}  {doc.disease:<45} score {r.score:.4f}")
        if doc.also_called:
            print(wrap("Also known as", ", ".join(doc.also_called[:5])))
        if full or partial:
            print(wrap("Matched symptoms", ", ".join(full + [p + " (partial)" for p in partial])))
        top_terms = ", ".join(f"{t} ({c:.3f})" for t, c in r.contributions[:6])
        print(wrap("Matched terms", top_terms))
        description = doc.description if len(doc.description) <= 260 else doc.description[:257] + "..."
        print(wrap("Description", description))
        print(wrap("Precautions", "; ".join(doc.precautions)))
    print_disclaimer()


def print_disclaimer():
    """Shown once a search has been executed and its results printed."""
    print("\n  " + "-" * (WIDTH - 2))
    print(textwrap.fill("NOTE: " + DISCLAIMER, WIDTH, initial_indent="  ",
                        subsequent_indent="        "))
    print("  " + "-" * (WIDTH - 2))


def print_side_by_side(engine, old, new, method_label, n=TOP_K):
    old_rank = {r.doc_id: i for i, r in enumerate(old, start=1)}
    print(f"\n  {'Rank':<5}{'Before feedback':<38}{'After ' + method_label:<38}Move")
    print("  " + "-" * 86)
    for i in range(n):
        left = right = move = ""
        if i < len(old):
            left = f"{engine.documents[old[i].doc_id].disease[:28]:<29}{old[i].score:.3f}"
        if i < len(new):
            doc_id = new[i].doc_id
            right = f"{engine.documents[doc_id].disease[:28]:<29}{new[i].score:.3f}"
            before = old_rank.get(doc_id)
            move = "new" if before is None else ("=" if before == i + 1 else f"{before - (i + 1):+d}")
        print(f"  {i + 1:<5}{left:<38}{right:<38}{move}")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
HELP = """
  Type your symptoms in plain words, e.g.  "high fever, chills and a headache".
    suggest <prefix>  or  <prefix>*   -> symptom suggestions (prefix matching), e.g. 'he*'
    severity on|off                   -> severity-weighted VSM scoring (default off)
    help / quit
  After each query you choose the answer mode:
    [R]anked  - vector space model (TF-IDF + cosine similarity), top 5 with scores
    [B]oolean - exact matching with AND, OR, NOT, parentheses and "quoted phrases";
                words without an operator are combined with AND
"""


def spelling_check(engine, text):
    """Ask 'Did you mean ...?' for every unknown word; return the corrected text."""
    for word, options in engine.misspellings(text).items():
        listed = ", ".join(f"[{i}] {o}" for i, o in enumerate(options, start=1))
        answer = ask(f"  Did you mean {listed} for '{word}'? (number / Enter = 1 / n = keep) ", "1")
        if answer.lower().startswith("n"):
            continue
        choice = options[int(answer) - 1] if answer.isdigit() and 1 <= int(answer) <= len(options) else options[0]
        text = re.sub(rf"\b{re.escape(word)}\b", choice, text, flags=re.IGNORECASE)
        print(f"  -> using '{choice}'")
    return text


def parse_numbers(answer, limit):
    return [int(x) for x in re.findall(r"\d+", answer) if 1 <= int(x) <= limit]


def feedback_round(engine, model, query_text, ranking):
    shown = ranking[:TOP_K]
    answer = ask("\n  Relevance feedback - numbers of RELEVANT results (e.g. '1 3'), Enter to skip: ")
    relevant = parse_numbers(answer, len(shown))
    if not relevant:
        return
    default_nonrel = " ".join(str(i) for i in range(1, len(shown) + 1) if i not in relevant)
    answer = ask(f"  Numbers of NON-relevant results [Enter = {default_nonrel or 'none'}]: ",
                 default_nonrel)
    nonrelevant = [i for i in parse_numbers(answer, len(shown)) if i not in relevant]
    choice = ask("  Method - [R]occhio, Ide [I] Regular, Ide [D]ec-Hi [R]: ", "r").lower()
    method = {"r": "rocchio", "i": "ide_regular", "d": "ide_dec_hi"}.get(choice[:1], "rocchio")
    query_vec = model.query_vector(query_text)
    new_q = relevance_feedback(model, query_vec,
                               [shown[i - 1].doc_id for i in relevant],
                               [shown[i - 1].doc_id for i in sorted(nonrelevant)], method)
    new_ranking = model.rank_vector(new_q)
    print_side_by_side(engine, ranking, new_ranking, METHOD_NAMES[method])
    expansion = added_terms(query_vec, new_q, 6)
    if expansion:
        print("\n  Terms added to the query: " + ", ".join(f"{t} ({w:.3f})" for t, w in expansion))


def boolean_round(engine, text):
    model = engine.boolean()
    try:
        tree = parse_query(text, model.method)
        hits = model.search(text)
    except BooleanQueryError as err:
        print(f"  Query error: {err}")
        return
    print(f"  Parsed query: {ast_to_string(tree)}")
    if model.unknown_terms:
        print(f"  (terms not in any document: {', '.join(sorted(set(model.unknown_terms)))})")
    if not hits:
        print("  No condition satisfies this Boolean query (Boolean retrieval is exact-match: "
              "try OR, or a normal ranked search).")
        return
    print(f"  {len(hits)} matching condition(s) (Boolean results are an unranked set, "
          "listed alphabetically):")
    terms = set(preprocess(text, model.method))
    for doc_id in hits[:10]:
        doc = engine.documents[doc_id]
        matched = sorted(terms & set(model.doc_tokens[doc_id]))
        print(f"\n  *  {doc.disease}")
        if doc.also_called:
            print(wrap("Also known as", ", ".join(doc.also_called[:5])))
        print(wrap("Matched terms", ", ".join(matched) or "-"))
        description = doc.description if len(doc.description) <= 260 else doc.description[:257] + "..."
        print(wrap("Description", description))
        print(wrap("Precautions", "; ".join(doc.precautions)))
    if len(hits) > 10:
        print(f"\n  ... and {len(hits) - 10} more")
    print_disclaimer()


def interactive(engine):
    severity = False
    mode = "r"  # last chosen answer mode, offered as the default next time
    print(HELP)
    while True:
        try:
            text = ask("\nSymptoms> ")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            return
        if not text:
            continue
        low = text.lower()
        if low in ("quit", "exit", "q"):
            print("Goodbye.")
            return
        if low == "help":
            print(HELP)
            continue
        if low.startswith("severity"):
            severity = low.endswith("on")
            print(f"  Severity weighting {'ON' if severity else 'OFF'}")
            continue
        if low.startswith("suggest ") or (low.endswith("*") and " " not in low):
            prefix = low.split(" ", 1)[1] if low.startswith("suggest ") else low[:-1]
            options = engine.suggest_symptoms(prefix)
            print("  Suggestions: " + (", ".join(options) if options else "(none)"))
            continue
        try:
            text = spelling_check(engine, text)
            default = "R" if mode == "r" else "B"
            answer = ask(f"  Answer mode - [R]anked (vector space) or [B]oolean (AND/OR/NOT)? [{default}]: ",
                         mode).lower()
            mode = "b" if answer.startswith("b") else "r"
            if mode == "b":
                boolean_round(engine, text)
                continue
            model = engine.vsm(severity_boost=DEFAULT_SEVERITY_BOOST if severity else 0.0)
            groups = extract_noun_groups(text)
            if groups:
                print(f"  Recognised symptom phrases: {', '.join(groups)}")
            ranking = model.search(text)
            print_vsm_results(engine, model, text, ranking[:TOP_K])
            if ranking:
                feedback_round(engine, model, text, ranking)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            return


# ---------------------------------------------------------------------------
# One-shot experiment demos
# ---------------------------------------------------------------------------
def demo_preprocess(text, method):
    print(f"\n[Exp 2] Preprocessing pipeline (method = {method})\n")
    for name, value in pipeline_steps(text, method).items():
        print(wrap(name, str(value), indent=2, label_width=25))


def demo_boolean(engine, query):
    print(f"\n[Exp 3] Boolean retrieval: {query}")
    print(f"  Incidence matrix: {engine.boolean().matrix.shape[0]} terms x "
          f"{engine.boolean().matrix.shape[1]} documents")
    boolean_round(engine, query)


def demo_tfidf(engine, query, severity):
    model = engine.vsm(severity_boost=DEFAULT_SEVERITY_BOOST if severity else 0.0)
    print(f"\n[Exp 4] Vector space model, query: {query}")
    q = model.query_vector(query)
    print("  Query vector (tf-idf" + (" x severity boost" if severity else "") + "):")
    for t, w in sorted(q.items(), key=lambda x: -x[1]):
        df = model.index.document_frequency(t)
        print(f"    {t:<14} df={df:<3} idf={model.idf[t]:.3f}  weight={w:.3f}")
    print_vsm_results(engine, model, query, model.search(query, TOP_K))


def demo_pattern(engine, pattern, kind):
    pm = engine.matcher
    print(f"\n[Exp 5] Pattern matching over {len(pm.vocabulary)} vocabulary words: '{pattern}'")
    if kind == "auto":
        kind = ("regex" if re.search(r"[\^$+{}|\\]", pattern)
                else "wildcard" if re.search(r"[*?\[]", pattern) else "all")
    runs = {
        "exact": lambda: pm.exact_match(pattern),
        "prefix": lambda: pm.prefix_match(pattern),
        "suffix": lambda: pm.suffix_match(pattern),
        "substring": lambda: pm.substring_match(pattern),
        "wildcard": lambda: pm.wildcard_search(pattern),
        "regex": lambda: pm.regex_search(pattern),
        "fuzzy": lambda: [f"{w} (d={d})" for w, d in pm.fuzzy_search(pattern)],
    }
    selected = list(runs) if kind == "all" else [kind]
    if kind == "all":
        selected.remove("wildcard")
        selected.remove("regex")
    for name in selected:
        result = runs[name]()
        print(wrap(name, ", ".join(result) if result else "(no match)", indent=2))
    if kind in ("prefix", "all"):
        print(wrap("symptom suggest", ", ".join(engine.suggest_symptoms(pattern)) or "(none)", indent=2))


def demo_range(engine, low, high):
    words = engine.matcher.range_search(low.lower(), high.lower())
    print(f"\n[Exp 5] Lexical range search '{low}' .. '{high}': {len(words)} words")
    print(textwrap.fill(", ".join(words), WIDTH, initial_indent="  ", subsequent_indent="  "))


def demo_fuzzy(engine, word):
    word = word.lower()
    hits = engine.matcher.fuzzy_search(word)
    print(f"\n[Exp 5] Fuzzy search for '{word}' (allowed edit distance {max_edits_for(word)}):")
    for w, d in hits[:8]:
        print(f"    {w:<20} edit distance {d}")
    if hits:
        best = hits[0][0]
        print(f"\n  Did you mean '{best}'?   DP table  D[i][j] for '{word}' vs '{best}':")
        table = levenshtein_matrix(word, best)
        print("        " + "  ".join(f"{c:>2}" for c in " " + best))
        for i, row in enumerate(table):
            label = " " if i == 0 else word[i - 1]
            print(f"    {label:>2}  " + "  ".join(f"{v:>2}" for v in row))


def demo_feedback(engine, query, relevant, nonrelevant, method):
    model = engine.vsm()
    ranking = model.search(query)
    print(f"\n[Exp 6] Relevance feedback ({METHOD_NAMES[method]}) for: {query}")
    for i, r in enumerate(ranking[:TOP_K], start=1):
        print(f"    {i}. {engine.documents[r.doc_id].disease}")
    rel = [ranking[i - 1].doc_id for i in relevant if i <= len(ranking)]
    nonrel = [ranking[i - 1].doc_id for i in nonrelevant if i <= len(ranking)]
    print(f"  relevant = {relevant}, non-relevant = {nonrelevant}")
    q = model.query_vector(query)
    new_q = relevance_feedback(model, q, rel, nonrel, method)
    print_side_by_side(engine, ranking, model.rank_vector(new_q), METHOD_NAMES[method])
    print("\n  Terms added to the query: " + ", ".join(
        f"{t} ({w:.3f})" for t, w in added_terms(q, new_q, 6)))


def demo_index_compare(engine, patterns):
    import pandas as pd
    index = InvertedIndex([preprocess(d.text, "none") for d in engine.documents])
    table, stats = compare_indexes(index, patterns)
    print("\n[Exp 7] Substring lookup: inverted-index vocabulary scan vs suffix array vs suffix trie")
    for k, v in stats.items():
        print(f"    {k:<24} {v:.2f}" if isinstance(v, float) else f"    {k:<24} {v}")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.max_colwidth", 45):
        print()
        print(table.to_string(index=False))
    print("\n  Times are microseconds per lookup (mean of 200 runs). The scan is O(V * L);")
    print("  the suffix array needs O(m log n) comparisons; the suffix trie walks O(m) nodes.")
    print("  Exact term lookup in the inverted index itself is a single hash lookup, e.g.")
    print(f"  postings('fever') = {index.postings('fever')[:4]} ...")


def demo_evaluate():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
    import run_evaluation
    print("\n[Exp 8] Evaluation\n")
    run_evaluation.run()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Symptom-to-Condition Retrieval System")
    parser.add_argument("--search", metavar="QUERY", help="one-shot ranked (VSM) search")
    parser.add_argument("--preprocess", metavar="TEXT", help="Exp 2: show preprocessing steps")
    parser.add_argument("--method", choices=["stem", "lemma"], default="stem",
                        help="normalisation for --preprocess / --search (default stem)")
    parser.add_argument("--boolean", metavar="QUERY", help="Exp 3: Boolean query")
    parser.add_argument("--export-matrix", metavar="CSV", help="Exp 3: save incidence matrix")
    parser.add_argument("--tfidf", metavar="QUERY", help="Exp 4: query vector and score breakdown")
    parser.add_argument("--severity", action="store_true", help="use severity-weighted scoring")
    parser.add_argument("--pattern", metavar="PATTERN", help="Exp 5: pattern query, e.g. 'head*'")
    parser.add_argument("--pattern-type", default="auto",
                        choices=["auto", "exact", "prefix", "suffix", "substring",
                                 "wildcard", "regex", "fuzzy"])
    parser.add_argument("--range", nargs=2, metavar=("LOW", "HIGH"), help="Exp 5: lexical range")
    parser.add_argument("--fuzzy", metavar="WORD", help="Exp 5: edit-distance search + DP table")
    parser.add_argument("--feedback", metavar="QUERY", help="Exp 6: explicit relevance feedback")
    parser.add_argument("--relevant", type=int, nargs="*", default=[1], help="ranks marked relevant")
    parser.add_argument("--nonrelevant", type=int, nargs="*", default=[], help="ranks marked non-relevant")
    parser.add_argument("--feedback-method", choices=list(METHODS), default="rocchio")
    parser.add_argument("--index-compare", nargs="*", metavar="PATTERN",
                        help="Exp 7: compare index structures (optional patterns)")
    parser.add_argument("--evaluate", action="store_true", help="Exp 8: run the evaluation")
    parser.add_argument("--no-external", action="store_true",
                        help="search only the original Kaggle documents (no MedlinePlus "
                             "names, no Symptom2Disease document expansion)")
    args = parser.parse_args(argv)

    banner()
    if args.preprocess:
        demo_preprocess(args.preprocess, args.method)
    if args.evaluate:
        demo_evaluate()
    needs_engine = any([args.search, args.boolean, args.export_matrix, args.tfidf, args.pattern,
                        args.range, args.fuzzy, args.feedback, args.index_compare is not None])
    ran_demo = bool(args.preprocess or args.evaluate)
    if needs_engine:
        engine = SearchEngine(method=args.method, medlineplus=not args.no_external,
                              symptom2disease=not args.no_external)
        ran_demo = True
        if args.search:
            model = engine.vsm(severity_boost=DEFAULT_SEVERITY_BOOST if args.severity else 0.0)
            query, fixes = engine.correct(args.search)
            if fixes:
                print("Spelling corrected: " + ", ".join(f"{a} -> {b}" for a, b in fixes.items()))
            print(f"\nTop {TOP_K} conditions for: {query}")
            print_vsm_results(engine, model, query, model.search(query, TOP_K))
        if args.boolean:
            demo_boolean(engine, args.boolean)
        if args.export_matrix:
            path = engine.boolean().export_matrix(args.export_matrix)
            print(f"\n[Exp 3] Incidence matrix saved to {path}")
        if args.tfidf:
            demo_tfidf(engine, args.tfidf, args.severity)
        if args.pattern:
            demo_pattern(engine, args.pattern.lower(), args.pattern_type)
        if args.range:
            demo_range(engine, *args.range)
        if args.fuzzy:
            demo_fuzzy(engine, args.fuzzy)
        if args.feedback:
            demo_feedback(engine, args.feedback, args.relevant, args.nonrelevant,
                          args.feedback_method)
        if args.index_compare is not None:
            demo_index_compare(engine, args.index_compare or
                               ["fever", "head", "itis", "rrhoe", "pain", "skin", "ache", "xyz"])
    if not ran_demo:
        interactive(SearchEngine(method=args.method, medlineplus=not args.no_external,
                                 symptom2disease=not args.no_external))


if __name__ == "__main__":
    main()
