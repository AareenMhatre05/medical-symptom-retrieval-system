"""
Experiment 8 - compare retrieval configurations on two test collections.

Test collections
  own   eval/test_queries.json - 38 hand-written queries covering all 41 diseases, graded
        judgments (2 = correct, 1 = plausible related condition)
  s2d   Symptom2Disease test split - 212 descriptions written by people, 22 diseases,
        one correct disease per query (grade 2).  Never used to build the documents.

Configurations (all use the enriched documents: MedlinePlus names + Symptom2Disease
document expansion from the TRAIN split)
  a) Boolean model (terms ANDed)  /  Boolean model (terms ORed)
  b) VSM + Porter stemming                    (baseline ranked model)
  c) VSM + WordNet lemmatisation
  d) VSM + fuzzy spelling correction (Levenshtein)
  e) VSM + severity weighting (lambda = 0.5)
  f) VSM + Rocchio pseudo-relevance feedback
  g) VSM + Ide Regular pseudo-relevance feedback
  h) VSM + Ide Dec-Hi pseudo-relevance feedback

Data-source comparison: VSM + stemming with the original documents only, + MedlinePlus
names, + Symptom2Disease expansion, and both.

Binary metrics: relevant = grade >= 1.  Ranked models: P / R / F1 over the top 5.
Boolean models return an unranked set: P / R / F1 over the whole set, and the set is
listed in document-id order for the rank-based metrics.

Outputs: eval/results.csv, eval/results_data_sources.csv, eval/results_per_query.csv,
         eval/severity_sweep.csv, eval/map_ndcg_by_config.png, eval/precision_recall_curve.png

Run:  python eval/run_evaluation.py
"""

import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EVAL_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ir.evaluation import evaluate_run, interpolated_precision, relevant_set  # noqa: E402
from ir.external_data import load_symptom2disease  # noqa: E402
from ir.preprocessing import preprocess  # noqa: E402
from ir.query_expansion import pseudo_relevance_feedback  # noqa: E402
from ir.search_engine import DEFAULT_SEVERITY_BOOST, SearchEngine  # noqa: E402

QUERIES_PATH = os.path.join(EVAL_DIR, "test_queries.json")
PRF_K = 3                 # top-k documents assumed relevant
PRF_NONREL = (10, 20)     # ranks 11-20 assumed non-relevant
CUTOFF = 5                # results shown to the user
TEST_SET_NAMES = {"own": "Own queries (38)", "s2d": "Symptom2Disease test (212)"}
COLUMNS = ["P", "R", "F1", "P@1", "P@3", "P@5", "MAP", "MRR", "NDCG@3", "NDCG@5", "retrieved"]


def load_queries(engine, path=QUERIES_PATH):
    """Own test collection -> (query texts, ids, judgments)."""
    with open(path, encoding="utf-8") as fh:
        queries = json.load(fh)
    judgments = [{engine.doc_id(name): grade for name, grade in q["relevance"].items()}
                 for q in queries]
    return [q["query"] for q in queries], [q["id"] for q in queries], judgments


def load_s2d_queries(engine):
    """Symptom2Disease test split -> (query texts, ids, judgments); empty if not downloaded."""
    rows = load_symptom2disease("test")
    return ([t for t, _ in rows], [f"S2D-{i:03d}" for i in range(1, len(rows) + 1)],
            [{engine.doc_id(d): 2} for _, d in rows])


def test_collections(engine):
    """{'own': ..., 's2d': ...}; the Symptom2Disease set is skipped if it is not downloaded."""
    sets = {"own": load_queries(engine)}
    s2d = load_s2d_queries(engine)
    if s2d[0]:
        sets["s2d"] = s2d
    return sets


def build_configurations(engine):
    """label -> (function(query text) -> ranked doc ids, is_ranked)."""
    stem_vsm = engine.vsm("stem")
    lemma_vsm = engine.vsm("lemma")
    sev_vsm = engine.vsm("stem", DEFAULT_SEVERITY_BOOST)
    boolean = engine.boolean("stem")

    def ids(results):
        return [r.doc_id for r in results]

    def prf(method):
        return lambda q: ids(pseudo_relevance_feedback(stem_vsm, q, method, k=PRF_K,
                                                       nonrel_ranks=PRF_NONREL)[0])

    return {
        "a) Boolean (AND)": (lambda q: boolean.search_terms(preprocess(q, "stem"), "AND"), False),
        "a) Boolean (OR)": (lambda q: boolean.search_terms(preprocess(q, "stem"), "OR"), False),
        "b) VSM + stemming": (lambda q: ids(stem_vsm.search(q)), True),
        "c) VSM + lemmatization": (lambda q: ids(lemma_vsm.search(q)), True),
        "d) VSM + fuzzy correction": (lambda q: ids(stem_vsm.search(engine.correct(q)[0])), True),
        "e) VSM + severity weighting": (lambda q: ids(sev_vsm.search(q)), True),
        "f) VSM + Rocchio (PRF)": (prf("rocchio"), True),
        "g) VSM + Ide Regular (PRF)": (prf("ide_regular"), True),
        "h) VSM + Ide Dec-Hi (PRF)": (prf("ide_dec_hi"), True),
    }


def evaluate_data_sources():
    """VSM + stemming on the original and the enriched collections."""
    variants = [
        ("Kaggle dataset only", False, False),
        ("+ MedlinePlus names", True, False),
        ("+ Symptom2Disease expansion", False, True),
        ("+ both (final system)", True, True),
    ]
    rows = []
    for label, medline, s2d in variants:
        engine = SearchEngine(medlineplus=medline, symptom2disease=s2d)
        model = engine.vsm("stem")
        for key, (texts, _, judgments) in test_collections(engine).items():
            summary, _ = evaluate_run([[r.doc_id for r in model.search(t)] for t in texts],
                                      judgments, cutoff=CUTOFF)
            rows.append({"data": label, "test_set": TEST_SET_NAMES[key],
                         "P@1": summary["P@1"], "P@3": summary["P@3"], "MAP": summary["MAP"],
                         "MRR": summary["MRR"], "NDCG@5": summary["NDCG@5"]})
    return pd.DataFrame(rows).round(4)


def run(engine=None, verbose=True, save=True):
    engine = engine or SearchEngine()
    sets = test_collections(engine)
    configs = build_configurations(engine)
    rows, per_query_rows, rankings = [], [], {}
    for key, (texts, ids, judgments) in sets.items():
        for label, (fn, ranked) in configs.items():
            ranked_lists = [fn(t) for t in texts]
            rankings[(key, label)] = ranked_lists
            summary, per_query = evaluate_run(ranked_lists, judgments,
                                              cutoff=CUTOFF if ranked else None)
            rows.append({"test_set": TEST_SET_NAMES[key], "configuration": label,
                         **{k: summary[k] for k in COLUMNS}})
            for qid, metrics in zip(ids, per_query):
                per_query_rows.append({"test_set": key, "configuration": label,
                                       "query_id": qid, "AP": metrics["AP"],
                                       "NDCG@5": metrics["NDCG@5"], "P@1": metrics["P@1"],
                                       "retrieved": metrics["retrieved"]})
    results = pd.DataFrame(rows).rename(columns={"retrieved": "avg_retrieved"}).round(4)

    # sensitivity of the severity boost (extra analysis, not one of the 8 configurations)
    sweep = []
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
        model = engine.vsm("stem", lam)
        row = {"lambda": lam}
        for key, (texts, _, judgments) in sets.items():
            summary, _ = evaluate_run([[r.doc_id for r in model.search(t)] for t in texts],
                                      judgments, cutoff=CUTOFF)
            row[f"MAP ({key})"] = summary["MAP"]
            row[f"P@1 ({key})"] = summary["P@1"]
        sweep.append(row)
    sweep = pd.DataFrame(sweep).round(4)

    sources = evaluate_data_sources()

    if save:
        results.to_csv(os.path.join(EVAL_DIR, "results.csv"), index=False)
        sources.to_csv(os.path.join(EVAL_DIR, "results_data_sources.csv"), index=False)
        pd.DataFrame(per_query_rows).round(4).to_csv(
            os.path.join(EVAL_DIR, "results_per_query.csv"), index=False)
        sweep.to_csv(os.path.join(EVAL_DIR, "severity_sweep.csv"), index=False)
        plot_bars(results)
        # PR curves need several relevant documents per query: a Symptom2Disease query has
        # exactly one, which makes every curve a flat line, so the own queries are used.
        pr_key = "own"
        plot_pr_curves({label: rankings[(pr_key, label)] for label in configs},
                       sets[pr_key][2], TEST_SET_NAMES[pr_key])
    if verbose:
        print("Documents:", engine.enrichment)
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            for name in results["test_set"].unique():
                print(f"\n=== {name}  (relevant = grade >= 1) ===")
                print(results[results["test_set"] == name].drop(columns="test_set")
                      .to_string(index=False))
            print("\n=== Data sources (VSM + stemming) ===")
            print(sources.to_string(index=False))
            print("\nSeverity boost sensitivity (VSM + stemming):")
            print(sweep.to_string(index=False))
        if save:
            print(f"\nSaved CSVs and charts to {EVAL_DIR}")
    return results


# ---------------------------------------------------------------------------
# Charts (static PNGs: light surface, recessive axes, values labelled on bars)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_BLUE = "#2a78d6"

# colour follows the entity (configuration / test set), identical in every chart
CONFIG_STYLE = {
    "b) VSM + stemming": ("#2a78d6", "o"),
    "a) Boolean (OR)": ("#eb6834", "s"),
    "c) VSM + lemmatization": ("#1baf7a", "^"),
    "d) VSM + fuzzy correction": ("#eda100", "D"),
    "e) VSM + severity weighting": ("#e87ba4", "v"),
    "f) VSM + Rocchio (PRF)": ("#008300", "s"),
    "g) VSM + Ide Regular (PRF)": ("#4a3aa7", "^"),
    "h) VSM + Ide Dec-Hi (PRF)": ("#e34948", "D"),
}
SET_COLORS = {"Own queries (38)": "#2a78d6", "Symptom2Disease test (212)": "#eb6834"}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def plot_bars(results):
    """Two panels on one 0-1 scale: MAP and NDCG@5 per configuration, one bar per test set."""
    labels = list(dict.fromkeys(results["configuration"]))
    sets = list(dict.fromkeys(results["test_set"]))
    height = 0.8 / len(sets)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True, facecolor=SURFACE)
    for ax, metric in zip(axes, ("MAP", "NDCG@5")):
        base = list(range(len(labels)))[::-1]
        for k, test_set in enumerate(sets):
            sub = results[results["test_set"] == test_set].set_index("configuration")
            values = [sub.loc[label, metric] for label in labels]
            y = [b + 0.4 - height * (k + 0.5) for b in base]
            ax.barh(y, values, height=height, color=SET_COLORS.get(test_set, SERIES_BLUE),
                    edgecolor=SURFACE, linewidth=2, label=test_set)
            for yi, v in zip(y, values):
                ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7.5,
                        color=TEXT_SECONDARY)
        ax.set_yticks(base)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.08)
        ax.set_title(metric, loc="left", color=TEXT_PRIMARY, fontsize=12)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _style_axes(ax)
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                        ncol=len(sets), frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    fig.suptitle("MAP and NDCG@5 per configuration", x=0.01, ha="left",
                 color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(EVAL_DIR, "map_ndcg_by_config.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pr_curves(rankings, judgments, set_name):
    """11-point interpolated precision-recall curves, averaged over the queries.
    Left: Boolean OR and the VSM variants.  Right: relevance-feedback methods vs. baseline.
    (Boolean AND returns at most a few documents and is left out of the curves.)"""
    levels = [i / 10 for i in range(11)]

    def averaged(label):
        curves = [interpolated_precision(r, relevant_set(g), levels)
                  for r, g in zip(rankings[label], judgments)]
        return [sum(c[i] for c in curves) / len(curves) for i in range(len(levels))]

    panels = [
        ("Retrieval models", ["b) VSM + stemming", "a) Boolean (OR)", "c) VSM + lemmatization",
                              "d) VSM + fuzzy correction", "e) VSM + severity weighting"]),
        ("Relevance feedback (pseudo, top-3)", ["b) VSM + stemming", "f) VSM + Rocchio (PRF)",
                                                "g) VSM + Ide Regular (PRF)",
                                                "h) VSM + Ide Dec-Hi (PRF)"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, facecolor=SURFACE)
    for ax, (title, labels) in zip(axes, panels):
        for label in labels:
            color, marker = CONFIG_STYLE[label]
            ax.plot(levels, averaged(label), color=color, marker=marker, markersize=6,
                    linewidth=2, label=label, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        # the band between the Boolean curve and the VSM curves is empty
        legend = ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), fontsize=9,
                           frameon=False)
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)
        _style_axes(ax)
    axes[0].set_ylabel("Interpolated precision")
    fig.suptitle(f"11-point interpolated precision-recall - {set_name}",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "precision_recall_curve.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    run()
