"""
Dataset cleaning and document construction.

Raw files (Kaggle "Disease Symptom Prediction" dataset, in data/archive/):
  dataset.csv              4920 rows = 41 diseases x ~120 duplicate/partial rows
  symptom_Description.csv  one description per disease
  symptom_precaution.csv   up to four precautions per disease
  Symptom-severity.csv     severity weight (1-7) per symptom

Problems handled here:
  1. Duplicate rows   -> every disease is merged into ONE document whose symptom list is
                         the union of the symptoms in all of its rows.
  2. Symptom names    -> " skin_rash" -> "skin rash" (strip, underscores to spaces,
                         collapse repeated spaces: "dischromic _patches" -> "dischromic patches").
  3. Disease names    -> "Diabetes " vs "Diabetes", double spaces, case differences, and a
                         real misspelling across files ("hemmorhoids" vs "hemorrhoids").
                         Names are normalised to a join key; keys that still do not match
                         are joined by the closest Levenshtein distance (<= 3).
  4. Severity table   -> "foul_smell_ofurine" vs "foul_smell_of urine" are matched with a
                         space-free key; the duplicate "fluid_overload" entry keeps the max
                         weight; the non-symptom row "prognosis" is ignored.

Each document's text = disease name + description + cleaned symptom names.
"""

import os
import re
from dataclasses import dataclass, field

import pandas as pd

from ir.pattern_matching import levenshtein

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "archive")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed_documents.csv")

MAX_NAME_DISTANCE = 3  # max edit distance accepted when fuzzy-joining disease names


@dataclass
class Document:
    """One retrievable unit: a disease with all of its information."""
    doc_id: int
    disease: str
    description: str
    symptoms: list = field(default_factory=list)
    precautions: list = field(default_factory=list)
    # optional enrichment from external sources (see ir/external_data.py)
    also_called: list = field(default_factory=list)        # MedlinePlus alternative names
    user_descriptions: list = field(default_factory=list)  # Symptom2Disease train texts

    @property
    def core_text(self):
        """Disease name (+ alternative names) + description + cleaned symptoms."""
        names = f" Also called: {', '.join(self.also_called)}." if self.also_called else ""
        return f"{self.disease}.{names} {self.description} Symptoms: {', '.join(self.symptoms)}."

    @property
    def text(self):
        """Indexed text = core text + user descriptions (document expansion)."""
        if not self.user_descriptions:
            return self.core_text
        return self.core_text + " " + " ".join(self.user_descriptions)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def clean_whitespace(text):
    """Strip and collapse runs of whitespace: '  Peptic  ulcer ' -> 'Peptic ulcer'."""
    return " ".join(str(text).split())


def clean_symptom(raw):
    """' skin_rash' -> 'skin rash';  'dischromic _patches' -> 'dischromic patches'."""
    return clean_whitespace(str(raw).replace("_", " ")).lower()


def symptom_key(raw):
    """Space-free key so 'foul_smell_ofurine' and 'foul_smell_of urine' match."""
    return re.sub(r"[\s_]+", "", str(raw)).lower()


def disease_key(raw):
    """Join key for disease names: lower case, single spaces, no space next to brackets."""
    key = clean_whitespace(raw).lower()
    key = re.sub(r"\s*\(\s*", "(", key)
    key = re.sub(r"\s*\)\s*", ") ", key).strip()
    return key


def clean_sentence_text(text):
    """Fix missing spaces after full stops ('body.They' -> 'body. They') and replace
    typographic dashes/quotes with ASCII so text prints on any console."""
    text = clean_whitespace(text)
    for fancy, plain in {"—": "-", "–": "-", "’": "'", "‘": "'",
                         "“": '"', "”": '"'}.items():
        text = text.replace(fancy, plain)
    return re.sub(r"\.(?=[A-Z])", ". ", text)


def _match_key(key, candidates, log):
    """Exact key match, otherwise the closest candidate by Levenshtein distance."""
    if key in candidates:
        return key
    best = min(candidates, key=lambda c: levenshtein(key, c))
    distance = levenshtein(key, best)
    if distance <= MAX_NAME_DISTANCE:
        log.append(f"fuzzy-joined disease '{key}' -> '{best}' (edit distance {distance})")
        return best
    log.append(f"no match for disease '{key}'")
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_severity(data_dir=DEFAULT_DATA_DIR):
    """Return {space-free symptom key: severity weight}. Duplicates keep the max weight."""
    sev = pd.read_csv(os.path.join(data_dir, "Symptom-severity.csv"))
    weights = {}
    for name, weight in zip(sev["Symptom"], sev["weight"]):
        key = symptom_key(name)
        weights[key] = max(int(weight), weights.get(key, 0))
    return weights


def load_documents(data_dir=DEFAULT_DATA_DIR, return_log=False):
    """Clean the four CSV files and build one Document per disease.

    Returns (documents, severity) where severity maps the cleaned symptom name
    (e.g. 'chest pain') to its weight 1-7.  With return_log=True a list of cleaning
    notes is returned as a third element.
    """
    log = []
    raw = pd.read_csv(os.path.join(data_dir, "dataset.csv"))
    desc = pd.read_csv(os.path.join(data_dir, "symptom_Description.csv"))
    prec = pd.read_csv(os.path.join(data_dir, "symptom_precaution.csv"))
    log.append(f"dataset.csv: {len(raw)} rows, {raw['Disease'].nunique()} raw disease names")

    # 1+2: merge duplicate rows -> union of cleaned symptoms per disease
    symptom_cols = [c for c in raw.columns if c.lower().startswith("symptom")]
    merged, display_name = {}, {}
    for _, row in raw.iterrows():
        key = disease_key(row["Disease"])
        display_name.setdefault(key, clean_whitespace(row["Disease"]))
        bucket = merged.setdefault(key, [])
        for col in symptom_cols:
            value = row[col]
            if pd.isna(value) or not str(value).strip():
                continue
            symptom = clean_symptom(value)
            if symptom not in bucket:
                bucket.append(symptom)
    log.append(f"merged into {len(merged)} disease documents")

    # 3: join description / precaution tables on the normalised disease key
    desc_by_key = {disease_key(d): (clean_whitespace(d), clean_sentence_text(t))
                   for d, t in zip(desc["Disease"], desc["Description"])}
    prec_by_key = {}
    for _, row in prec.iterrows():
        items = [clean_whitespace(v) for v in row.iloc[1:] if not pd.isna(v) and str(v).strip()]
        prec_by_key[disease_key(row["Disease"])] = [p[0].upper() + p[1:] for p in items]

    documents = []
    for doc_id, key in enumerate(sorted(merged)):
        d_key = _match_key(key, desc_by_key, log)
        p_key = _match_key(key, prec_by_key, log)
        name, description = desc_by_key[d_key] if d_key else (display_name[key], "")
        documents.append(Document(
            doc_id=doc_id,
            disease=name,  # description file has the correctly spelled names
            description=description,
            symptoms=merged[key],
            precautions=prec_by_key.get(p_key, []) if p_key else [],
        ))

    # 4: severity weights keyed by the cleaned symptom name
    sev_by_key = load_severity(data_dir)
    all_symptoms = sorted({s for doc in documents for s in doc.symptoms})
    severity = {}
    for symptom in all_symptoms:
        weight = sev_by_key.get(symptom_key(symptom))
        if weight is None:
            log.append(f"no severity weight for symptom '{symptom}'")
        else:
            severity[symptom] = weight
    log.append(f"{len(all_symptoms)} distinct symptoms, {len(severity)} with severity weights")

    if return_log:
        return documents, severity, log
    return documents, severity


def documents_to_dataframe(documents):
    """Tabular view of the cleaned collection (lists joined with ' | ')."""
    return pd.DataFrame([{
        "doc_id": d.doc_id,
        "disease": d.disease,
        "num_symptoms": len(d.symptoms),
        "symptoms": " | ".join(d.symptoms),
        "description": d.description,
        "precautions": " | ".join(d.precautions),
        "text": d.text,
    } for d in documents])


def save_processed(documents, path=PROCESSED_PATH):
    """Write the cleaned one-row-per-disease collection to CSV."""
    documents_to_dataframe(documents).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    docs, sev, notes = load_documents(return_log=True)
    for note in notes:
        print("-", note)
    print("saved", save_processed(docs))
