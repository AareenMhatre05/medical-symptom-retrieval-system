"""
External data sources used to enrich the 41 disease documents.

1. MedlinePlus health topics (U.S. National Library of Medicine, https://medlineplus.gov/xml.html)
   Each disease is mapped by hand to one MedlinePlus topic (MEDLINEPLUS_TOPICS).  The topic's
   "also called" names (e.g. Hemorrhoids -> "Piles", HIV -> "AIDS", Gastroenteritis ->
   "Stomach flu") are added to the document, so users can search by alternative names.
   The full MedlinePlus summaries are NOT indexed: on the development queries they made
   ranking worse (long general text dilutes the symptom terms).

2. Symptom2Disease (gretelai/symptom_to_diagnosis on Hugging Face, Apache-2.0)
   1065 plain-language symptom descriptions written by people, each labelled with one of
   22 of our diseases ("I've been having a lot of pain in my neck and back ...").
     * train split (853) -> DOCUMENT EXPANSION: the descriptions of a disease are appended
       to its document, so the documents contain the words real users type.
     * test split (212)  -> an independent evaluation set (never used for building).
   Train descriptions that also occur in the test split are dropped to avoid leakage.

Both sources are optional: if the files are missing the system falls back to the
original Kaggle documents.  `python download_data.py` downloads them.
"""

import glob
import json
import os
from dataclasses import replace

from ir.data_loader import PROJECT_ROOT, disease_key

EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "data", "external")
MEDLINEPLUS_DIR = os.path.join(EXTERNAL_DIR, "medlineplus")
MEDLINEPLUS_SUBSET = os.path.join(MEDLINEPLUS_DIR, "medlineplus_subset.json")
S2D_DIR = os.path.join(EXTERNAL_DIR, "symptom2disease")

# Our disease name -> MedlinePlus topic title.  Chosen by reading the topic titles,
# "also called" lists and summaries; the less obvious choices are:
#   Hepatitis D / E          -> "Hepatitis" (the general topic describes types D and E)
#   Alcoholic hepatitis      -> "Steatotic Liver Disease" (alcohol-associated liver disease)
#   Typhoid                  -> "Salmonella Infections" (typhoid is caused by Salmonella Typhi)
#   Paralysis (brain hem.)   -> "Hemorrhagic Stroke"
#   Chronic cholestasis      -> "Bile Duct Diseases" (cholestasis = blocked bile flow)
#   Cervical spondylosis     -> "Neck Injuries and Disorders"
MEDLINEPLUS_TOPICS = {
    "(vertigo) Paroymsal Positional Vertigo": "Dizziness and Vertigo",
    "Acne": "Acne", "AIDS": "HIV", "Alcoholic hepatitis": "Steatotic Liver Disease",
    "Allergy": "Allergy", "Arthritis": "Arthritis", "Bronchial Asthma": "Asthma",
    "Cervical spondylosis": "Neck Injuries and Disorders", "Chicken pox": "Chickenpox",
    "Chronic cholestasis": "Bile Duct Diseases", "Common Cold": "Common Cold",
    "Dengue": "Dengue", "Diabetes": "Diabetes", "Dimorphic hemorrhoids(piles)": "Hemorrhoids",
    "Drug Reaction": "Drug Reactions", "Fungal infection": "Fungal Infections",
    "Gastroenteritis": "Gastroenteritis", "GERD": "GERD", "Heart attack": "Heart Attack",
    "hepatitis A": "Hepatitis A", "Hepatitis B": "Hepatitis B", "Hepatitis C": "Hepatitis C",
    "Hepatitis D": "Hepatitis", "Hepatitis E": "Hepatitis",
    "Hypertension": "High Blood Pressure", "Hyperthyroidism": "Hyperthyroidism",
    "Hypoglycemia": "Hypoglycemia", "Hypothyroidism": "Hypothyroidism", "Impetigo": "Impetigo",
    "Jaundice": "Jaundice", "Malaria": "Malaria", "Migraine": "Migraine",
    "Osteoarthristis": "Osteoarthritis", "Paralysis (brain hemorrhage)": "Hemorrhagic Stroke",
    "Peptic ulcer diseae": "Peptic Ulcer", "Pneumonia": "Pneumonia", "Psoriasis": "Psoriasis",
    "Tuberculosis": "Tuberculosis", "Typhoid": "Salmonella Infections",
    "Urinary tract infection": "Urinary Tract Infections", "Varicose veins": "Varicose Veins",
}

# Diseases mapped to a BROADER topic: the topic is kept for reference, but its title and
# "also called" names are not synonyms of the disease (Typhoid is not "Salmonella
# Infections", Alcoholic hepatitis is not "Fatty liver disease"), so they are not added.
BROADER_TOPICS = {
    "(vertigo) Paroymsal Positional Vertigo", "Alcoholic hepatitis", "Cervical spondylosis",
    "Chronic cholestasis", "Hepatitis D", "Hepatitis E", "Typhoid",
}

# Symptom2Disease label -> our disease name (only labels that are spelled differently)
S2D_LABELS = {
    "gastroesophageal reflux disease": "GERD",
    "peptic ulcer disease": "Peptic ulcer diseae",
}


# ---------------------------------------------------------------------------
# MedlinePlus
# ---------------------------------------------------------------------------
def latest_medlineplus_xml(directory=MEDLINEPLUS_DIR):
    files = sorted(glob.glob(os.path.join(directory, "mplus_topics_????-??-??.xml")))
    return files[-1] if files else None


def build_medlineplus_subset(xml_path, out_path=MEDLINEPLUS_SUBSET):
    """Parse the full MedlinePlus XML (about 30 MB) and keep only the mapped topics.

    Writes {disease: {"topic", "url", "also_called"}} to a small JSON file so the project
    does not have to ship or re-parse the full XML.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    topics = {}
    for t in root.findall("health-topic"):
        if t.get("language") == "English":
            topics[t.get("title")] = {
                "topic": t.get("title"),
                "url": t.get("url"),
                "also_called": [a.text.strip() for a in t.findall("also-called") if a.text],
            }
    missing = [title for title in MEDLINEPLUS_TOPICS.values() if title not in topics]
    if missing:
        raise ValueError(f"MedlinePlus topics not found in {xml_path}: {missing}")
    subset = {disease: topics[title] for disease, title in MEDLINEPLUS_TOPICS.items()}
    subset["_source"] = {"file": os.path.basename(xml_path),
                         "attribution": "Information from MedlinePlus.gov (U.S. National Library of Medicine)"}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(subset, fh, indent=1)
    return out_path


def load_medlineplus(path=MEDLINEPLUS_SUBSET):
    """{disease: {"topic", "url", "also_called"}}; {} if the file is missing."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("_source", None)
    return data


# ---------------------------------------------------------------------------
# Symptom2Disease
# ---------------------------------------------------------------------------
def load_symptom2disease(split, directory=S2D_DIR):
    """[(description, our disease name)] for 'train' or 'test'; [] if the file is missing."""
    path = os.path.join(directory, f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                label = item["output_text"].strip().lower()
                rows.append((item["input_text"].strip(), S2D_LABELS.get(label, label)))
    return rows


def expansion_texts(directory=S2D_DIR):
    """Train descriptions grouped by disease key, minus any text that is also in the test
    split (so the test queries stay unseen)."""
    test_texts = {t.lower() for t, _ in load_symptom2disease("test", directory)}
    grouped = {}
    for text, disease in load_symptom2disease("train", directory):
        if text.lower() not in test_texts:
            grouped.setdefault(disease_key(disease), []).append(text)
    return grouped


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------
def enrich_documents(documents, medlineplus=True, symptom2disease=True):
    """Return copies of the documents with alternative names and/or user descriptions.

    Returns (documents, info) where info reports what was added.
    """
    names = load_medlineplus() if medlineplus else {}
    expansions = expansion_texts() if symptom2disease else {}
    enriched, info = [], {"medlineplus_docs": 0, "expanded_docs": 0, "expansion_texts": 0}
    for doc in documents:
        also = []
        if doc.disease in names and doc.disease not in BROADER_TOPICS:
            # keep only names that differ from the disease name itself
            also = [a for a in names[doc.disease]["also_called"]
                    if disease_key(a) != disease_key(doc.disease)]
            topic = names[doc.disease]["topic"]
            if disease_key(topic) != disease_key(doc.disease):
                also = [topic] + also
            info["medlineplus_docs"] += 1
        texts = expansions.get(disease_key(doc.disease), [])
        if texts:
            info["expanded_docs"] += 1
            info["expansion_texts"] += len(texts)
        enriched.append(replace(doc, also_called=also, user_descriptions=texts))
    return enriched, info
