"""
Builds eval/test_queries.json.

Each test query has
  * a plain-language query (some contain deliberate spelling mistakes),
  * the canonical symptoms it describes (used only to create judgments),
  * the intended condition                                 -> grade 2,
  * plausible related conditions                           -> grade 1, obtained from
      (a) severity-weighted symptom overlap with Symptom-severity.csv:
              overlap(D) = sum_{s in Sq ∩ SD} sev(s) / sum_{s in Sq} sev(s)
          a disease with overlap >= OVERLAP_THRESHOLD is a plausible condition, and
      (b) a few hand-added clinically related conditions (e.g. other hepatitis types),
  * every other condition                                  -> grade 0 (implicit).

Run:  python eval/build_test_queries.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ir.data_loader import load_documents  # noqa: E402

OVERLAP_THRESHOLD = 0.6
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries.json")

# (query text, canonical symptoms described, intended disease, hand-added related diseases)
QUERIES = [
    ("Itchy skin with a red rash and small bumps, and some discoloured patches",
     ["itching", "skin rash", "nodal skin eruptions", "dischromic patches"],
     "Fungal infection", []),
    ("I keep sneezing nonstop, I have chills, shivering and watery eyes",
     ["continuous sneezing", "chills", "shivering", "watering from eyes"],
     "Allergy", ["Common Cold"]),
    ("Heartburn after meals, stomach pain, sores on my tongue and a cough",
     ["acidity", "stomach pain", "ulcers on tongue", "cough"],
     "GERD", ["Peptic ulcer diseae"]),
    ("Itching all over, yellow skin and yellow eyes, nausea and no appetite",
     ["itching", "yellowish skin", "yellowing of eyes", "nausea", "loss of appetite"],
     "Chronic cholestasis", ["Jaundice"]),
    ("Skin rash and itching after starting a new medicine, burning when I pee",
     ["skin rash", "itching", "burning micturition"],
     "Drug Reaction", []),
    ("Pain during bowel movements, constipation, blood in stool and irritation in the anus",
     ["pain during bowel movements", "constipation", "bloody stool", "irritation in anus"],
     "Dimorphic hemorrhoids(piles)", []),
    ("Always hungry, losing weight, blurry vision, peeing a lot and very tired",
     ["excessive hunger", "weight loss", "blurred and distorted vision", "polyuria", "fatigue"],
     "Diabetes", ["Hyperthyroidism"]),
    ("Vomitting, sunken eyes, I am dehydrated and have watery diarhea",
     ["vomiting", "sunken eyes", "dehydration", "diarrhoea"],
     "Gastroenteritis", []),
    ("Short of breath, cough with mucoid sputum and a family history of asthma",
     ["breathlessness", "cough", "mucoid sputum", "family history"],
     "Bronchial Asthma", ["Pneumonia"]),
    ("Headache, chest pain, dizziness and I can't concentrate",
     ["headache", "chest pain", "dizziness", "lack of concentration"],
     "Hypertension", []),
    ("Throbbing headache with blurred vision, visual disturbances, acidity and a stiff neck",
     ["headache", "blurred and distorted vision", "visual disturbances", "acidity", "stiff neck"],
     "Migraine", []),
    ("Neck pain, back pain, feeling dizzy and weakness in my limbs",
     ["neck pain", "back pain", "dizziness", "weakness in limbs"],
     "Cervical spondylosis", []),
    ("Sudden weakness of one body side, headache, vomiting and altered sensorium",
     ["weakness of one body side", "headache", "vomiting", "altered sensorium"],
     "Paralysis (brain hemorrhage)", []),
    ("Yellowish skin, dark urine, high fever, itching and weight loss",
     ["yellowish skin", "dark urine", "high fever", "itching", "weight loss"],
     "Jaundice", ["Hepatitis E"]),
    ("High feaver with chils and sweating, headake, nausia and muscle pain",
     ["high fever", "chills", "sweating", "headache", "nausea", "muscle pain"],
     "Malaria", ["Typhoid", "Dengue"]),
    ("Fever with red spots all over my body, itching, tiredness and loss of appetite",
     ["mild fever", "red spots over body", "itching", "fatigue", "loss of appetite"],
     "Chicken pox", ["Dengue"]),
    ("High fever, pain behind the eyes, joint pain, body aches and a skin rash",
     ["high fever", "pain behind the eyes", "joint pain", "muscle pain", "skin rash"],
     "Dengue", ["Chicken pox", "Malaria"]),
    ("High fever for days, belly pain, constipation, headache and a toxic look",
     ["high fever", "belly pain", "constipation", "headache", "toxic look (typhos)"],
     "Typhoid", ["Malaria"]),
    ("Yellow eyes, dark urine, joint pain, mild fever and diarrhoea",
     ["yellowing of eyes", "dark urine", "joint pain", "mild fever", "diarrhoea"],
     "hepatitis A", ["Hepatitis E", "Hepatitis D"]),
    ("Yellow skin and fatigue after receiving a blood transfusion and unsterile injections",
     ["yellowish skin", "fatigue", "receiving blood transfusion", "receiving unsterile injections"],
     "Hepatitis B", ["Hepatitis C"]),
    ("Yelow skin, vomiting, swelling of stomach and a history of alcohol consumption",
     ["yellowish skin", "vomiting", "swelling of stomach", "history of alcohol consumption"],
     "Alcoholic hepatitis", ["Hepatitis E"]),
    ("Crushing chest pain, sweating heavily, vomiting and short of breath",
     ["chest pain", "sweating", "vomiting", "breathlessness"],
     "Heart attack", []),
    ("Racing heart, excessive sweating, weight loss, mood swings and abnormal menstruation",
     ["fast heart rate", "sweating", "weight loss", "mood swings", "abnormal menstruation"],
     "Hyperthyroidism", ["Hypothyroidism"]),
    ("Sweating, anxiety, palpitations, slurred speech, tingling lips and excessive hunger",
     ["sweating", "anxiety", "palpitations", "slurred speech", "drying and tingling lips",
      "excessive hunger"],
     "Hypoglycemia", []),
    ("Weight gain, cold hands and feet, puffy face and eyes, brittle nails and depression",
     ["weight gain", "cold hands and feets", "puffy face and eyes", "brittle nails", "depression"],
     "Hypothyroidism", ["Hyperthyroidism"]),
    ("Red sore around the nose, blisters with yellow crust oozing and high fever",
     ["red sore around nose", "blister", "yellow crust ooze", "high fever"],
     "Impetigo", []),
    ("Knee pain, hip joint pain, swollen joints and painful walking",
     ["knee pain", "hip joint pain", "swelling joints", "painful walking"],
     "Osteoarthristis", ["Arthritis"]),
    ("Muscle weakness, stiff neck, swelling joints and movement stiffness",
     ["muscle weakness", "stiff neck", "swelling joints", "movement stiffness"],
     "Arthritis", ["Osteoarthristis"]),
    ("Indigestion, abdominal pain, passing gas, loss of appetite and vomiting",
     ["indigestion", "abdominal pain", "passage of gases", "loss of appetite", "vomiting"],
     "Peptic ulcer diseae", ["GERD"]),
    ("High fever, cough with rusty sputum, chest pain, chills and fast heart rate",
     ["high fever", "cough", "rusty sputum", "chest pain", "chills", "fast heart rate"],
     "Pneumonia", ["Tuberculosis", "Bronchial Asthma"]),
    ("Skin peeling, silver like dusting on the skin, joint pain and small dents in nails",
     ["skin peeling", "silver like dusting", "joint pain", "small dents in nails"],
     "Psoriasis", []),
    ("Persistant cough with blood in sputum, weigt loss, night sweats and mild fever",
     ["cough", "blood in sputum", "weight loss", "sweating", "mild fever"],
     "Tuberculosis", ["Pneumonia"]),
    ("Burning when I pee, bladder discomfort, smelly urine and a continuous feel of urine",
     ["burning micturition", "bladder discomfort", "foul smell of urine", "continuous feel of urine"],
     "Urinary tract infection", []),
    ("Swollen legs, prominent veins on my calf, cramps and bruising",
     ["swollen legs", "prominent veins on calf", "cramps", "bruising"],
     "Varicose veins", []),
    ("The room is spinning, nausea, vomiting, unsteadiness and loss of balance",
     ["spinning movements", "nausea", "vomiting", "unsteadiness", "loss of balance"],
     "(vertigo) Paroymsal Positional Vertigo", []),
    ("Pus filled pimples, blackheads and a skin rash on my face",
     ["pus filled pimples", "blackheads", "skin rash"],
     "Acne", []),
    ("Runny nose, sneezing, sore throat, cough, congestion and loss of smell",
     ["runny nose", "continuous sneezing", "throat irritation", "cough", "congestion", "loss of smell"],
     "Common Cold", ["Allergy"]),
    ("Muscle wasting, white patches in throat, high fever and extra marital contacts",
     ["muscle wasting", "patches in throat", "high fever", "extra marital contacts"],
     "AIDS", []),
]


def build():
    documents, severity = load_documents()
    by_name = {d.disease: d for d in documents}
    queries = []
    for qid, (text, symptoms, target, related) in enumerate(QUERIES, start=1):
        assert target in by_name, target
        unknown = [s for s in symptoms if s not in severity]
        assert not unknown, f"Q{qid}: unknown symptoms {unknown}"
        query_weight = sum(severity[s] for s in symptoms)
        grades = {target: 2}
        overlap_scores = {}
        for doc in documents:
            if doc.disease == target:
                continue
            shared = set(symptoms) & set(doc.symptoms)
            overlap = sum(severity[s] for s in shared) / query_weight
            if overlap >= OVERLAP_THRESHOLD:
                grades[doc.disease] = 1
                overlap_scores[doc.disease] = round(overlap, 2)
        for name in related:
            assert name in by_name, name
            grades.setdefault(name, 1)
        queries.append({
            "id": f"Q{qid:02d}",
            "query": text,
            "symptoms": symptoms,
            "relevance": grades,
            "severity_overlap": overlap_scores,
        })
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(queries, fh, indent=2)
    return queries


if __name__ == "__main__":
    qs = build()
    for q in qs:
        print(q["id"], q["relevance"])
    print(f"wrote {len(qs)} queries to {OUT_PATH}")
