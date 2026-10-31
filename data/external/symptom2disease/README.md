---
license: apache-2.0
task_categories:
- text-classification
task_ids:
- multi-class-classification
language:
- en
tags:
- medical
pretty_name: Gretel/symptoms_to_diagnosis
size_categories:
- 10K<n<100K
---

# Dataset Summary

This dataset contains natural language descriptions of symptoms labeled with 22 corresponding diagnoses. `Gretel/symptom_to_diagnosis` provides 1065 symptom descriptions in the English language labeled with 22 diagnoses, focusing on fine-grained single-domain diagnosis.

## Data Fields
Each row contains the following fields:
* `input_text` : A string field containing symptoms
* `output_text` : A string field containing a diagnosis

Example:
```
{
"output_text": "drug reaction",
"input_text": "I've been having headaches and migraines, and I can't sleep. My whole body shakes and twitches. Sometimes I feel lightheaded."
}
```

## Diagnoses
This table contains the count of each diagnosis in the train and test splits.

| | Diagnosis | train.jsonl | test.jsonl |
|---:|:--------------------------------|--------------:|-------------:|
| 0 | drug reaction | 40 | 8 |
| 1 | allergy | 40 | 10 |
| 2 | chicken pox | 40 | 10 |
| 3 | diabetes | 40 | 10 |
| 4 | psoriasis | 40 | 10 |
| 5 | hypertension | 40 | 10 |
| 6 | cervical spondylosis | 40 | 10 |
| 7 | bronchial asthma | 40 | 10 |
| 8 | varicose veins | 40 | 10 |
| 9 | malaria | 40 | 10 |
| 10 | dengue | 40 | 10 |
| 11 | arthritis | 40 | 10 |
| 12 | impetigo | 40 | 10 |
| 13 | fungal infection | 39 | 9 |
| 14 | common cold | 39 | 10 |
| 15 | gastroesophageal reflux disease | 39 | 10 |
| 16 | urinary tract infection | 39 | 9 |
| 17 | typhoid | 38 | 9 |
| 18 | pneumonia | 37 | 10 |
| 19 | peptic ulcer disease | 37 | 10 |
| 20 | jaundice | 33 | 7 |
| 21 | migraine | 32 | 10 |

## Data Splits
The data is split to 80% train (853 examples, 167kb) and 20% test (212 examples, 42kb).

## Dataset Creation
Data was filtered to remove unwanted categories and updated using an LLM to create language more consistent with how a patient would describe symptoms in natural language to a doctor.

## Source Data
This dataset was adapted based on the [Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) dataset from Kaggle.

## Personal and Sensitive Information
The symptoms in this dataset were modified from their original format using an LLM and do not contain personal data.

## Limitations
This dataset is licensed Apache 2.0 and free for use.