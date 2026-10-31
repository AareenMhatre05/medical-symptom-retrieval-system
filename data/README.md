# Data and licences

The code in this repository and the data in this folder are licensed separately.
Each dataset below keeps its own licence.

## `archive/` — Disease Symptom Prediction (CC BY-SA 4.0)

- **Title:** Disease Symptom Prediction
- **Creator:** Pranay Patil (Kaggle user `itachi9604`), created with Pratik Rathod
- **Source:** https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- **Licence:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)

The four files in `archive/` (`dataset.csv`, `Symptom-severity.csv`,
`symptom_Description.csv`, `symptom_precaution.csv`) are the **original files, unmodified**.

## `processed_documents.csv` — adapted from the dataset above (CC BY-SA 4.0)

This file is an adaptation of Disease Symptom Prediction by Pranay Patil, produced by
`ir/data_loader.py`. In line with the ShareAlike condition, it is shared under the same
licence, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Changes made to the original data:

- The 4,920 rows of `dataset.csv` (about 120 rows per disease) were merged into one row
  per disease, holding the union of that disease's symptoms (41 rows).
- Symptom names were cleaned: leading and trailing spaces removed, underscores replaced
  by spaces, repeated spaces collapsed (`" skin_rash"` → `skin rash`).
- Disease names were normalised across the files. The spelling "Dimorphic hemmorhoids(piles)"
  in `dataset.csv` was matched to "Dimorphic hemorrhoids(piles)" in the description file,
  and the description file's spelling is used.
- Descriptions and precautions from the other two files were joined to each disease.
  Typographic dashes and quotes were converted to ASCII, a missing space after full stops
  was added, and the first letter of each precaution was capitalised.
- A `text` column was added: disease name + description + symptom list.

The severity weights from `Symptom-severity.csv` are used unchanged at run time
(duplicate entries keep their highest weight).

## `external/symptom2disease/` — Symptom2Disease (Apache-2.0)

- **Source:** `gretelai/symptom_to_diagnosis`, https://huggingface.co/datasets/gretelai/symptom_to_diagnosis
- **Licence:** Apache License 2.0. The full text is in `external/symptom2disease/LICENSE`.
- `train.jsonl`, `test.jsonl` and `README.md` (the dataset card) are unmodified copies.

## `external/medlineplus/` — MedlinePlus health topics

- **Source:** U.S. National Library of Medicine, MedlinePlus Health Topic XML,
  https://medlineplus.gov/xml.html
- Information from MedlinePlus.gov.
- `medlineplus_subset.json` is an extract of 41 health topics (topic title, URL and
  "also called" names), created by `download_data.py`.
