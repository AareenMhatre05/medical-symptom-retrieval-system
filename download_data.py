"""
Download the optional external data into data/external/ (standard library only).

  * Symptom2Disease  - gretelai/symptom_to_diagnosis (Hugging Face, Apache-2.0)
                       train.jsonl / test.jsonl / README.md
  * MedlinePlus      - newest health-topics XML from https://medlineplus.gov/xml.html,
                       reduced to the 41 mapped topics in medlineplus_subset.json
                       (the 30 MB XML is deleted afterwards unless --keep-xml is given)

Run:  python download_data.py [--keep-xml]
"""

import argparse
import os
import re
import sys
import urllib.request

from ir.external_data import MEDLINEPLUS_DIR, S2D_DIR, build_medlineplus_subset

HF_BASE = "https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/resolve/main/"
MEDLINEPLUS_INDEX = "https://medlineplus.gov/xml.html"


def fetch(url, path):
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response, open(path, "wb") as out:
        out.write(response.read())
    return path


def download_symptom2disease():
    os.makedirs(S2D_DIR, exist_ok=True)
    for name in ("train.jsonl", "test.jsonl", "README.md"):
        fetch(HF_BASE + name, os.path.join(S2D_DIR, name))


def download_medlineplus(keep_xml=False):
    os.makedirs(MEDLINEPLUS_DIR, exist_ok=True)
    with urllib.request.urlopen(MEDLINEPLUS_INDEX, timeout=60) as response:
        page = response.read().decode("utf-8", errors="replace")
    links = sorted(set(re.findall(r"https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml", page)))
    if not links:
        sys.exit("Could not find a health-topics XML link on " + MEDLINEPLUS_INDEX)
    xml_path = fetch(links[-1], os.path.join(MEDLINEPLUS_DIR, links[-1].rsplit("/", 1)[1]))
    print("  wrote", build_medlineplus_subset(xml_path))
    if not keep_xml:
        os.remove(xml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep-xml", action="store_true", help="keep the full MedlinePlus XML")
    args = parser.parse_args()
    print("Symptom2Disease:")
    download_symptom2disease()
    print("MedlinePlus:")
    download_medlineplus(args.keep_xml)
    print("Done.")
