#!/usr/bin/env python3
"""
make_classifier_input.py
------------------------
Produces classifier_input.csv: one row per analysed character, carrying the
stable reference id (ref_id), the labels, and the cleaned dialogue the DistilBERT
classifier reads.

The ref_ids and labels are taken from character_features_with_refids.csv (the
canonical file, 8,184 analysed characters). This script does NOT renumber anything:
it only re-reads the dialogue text from your movie_character_texts folders and
attaches it to each character by matching on (IMDb ID, character). That keeps every
id identical to the ones already in the thesis and in the released data.

RUN THIS LOCALLY, in the folder that contains:
  - character_features_with_refids.csv   (shipped to you; the canonical ids+labels)
  - movie_character_texts/               (your per-film dialogue folders)

Usage:
  python3 make_classifier_input.py

Output:
  classifier_input.csv   (ref_id, IMDb ID, title, character, gender_norm,
                          generation, genre_macro, clean_dialogue)
It prints how many of the 8,184 characters got their dialogue back. That number
should be 8,184; if any are missing, it lists a few so you can check the folder.
"""
import os, re
import pandas as pd

CANON      = "character_features_with_refids.csv"
TEXTS_ROOT = "movie_character_texts"

def pad_id(x):
    x = re.sub(r"\D", "", str(x).strip()); return x.zfill(7) if x else ""
def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def extract_speech(raw):
    # identical cleaning logic to gemosc_analysis.py
    parts = []
    for line in raw.splitlines():
        m = re.match(r'\s*\d+\)\s*\d+\)\s*dialog:\s*(.*)', line)
        if not m: continue
        c = m.group(1).strip()
        if not c: continue
        c = re.split(r'\b(?:INT|EXT)\.?\s', c)[0]
        c = re.sub(r'\(\s*CONTINUED\s*\)', '', c, flags=re.I)
        c = re.sub(r'\([^)]*\)', '', c)
        c = re.sub(r'\b\d+(\s+\d+)*\.?\b', '', c)
        c = re.sub(r"\b[A-Z]{2,}(?:\'S)?\b", '', c)
        c = re.sub(r'\s+', ' ', c).strip(' .-')
        if c: parts.append(c)
    return ' '.join(parts)

# --- canonical ids + labels (no renumbering) ---
canon = pd.read_csv(CANON, dtype=str).fillna("")
canon["imdb_padded"] = canon["IMDb ID"].apply(pad_id)
print(f"Loaded {len(canon)} analysed characters with fixed ref_ids.")

# --- index the dialogue folders ---
folder_by_id = {}
for name in os.listdir(TEXTS_ROOT):
    full = os.path.join(TEXTS_ROOT, name)
    if os.path.isdir(full):
        m = re.search(r"_(\d{6,8})$", name.strip())
        if m: folder_by_id[m.group(1).zfill(7)] = full

def char_files(folder):
    out = {}
    for f in os.listdir(folder):
        if f.lower().endswith("_text.txt"):
            out[f[:-len("_text.txt")]] = os.path.join(folder, f)
    return out

# --- re-read each character's dialogue and attach by (IMDb ID, character) ---
try:
    from rapidfuzz import process, fuzz
    HAVE_FUZZ = True
except ImportError:
    HAVE_FUZZ = False
    print("note: rapidfuzz not installed; exact name matching only.")
FUZZY_THRESHOLD = 85

texts = []
for _, row in canon.iterrows():
    folder = folder_by_id.get(row["imdb_padded"]); text = ""
    if folder:
        files = char_files(folder)
        if files:
            lookup = {norm(k): k for k in files}
            hit = lookup.get(norm(row["character"]))
            if hit is None and HAVE_FUZZ:
                best = process.extractOne(norm(row["character"]),
                                          list(lookup.keys()), scorer=fuzz.ratio)
                if best and best[1] >= FUZZY_THRESHOLD: hit = lookup[best[0]]
            if hit:
                try: text = extract_speech(open(files[hit], encoding="utf-8",
                                                errors="ignore").read())
                except: text = ""
    texts.append(text)

canon["clean_dialogue"] = texts
got = (canon["clean_dialogue"].str.len() > 0).sum()
print(f"Attached dialogue for {got} / {len(canon)} characters.")
if got < len(canon):
    miss = canon[canon["clean_dialogue"].str.len() == 0]
    print("Missing a few (check these folders/names):")
    print(miss[["ref_id","IMDb ID","title","character"]].head(10).to_string(index=False))

out = canon[["ref_id","IMDb ID","title","character","gender_norm","generation",
             "genre_macro","clean_dialogue"]]
out.to_csv("classifier_input.csv", index=False)
print("Wrote classifier_input.csv")
