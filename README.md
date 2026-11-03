# GeMoSC: Generational Movie Script Corpus (extension)

GeMoSC, the Generational Movie Script Corpus, was created by Balestrucci, Dusek,
Anselma, and Mazzei (Findings of EMNLP 2025). It pairs the dialogue of film characters
with the gender and year of birth of the actors who play them, from which a generational
cohort is derived.

This repository holds an **extension** of that corpus, produced as the MSc thesis of
Zahra Mohammadinezhad (University of Turin), supervised by Prof. Valerio Basile,
Prof. Alessandro Mazzei, and Pier Felice Balestrucci. The extension adds 896 films
released between 2008 and 2017, annotated with the same scheme, together with the
analysis and classification code from the thesis.

The original corpus (75 films, 2018 to 2021) and its code are released separately by its
authors. See `ORIGINAL_GEMOSC.md`.

## What is here, and what is not

To respect the copyright on the film dialogue, this repository publishes the
**annotations and the code only, never the dialogue text**. The dialogue comes from a
public Kaggle corpus (see `docs/how_to_reproduce.md`), which anyone can download to
regenerate the features locally.

```
data/
  characters.csv            one row per annotated character (10,666 rows): ref_id,
                            IMDb ID, title, year, genres, character, actor, gender,
                            year_of_birth, generation. No dialogue, plot, or synopsis.
  character_features.csv    the 8,184 analysed characters with 45 computed linguistic
                            feature values (statistics, no text).
code/
  gemosc_analysis.py            feature extraction and statistics
  make_classifier_input.py      rebuilds the classifier input from the dialogue and ids
  gemosc_classifiers_rerun.ipynb  the DistilBERT classification experiment
  requirements.txt              dependencies
splits/
  *_test_ids.csv            the held-out test set of each task, by ref_id
results/
  *_predictions.csv         each test character's true and predicted label, by ref_id
                            (no dialogue)
  classifier_results.csv    the summary scores
  significance_tests.csv, summary_*.csv  the feature-level statistics
docs/
  annotation_guidelines.md  the guidelines the annotation followed
  how_to_reproduce.md       how to regenerate everything from the Kaggle source
ORIGINAL_GEMOSC.md          pointer to the original GeMoSC release (not re-hosted here)
```

## Reference identifiers

Every annotated character has a stable `ref_id` (C00001, C00002, ...) used across all
files, so a character in the thesis, the annotations, the split, and the predictions is
always the same row. The IDs are fixed in `data/characters.csv`.

## License

The annotations and code in this repository are released under CC BY 4.0, the same
license as the original GeMoSC. Film dialogue is not included and is not covered by this
license.

## How to cite

Please cite the original corpus:

> Pier Felice Balestrucci, Ondrej Dusek, Luca Anselma, and Alessandro Mazzei. 2025.
> Can Large Language Models Personalize Dialogues to Generational Styles? In Findings of
> the Association for Computational Linguistics: EMNLP 2025, pages 64 to 77.
