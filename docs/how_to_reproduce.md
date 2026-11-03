# How to reproduce the features and results

Because the film dialogue is copyrighted, it is not included here. The pipeline
regenerates it locally from a public source, so nothing copyrighted is redistributed.

## 1. Get the dialogue

Download the Movie Scripts Corpus from Kaggle:
https://www.kaggle.com/datasets/gufukuro/movie-scripts-corpus

It is organized as one folder per film, named `{Title}_{IMDbID}`, with one plain-text
file per character named `{Character}_text.txt`.

## 2. Get the lexical resources

- **HurtLex** (offensive-language lexicon): download the English conservative list from
  the HurtLex project and point `gemosc_analysis.py` at it.
- **Slang list**: build it once from the Urban Dictionary subset of Sun et al. (2021),
  restricted to single-word entries and filtered against the Brown corpus to drop common
  words. The resulting `slang_list.txt` is read by `gemosc_analysis.py`.

## 3. Install dependencies

```
pip install pandas numpy scipy spacy nrclex textblob matplotlib seaborn rapidfuzz scikit-learn statsmodels
python -m spacy download en_core_web_sm
```

## 4. Run the feature extraction and statistics

Place the annotation sheet (with the columns of `data/characters.csv`) and the
`movie_character_texts` folder in the working directory, then:

```
python code/gemosc_analysis.py
```

This links each character to their dialogue, cleans the dialogue, extracts the 45
features, and runs the Kruskal-Wallis tests. It reproduces `character_features.csv`,
`significance_tests.csv`, and the `summary_*.csv` files.

## 5. Reproduce the classifier

```
python code/make_classifier_input.py        # builds classifier_input.csv (ref_id + dialogue)
```

Then open `gemosc_classifiers_rerun.ipynb` in Google Colab, set the runtime to a T4 GPU,
upload `classifier_input.csv`, and run all cells. It reproduces the train/test splits
(fixed with random seed 42, grouped by film), the per-character predictions, the
confusion matrices, and `classifier_results.csv`.

Note: a fresh neural training run can differ from the published scores by a point or two,
since fine-tuning has its own randomness. The split itself is fixed by the seed and
recorded in the `*_test_ids.csv` files.
