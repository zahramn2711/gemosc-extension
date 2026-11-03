#!/usr/bin/env python3
"""
gemosc_analysis.py  —  GeMoSC linguistic analysis pipeline
==========================================================
Runs the FULL analysis for the thesis in one go:
  1. Links each annotated character to their dialogue text file
  2. Cleans the dialogue (extracts spoken lines, strips stage directions)
  3. Extracts linguistic features (lexical, sentiment, affective, syntactic, offensive)
  4. Runs group analysis across generation / gender / genre
     - summary tables (means per group)
     - significance tests (Kruskal-Wallis)
     - paper-style heatmaps
  Following the feature families in Frenda et al. (CEUR "Does anyone see the irony here?").

REQUIREMENTS (run once in Terminal):
  pip3 install pandas numpy scipy spacy nrclex textblob matplotlib seaborn rapidfuzz
  pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
  python3 -c "import nltk; [nltk.download(p,quiet=True) for p in ['punkt','punkt_tab','brown','wordnet']]"

FILES NEEDED in the same folder:
  - gemosc_analysis.py            (this script)
  - thesis_data_-_Sheet1.csv      (your annotations)
  - movie_character_texts/        (the folder of movie folders)
  - hurtlex_EN.tsv                (offensive-word lexicon; download instructions below)

  Download hurtlex_EN.tsv once with:
    curl -sL "https://raw.githubusercontent.com/valeriobasile/hurtlex/master/lexica/EN/1.2/hurtlex_EN.tsv" -o hurtlex_EN.tsv

HOW TO RUN:
  python3 gemosc_analysis.py

OUTPUTS (created in ./output/):
  - character_features.csv     one row per character + all features (the master table)
  - summary_generation.csv     mean features per generation
  - summary_gender.csv         mean features per gender
  - summary_genre.csv          mean features per macro-genre
  - significance_tests.csv     Kruskal-Wallis results (which features differ significantly)
  - heatmap_generation.png     paper-style heatmap
  - heatmap_gender.png
  - heatmap_genre.png
"""

import os, re, sys, glob, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
CSV_PATH   = "thesis_data_-_Sheet1.csv"
TEXTS_ROOT = "movie_character_texts"
HURTLEX    = "hurtlex_EN.tsv"
OUTDIR     = "output"
FUZZY_THRESHOLD = 85
MIN_TOKENS = 20          # drop characters with < this many spoken words (too little signal)
MIN_GROUP  = 30          # groups smaller than this are excluded from significance tests
# ----------------------------

os.makedirs(OUTDIR, exist_ok=True)

print("Loading libraries (spaCy, NRCLex, TextBlob)...")
import spacy
from nrclex import NRCLex
from textblob import TextBlob
from rapidfuzz import process, fuzz
from scipy.stats import kruskal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("en_core_web_sm")

# ============================================================
# 1. LINK DIALOGUE
# ============================================================
def pad_id(x):
    x = re.sub(r"\D","",str(x).strip()); return x.zfill(7) if x else ""
def norm(s):
    return re.sub(r"\s+"," ",str(s).strip().lower())

print("\n[1/4] Linking dialogue to annotations...")
df = pd.read_csv(CSV_PATH, dtype=str)
df.columns = [c.strip() for c in df.columns]
df["imdb_padded"] = df["IMDb ID"].apply(pad_id)

folder_by_id = {}
for name in os.listdir(TEXTS_ROOT):
    full = os.path.join(TEXTS_ROOT,name)
    if os.path.isdir(full):
        m = re.search(r"_(\d{6,8})$", name.strip())
        if m: folder_by_id[m.group(1).zfill(7)] = full

def char_files(folder):
    out={}
    for f in os.listdir(folder):
        if f.lower().endswith("_text.txt"):
            out[f[:-len("_text.txt")]] = os.path.join(folder,f)
    return out

def extract_speech(raw):
    parts=[]
    for line in raw.splitlines():
        m=re.match(r'\s*\d+\)\s*\d+\)\s*dialog:\s*(.*)',line)
        if not m: continue
        c=m.group(1).strip()
        if not c: continue
        c=re.split(r'\b(?:INT|EXT)\.?\s',c)[0]
        c=re.sub(r'\(\s*CONTINUED\s*\)','',c,flags=re.I)
        c=re.sub(r'\([^)]*\)','',c)
        c=re.sub(r'\b\d+(\s+\d+)*\.?\b','',c)
        c=re.sub(r'\b[A-Z]{2,}(?:\'S)?\b','',c)
        c=re.sub(r'\s+',' ',c).strip(' .-')
        if c: parts.append(c)
    return ' '.join(parts)

dialogues=[]
for _,row in df.iterrows():
    folder=folder_by_id.get(row["imdb_padded"])
    text=""
    if folder:
        files=char_files(folder)
        if files:
            lookup={norm(k):k for k in files}
            hit=lookup.get(norm(str(row["character"])))
            if hit is None:
                best=process.extractOne(norm(str(row["character"])),list(lookup.keys()),scorer=fuzz.ratio)
                if best and best[1]>=FUZZY_THRESHOLD: hit=lookup[best[0]]
            if hit:
                try: text=extract_speech(open(files[hit],encoding="utf-8",errors="ignore").read())
                except: text=""
    dialogues.append(text)
df["clean_dialogue"]=dialogues
df["n_tokens"]=df["clean_dialogue"].apply(lambda t: len(t.split()))
linked=df[df["n_tokens"]>=MIN_TOKENS].copy()
print(f"   linked & above {MIN_TOKENS} tokens: {len(linked)}/{len(df)} characters")

# ============================================================
# 2. NORMALISE GROUPING VARIABLES
# ============================================================
print("[2/4] Normalising grouping variables (gender, genre)...")
# gender
linked["gender_norm"]=linked["gender"].str.strip().str.lower().replace(
    {"m":"male","f":"female"})
linked.loc[~linked["gender_norm"].isin(["male","female"]),"gender_norm"]=np.nan
# generation - keep as-is, drop missing later
linked["generation"]=linked["generation"].str.strip()
# genre -> macro
def macro_genre(g):
    if pd.isna(g) or not str(g).strip(): return np.nan
    # EXCLUDE animated films entirely (guideline: human characters only, no voice roles).
    # Any film tagged Animation anywhere in its genre string is dropped.
    if "animation" in str(g).lower(): return np.nan
    first=str(g).split(",")[0].strip().lower()
    M={"comedy":"Comedy","romantic comedy":"Comedy",
       "drama":"Drama",
       "biography":"Biography",
       "action":"Action/Adventure","adventure":"Action/Adventure","thriller":"Action/Adventure",
       "crime":"Crime",
       "horror":"Horror",
       "family":"Drama",
       "sci-fi":"SciFi/Fantasy","fantasy":"SciFi/Fantasy",
       "romance":"Romance","documentary":"Documentary","mystery":"Action/Adventure"}
    return M.get(first, np.nan)   # rare/ambiguous genres -> excluded
linked["genre_macro"]=linked["genres"].apply(macro_genre)
# Also drop animation-tagged rows from ALL analyses (not just genre), since the
# generation/gender comparisons should be live-action only too.
linked=linked[~linked["genres"].fillna("").str.contains("animation",case=False)].copy()

# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================
print("[3/4] Extracting linguistic features (this is the slow step)...")

# HurtLex with POS filter + stoplist
hl=pd.read_csv(HURTLEX,sep="\t"); hl=hl[hl["level"]=="conservative"]
POSMAP={"n":"NOUN","a":"ADJ","v":"VERB","r":"ADV"}
STOP=set("""know knows knew do does did done love loves loved mark marks con cons
bed beds tool tools fool fools fooled dope sap mug mock jazz gull rook bang boob
eff effs gip gyp hump""".split())
hurt_entries={}
for _,r in hl.iterrows():
    lw=str(r["lemma"]).lower()
    if lw in STOP: continue
    hurt_entries.setdefault(lw,[]).append((str(r["pos"]),r["category"]))
HURT_CATS=sorted(hl["category"].unique())
EMOTIONS=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]

# --- register / informality lexicons (generation-sensitive features) ---
CONTRACTIONS=re.compile(r"\b(gonna|wanna|gotta|kinda|sorta|dunno|lemme|gimme|ain't|"
    r"y'all|'cause|cuz|coulda|shoulda|woulda|outta|lotta|hafta)\b", re.I)
DISCOURSE={"like","well","so","anyway","actually","basically","literally","totally",
    "really","just","right","okay","ok","yeah","yep","nope","hey"}
INTENSIFIERS={"very","really","so","totally","absolutely","completely","super",
    "way","damn","freaking","fucking","hella","dead","mad"}
FILLERS={"uh","um","er","hmm","huh","oh","ah","wow","ugh","eh","oops"}
PROFANITY={"damn","hell","shit","fuck","fucking","ass","bitch","bastard","crap",
    "piss","dick","cock","asshole","bullshit","goddamn"}
try:
    SLANG=set(open("slang_list.txt").read().split())
except FileNotFoundError:
    print("   [warn] slang_list.txt not found - slang_per100 will be 0. "
          "Generate it once with the build_slang_list step.")
    SLANG=set()

def depth(tok):
    d=0;cur=tok
    while cur.head!=cur:
        d+=1;cur=cur.head
        if d>50:break
    return d

def features(text):
    f={}
    doc=nlp(text)
    words=[t.text.lower() for t in doc if t.is_alpha]
    n=len(words); f["n_tokens"]=n
    f["ttr"]=len(set(words))/n if n else 0
    f["n_interjections"]=sum(1 for t in doc if t.pos_=="INTJ")
    f["n_negations"]=sum(1 for t in doc if t.dep_=="neg")
    f["n_named_entities"]=len(doc.ents)
    f["n_adverbs"]=sum(1 for t in doc if t.pos_=="ADV")
    f["mean_tree_depth"]=(sum(depth(t) for t in doc)/len(doc)) if len(doc) else 0
    # rates per 100 tokens (comparable across characters of different lengths)
    for k in ["n_interjections","n_negations","n_named_entities","n_adverbs"]:
        f[k+"_per100"]=100*f[k]/n if n else 0
    # affective
    nrc=NRCLex(); nrc.load_raw_text(text); nrc._build_word_affect()
    emo=nrc.affect_frequencies
    for e in EMOTIONS: f["emo_"+e]=emo.get(e,0.0)
    # sentiment
    tb=TextBlob(text).sentiment
    f["sent_polarity"]=tb.polarity; f["sent_subjectivity"]=tb.subjectivity
    # offensive (POS-filtered, stoplisted)
    from collections import Counter
    cat_hits=Counter()
    for t in doc:
        lw=t.text.lower()
        if lw in hurt_entries:
            for hp,cat in hurt_entries[lw]:
                want=POSMAP.get(hp)
                if want is None or t.pos_==want:
                    cat_hits[cat]+=1; break
    for cat in HURT_CATS: f["hurt_"+cat]=100*cat_hits.get(cat,0)/n if n else 0
    # register / informality (generation markers)
    f["contraction_per100"]=100*len(CONTRACTIONS.findall(text))/n if n else 0
    f["discourse_per100"]=100*sum(1 for w in words if w in DISCOURSE)/n if n else 0
    f["intensifier_per100"]=100*sum(1 for w in words if w in INTENSIFIERS)/n if n else 0
    f["filler_per100"]=100*sum(1 for w in words if w in FILLERS)/n if n else 0
    f["profanity_per100"]=100*sum(1 for w in words if w in PROFANITY)/n if n else 0
    f["slang_per100"]=100*sum(1 for w in words if w in SLANG)/n if n else 0
    f["mean_word_len"]=sum(len(w) for w in words)/n if n else 0
    return f

feats=[]
total=len(linked)
for i,(_,row) in enumerate(linked.iterrows(),1):
    feats.append(features(row["clean_dialogue"]))
    if i%500==0: print(f"   {i}/{total} characters processed...")
featdf=pd.DataFrame(feats,index=linked.index)
# merge metadata + features
meta_cols=["IMDb ID","title","year","character","actor","gender_norm",
           "generation","genre_macro"]
result=pd.concat([linked[meta_cols],featdf],axis=1)
result.to_csv(f"{OUTDIR}/character_features.csv",index=False)
print(f"   saved {OUTDIR}/character_features.csv ({len(result)} rows)")

# feature columns for analysis
FEATURE_COLS=[c for c in featdf.columns if c not in ("n_tokens",)] \
    + ["n_tokens"]
# use the "meaningful" analysis features
ANALYSIS_FEATS=["n_tokens","ttr","mean_tree_depth","mean_word_len","sent_polarity","sent_subjectivity",
    "n_interjections_per100","n_negations_per100","n_named_entities_per100","n_adverbs_per100",
    "contraction_per100","discourse_per100","intensifier_per100","filler_per100",
    "profanity_per100","slang_per100"] \
    + ["emo_"+e for e in EMOTIONS] \
    + ["hurt_"+c for c in HURT_CATS]

# ============================================================
# 4. GROUP ANALYSIS
# ============================================================
print("[4/4] Group analysis (tables, tests, heatmaps)...")

def analyse(dimension, valid_order=None):
    sub=result.dropna(subset=[dimension])
    groups=sub[dimension].value_counts()
    keep=groups[groups>=MIN_GROUP].index.tolist()
    if valid_order: keep=[g for g in valid_order if g in keep]
    sub=sub[sub[dimension].isin(keep)]
    dname=dimension.replace('_norm','').replace('_macro','')
    if len(keep)<2:
        print(f"   [skip] {dname}: fewer than 2 groups with >= {MIN_GROUP} characters")
        return pd.DataFrame(), keep
    # summary table: mean per group
    summary=sub.groupby(dimension)[ANALYSIS_FEATS].mean().T
    summary.to_csv(f"{OUTDIR}/summary_{dname}.csv")
    # significance: Kruskal-Wallis per feature
    rows=[]
    for feat in ANALYSIS_FEATS:
        data=[sub[sub[dimension]==g][feat].dropna() for g in keep]
        if all(len(d)>0 for d in data):
            try:
                H,p=kruskal(*data)
                rows.append({"feature":feat,"dimension":dimension,"H":H,"p":p,
                             "significant":p<0.05})
            except: pass
    sig=pd.DataFrame(rows)
    # heatmap: z-scored group means
    z=summary.sub(summary.mean(axis=1),axis=0).div(summary.std(axis=1)+1e-9,axis=0)
    z=z.dropna(how="all")
    if not z.empty:
        plt.figure(figsize=(max(6,1.2*len(keep)),0.35*len(z)+2))
        sns.heatmap(z,cmap="RdBu_r",center=0,cbar_kws={"label":"z-score of group mean"},
                    linewidths=.3,linecolor="#eee")
        plt.title(f"Linguistic features by {dname}")
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/heatmap_{dname}.png",dpi=150)
        plt.close()
    return sig, keep

GEN_ORDER=["Silent","Boomer","Gen X","Millennial","Gen Z"]
all_sig=[]
for dim,order in [("generation",GEN_ORDER),("gender_norm",None),("genre_macro",None)]:
    sig,keep=analyse(dim,order)
    all_sig.append(sig)
    print(f"   {dim}: groups analysed = {keep}")
pd.concat(all_sig).to_csv(f"{OUTDIR}/significance_tests.csv",index=False)

# ============================================================
# 5. TF-IDF DISTINCTIVE N-GRAMS (paper 2, section 4.1.4)
#    Surfaces the most group-distinctive words/phrases empirically
#    -> this is where generation-specific slang/vocabulary shows up.
# ============================================================
print("[5/5] TF-IDF distinctive n-grams per group...")
from sklearn.feature_extraction.text import TfidfVectorizer

def top_ngrams(dimension, top_k=25, valid_order=None):
    sub=result.dropna(subset=[dimension]).copy()
    # attach dialogue text back
    sub=sub.join(linked["clean_dialogue"])
    groups=sub[dimension].value_counts()
    keep=groups[groups>=MIN_GROUP].index.tolist()
    if valid_order: keep=[g for g in valid_order if g in keep]
    if len(keep)<2: return
    dname=dimension.replace('_norm','').replace('_macro','')
    # Build a stoplist of proper nouns / character names so vocabulary (not names)
    # surfaces. Character column gives the names directly; also strip spaCy PROPN.
    name_stop=set()
    for c in linked["character"].dropna():
        for tok in re.split(r"\s+", str(c).lower()):
            if tok.isalpha(): name_stop.add(tok)
    # remove proper nouns from each group document using spaCy
    def strip_propn(text):
        doc=nlp(text)
        return " ".join(t.text for t in doc
                        if t.pos_!="PROPN" and t.text.lower() not in name_stop)
    docs=[strip_propn(" ".join(sub[sub[dimension]==g]["clean_dialogue"].fillna("")))
          for g in keep]
    ndocs=len(docs)
    min_df=1 if ndocs<5 else 2
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop=list(ENGLISH_STOP_WORDS.union(name_stop))
    vec=TfidfVectorizer(ngram_range=(1,3),min_df=min_df,max_df=1.0,stop_words=stop,
                        max_features=5000,sublinear_tf=True)
    X=vec.fit_transform(docs)
    terms=vec.get_feature_names_out()
    rows=[]
    for i,g in enumerate(keep):
        scores=X[i].toarray().ravel()
        top=scores.argsort()[::-1][:top_k]
        for rank,idx in enumerate(top,1):
            rows.append({"group":g,"rank":rank,"ngram":terms[idx],
                         "tfidf":round(float(scores[idx]),4)})
    pd.DataFrame(rows).to_csv(f"{OUTDIR}/tfidf_ngrams_{dname}.csv",index=False)

for dim,order in [("generation",GEN_ORDER),("gender_norm",None),("genre_macro",None)]:
    top_ngrams(dim,valid_order=order)

print("\nDONE. See the ./output/ folder:")
for f in sorted(os.listdir(OUTDIR)): print("  -",f)
print("\nUpload the whole ./output/ folder back to review the results together.")
