"""
Experiment 2 - Text preprocessing.

Pipeline (applied identically to documents and queries):

    raw text
      -> lowercase
      -> punctuation removal       (non-alphanumeric characters become spaces)
      -> medical synonym mapping   ("stomach ache" -> "abdominal pain", phrase level)
      -> tokenisation              (split on whitespace)
      -> stopword removal          (NLTK English list + a few query filler words)
      -> normalisation             (Porter stemming [from scratch] or WordNet lemmatisation)

Noun groups ("chest pain", "high fever") are identified separately with NLTK POS tags
and a regular-expression chunker; they are shown in the pipeline demonstration and used
by the command-line demo to report the symptom phrases it recognised.
"""

import re
from collections import OrderedDict

from ir.porter_stemmer import PorterStemmer

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "to",
    "from", "in", "on", "is", "am", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "it", "its", "they", "them", "their", "this", "that", "these", "those", "so",
    "very", "too", "can", "will", "just", "not", "no", "nor", "all", "any", "some",
}

# Filler words that appear in plain-language queries but never describe a symptom.
EXTRA_STOPWORDS = {
    "feel", "feeling", "feels", "felt", "getting", "got", "get", "also", "since", "lately",
    "day", "days", "week", "weeks", "lot", "really", "bit", "symptom", "symptoms",
    "sometimes", "like", "keep", "keeps", "kind", "past", "last", "constantly", "always",
    "experiencing", "suffering", "noticed", "think", "seem", "seems", "quite", "every",
    "many", "much", "often", "recently", "started", "s",
    # contractions after apostrophe removal ("I'm" -> "im", "isn't" -> "isnt")
    "im", "ive", "id", "isnt", "dont", "doesnt", "didnt", "wont", "arent", "wasnt", "werent",
    "youre", "theyre", "thats", "its", "hasnt", "havent", "cannot",
    # indefinite pronouns: not in WordNet, so without this the spelling corrector would
    # "fix" them ('something' -> 'vomiting', edit distance 3)
    "something", "anything", "everything", "nothing", "someone", "anyone", "everyone",
    "somewhere", "anywhere", "everywhere", "somehow",
}


def load_stopwords():
    """NLTK's English stopword list (fallback: a built-in list) plus EXTRA_STOPWORDS."""
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    except LookupError:
        words = set(_FALLBACK_STOPWORDS)
    return words | EXTRA_STOPWORDS


STOPWORDS = load_stopwords()

# ---------------------------------------------------------------------------
# Medical synonym map: lay phrase -> dataset vocabulary
# ---------------------------------------------------------------------------
SYNONYMS = {
    # digestive
    "stomach ache": "abdominal pain", "stomachache": "abdominal pain",
    "tummy ache": "abdominal pain", "belly ache": "abdominal pain",
    "tummy pain": "abdominal pain", "cramping stomach": "abdominal pain",
    "throwing up": "vomiting", "threw up": "vomiting", "throw up": "vomiting",
    "puking": "vomiting", "vomit": "vomiting", "vomited": "vomiting",
    "nauseous": "nausea", "queasy": "nausea", "nauseated": "nausea",
    "diarrhea": "diarrhoea", "loose motions": "diarrhoea", "loose stools": "diarrhoea",
    "watery stools": "diarrhoea",
    "constipated": "constipation", "heartburn": "acidity", "acid reflux": "acidity",
    "bloating": "distention of abdomen", "bloated": "distention of abdomen",
    "gas": "passage of gases", "farting": "passage of gases",
    "blood in stool": "bloody stool", "blood in my stool": "bloody stool",
    "no appetite": "loss of appetite", "not hungry": "loss of appetite",
    "always hungry": "excessive hunger", "hungry all the time": "excessive hunger",
    # fever / general
    "high temperature": "high fever", "temperature": "fever", "feverish": "fever",
    "mild temperature": "mild fever", "low grade fever": "mild fever",
    "tired": "fatigue", "tiredness": "fatigue", "exhausted": "fatigue",
    "exhaustion": "fatigue", "sluggish": "lethargy",
    "sweaty": "sweating", "night sweats": "sweating", "shivers": "shivering",
    "losing weight": "weight loss", "lost weight": "weight loss",
    "gaining weight": "weight gain", "gained weight": "weight gain",
    "dehydrated": "dehydration", "unwell": "malaise",
    # respiratory
    "short of breath": "breathlessness", "shortness of breath": "breathlessness",
    "breathing difficulty": "breathlessness", "difficulty breathing": "breathlessness",
    "hard to breathe": "breathlessness", "out of breath": "breathlessness",
    "sore throat": "throat irritation", "scratchy throat": "throat irritation",
    "stuffy nose": "congestion", "blocked nose": "congestion", "stuffed nose": "congestion",
    "sneezes": "continuous sneezing", "sneeze": "continuous sneezing",
    "coughing up blood": "blood in sputum", "mucus": "phlegm",
    "chest tightness": "chest pain", "tight chest": "chest pain",
    # head / neuro
    "head ache": "headache", "head pain": "headache", "headaches": "headache",
    "dizzy": "dizziness", "lightheaded": "dizziness", "light headed": "dizziness",
    "room spinning": "spinning movements", "room is spinning": "spinning movements",
    "blurry vision": "blurred and distorted vision", "blurred vision": "blurred and distorted vision",
    "can't concentrate": "lack of concentration", "cant concentrate": "lack of concentration",
    "trouble concentrating": "lack of concentration", "poor concentration": "lack of concentration",
    "anxious": "anxiety", "depressed": "depression", "irritable": "irritability",
    "racing heart": "fast heart rate", "heart racing": "fast heart rate",
    "rapid heartbeat": "fast heart rate", "heart pounding": "palpitations",
    "slurring": "slurred speech",
    # skin
    "itchy": "itching", "itch": "itching", "itches": "itching",
    "red rash": "skin rash", "red dots": "red spots over body",
    "yellow skin": "yellowish skin", "yellow eyes": "yellowing of eyes",
    "red eyes": "redness of eyes", "watery eyes": "watering from eyes",
    "acne pimples": "pus filled pimples", "peeling skin": "skin peeling",
    # musculoskeletal
    "body ache": "muscle pain", "body aches": "muscle pain", "muscle ache": "muscle pain",
    "muscle aches": "muscle pain", "aching muscles": "muscle pain",
    "joint ache": "joint pain", "aching joints": "joint pain", "sore joints": "joint pain",
    "swollen joints": "swelling joints", "stiff joints": "movement stiffness",
    "back ache": "back pain", "backache": "back pain",
    "neck stiffness": "stiff neck", "swollen feet": "swollen legs",
    # urinary
    "burning urination": "burning micturition", "painful urination": "burning micturition",
    "burning when i pee": "burning micturition", "burning while urinating": "burning micturition",
    "peeing a lot": "polyuria", "frequent urination": "polyuria", "urinating often": "polyuria",
    "smelly urine": "foul smell of urine", "dark pee": "dark urine",
}

_SYNONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(SYNONYMS, key=len, reverse=True)) + r")\b")


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------
def lowercase(text):
    return text.lower()


def remove_punctuation(text):
    """Replace every character that is not a-z, 0-9, an apostrophe or whitespace by a space.
    Apostrophes are dropped afterwards so "can't" -> "cant"."""
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.replace("'", "")


def apply_synonyms(text):
    """Phrase-level replacement, longest phrase first, in a single pass (no chaining)."""
    return _SYNONYM_PATTERN.sub(lambda m: SYNONYMS[m.group(1)], text)


def tokenize(text):
    """Whitespace tokenisation of already-cleaned text."""
    return text.split()


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


_STEMMER = PorterStemmer()
_LEMMATIZER = None
_LEMMA_CACHE = {}


def stem(token):
    """Porter stem (see ir/porter_stemmer.py)."""
    return _STEMMER.stem(token)


def lemmatize(token):
    """WordNet lemma.  The word is lemmatised as a noun and, if that leaves it unchanged,
    as a verb ('rashes' -> 'rash', 'vomiting' -> 'vomit').  A fixed, context-free rule is
    used so that a symptom gets the same lemma in a document list and in a free-text query
    (POS tags of the same word differ between those two contexts)."""
    global _LEMMATIZER
    if token in _LEMMA_CACHE:
        return _LEMMA_CACHE[token]
    if _LEMMATIZER is None:
        from nltk.stem import WordNetLemmatizer
        _LEMMATIZER = WordNetLemmatizer()
    try:
        lemma = _LEMMATIZER.lemmatize(token, "n")
        if lemma == token:
            lemma = _LEMMATIZER.lemmatize(token, "v")
    except LookupError:  # WordNet data not installed
        lemma = token
    _LEMMA_CACHE[token] = lemma
    return lemma


NORMALIZERS = {"stem": stem, "lemma": lemmatize, "none": lambda t: t}


def normalize_tokens(tokens, method="stem"):
    """method: 'stem' (Porter), 'lemma' (WordNet) or 'none'."""
    if method not in NORMALIZERS:
        raise ValueError(f"unknown normalisation method '{method}'")
    fn = NORMALIZERS[method]
    return [fn(t) for t in tokens]


# ---------------------------------------------------------------------------
# Noun groups (POS tagging + regex chunker)
# ---------------------------------------------------------------------------
# Optional adjectives / participles / nouns followed by a head noun:
#   "high/JJ fever/NN", "chest/NN pain/NN", "swollen/VBN legs/NNS"
NOUN_GROUP_GRAMMAR = r"NG: {<JJ.*|VBN|VBG|NN.*>*<NN.*>}"


def pos_tag(tokens):
    """NLTK averaged-perceptron POS tags; [] if the tagger model is not installed."""
    import nltk
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        return []


def extract_noun_groups(text, multiword_only=False):
    """Return noun groups such as ['chest pain', 'high fever'] from free text."""
    import nltk
    # keep punctuation as separate tokens so a comma ends a noun group
    cleaned = apply_synonyms(lowercase(text))
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[.,;:!?()]", cleaned)
    tagged = pos_tag(tokens)
    if not tagged:
        return []
    tree = nltk.RegexpParser(NOUN_GROUP_GRAMMAR).parse(tagged)
    groups = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NG"):
        words = [w for w, _ in subtree.leaves() if w not in STOPWORDS]
        if words and (len(words) > 1 or not multiword_only):
            groups.append(" ".join(words))
    return groups


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess(text, method="stem", use_synonyms=True, remove_stops=True):
    """Text -> list of index terms (the representation used by every retrieval model)."""
    text = remove_punctuation(lowercase(text))
    if use_synonyms:
        text = apply_synonyms(text)
    tokens = tokenize(text)
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return normalize_tokens(tokens, method)


def pipeline_steps(text, method="stem"):
    """Return every intermediate result of the pipeline (for demonstration / viva)."""
    steps = OrderedDict()
    steps["1. original"] = text
    steps["2. lowercased"] = lowercase(text)
    steps["3. punctuation removed"] = remove_punctuation(steps["2. lowercased"])
    steps["4. synonyms mapped"] = apply_synonyms(steps["3. punctuation removed"])
    steps["5. tokens"] = tokenize(steps["4. synonyms mapped"])
    steps["6. stopwords removed"] = remove_stopwords(steps["5. tokens"])
    steps["7. POS tags"] = pos_tag(steps["5. tokens"])
    steps["8. noun groups"] = extract_noun_groups(text)
    steps["9. stemmed (Porter)"] = normalize_tokens(steps["6. stopwords removed"], "stem")
    steps["10. lemmatized (WordNet)"] = normalize_tokens(steps["6. stopwords removed"], "lemma")
    steps["final terms"] = steps["9. stemmed (Porter)"] if method == "stem" else (
        steps["10. lemmatized (WordNet)"] if method == "lemma" else steps["6. stopwords removed"])
    return steps


def print_pipeline(text, method="stem"):
    for name, value in pipeline_steps(text, method).items():
        print(f"{name:<28}: {value}")
