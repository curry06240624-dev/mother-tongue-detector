"""Mother-Tongue Detector: demo app.

First run the notebook (it trains the model and saves models/mother_tongue_detector.joblib), then:
    python app.py
and open the address it prints.
"""
import os
import re
import unicodedata

import gradio as gr
import joblib
import numpy as np
from collections import Counter
from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0
HERE = os.path.dirname(os.path.abspath(__file__))
HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134F]")


def clean(text):
    """Keep only the Chinese characters (the model was trained on text cleaned the same way)."""
    return "".join(HAN.findall(unicodedata.normalize("NFKC", text)))


bundle = joblib.load(os.path.join(HERE, "models", "mother_tongue_detector.joblib"))
model, THRESHOLD, MIN_CHARS = bundle["model"], bundle["threshold"], bundle["min_chars"]
vec = model.named_steps["tfidfvectorizer"]
clf = model.named_steps["logisticregression"]
CLASSES = list(clf.classes_)
NAMES = {"Mandarin": "Mandarin 華語", "Taiwanese Hokkien": "Taiwanese Hokkien 台語",
         "Hakka": "Hakka 客語", "Cantonese": "Cantonese 粵語"}
LANGDETECT_NAMES = {"ko": "Korean", "zh-tw": "Chinese", "zh-cn": "Chinese", "ja": "Japanese", "vi": "Vietnamese"}

# Real sentences from the test set (the model never saw them). Pairs are one dictionary example + its translation.
EXAMPLES = ["阮先生娘生做真媠。",            # Taiwanese Hokkien: "Our teacher's wife is very pretty."
            "我們的師母長得很美。",          # the same sentence in Mandarin
            "該家人為著防賊仔，起盡高个圍牆。",  # Hakka: "That family built a very high wall to keep thieves out."
            "那家人為了防小偷，蓋很高的圍牆。",  # the same sentence in Mandarin
            "我喺屋企同屋企人一齊嗰陣最開心。"]  # Cantonese: "I'm happiest at home with my family."


def char_evidence(text, k):
    """How much each character pushed the model towards class k (positive) or away from it (negative)."""
    x = vec.transform([text])
    lo, hi = vec.ngram_range
    grams = [(i, n, text[i:i + n]) for n in range(lo, hi + 1) for i in range(len(text) - n + 1)]
    counts = Counter(g for _, _, g in grams)
    score = np.zeros(len(text))
    for i, n, g in grams:
        j = vec.vocabulary_.get(g)
        if j is not None:
            weight = clf.coef_[k, j] - clf.coef_[:, j].mean()
            score[i:i + n] += x[0, j] * weight / counts[g] / n
    return score


def langdetect_says(sentence):
    try:
        best = detect_langs(sentence)[0]
        return f"{LANGDETECT_NAMES.get(best.lang, best.lang)} ({best.prob:.0%} sure)"
    except LangDetectException:
        return "no answer"


def run(sentence):
    text = clean(sentence)
    other = f"For comparison, the off-the-shelf detector **langdetect** says: **{langdetect_says(sentence)}**" \
        if sentence.strip() else ""
    if len(text) < MIN_CHARS:
        return (f"## Too short to tell\nPlease type at least {MIN_CHARS} Chinese characters.", {}, [], other)
    p = model.predict_proba([text])[0]
    k = int(p.argmax())
    lang = CLASSES[k]
    ev = char_evidence(text, k)
    scale = max(np.abs(ev).max(), 1e-9)
    highlighted = [(ch, float(v / scale)) for ch, v in zip(text, ev)]
    probs = {NAMES[c]: float(p[i]) for i, c in enumerate(CLASSES)}
    if p.max() < THRESHOLD:
        verdict = (f"## Not sure\nBest guess: {NAMES[lang]} ({p.max():.0%}). "
                   "It may be another language or a mix, so a person should check.")
    else:
        verdict = f"## {NAMES[lang]}\n{p.max():.0%} sure"
    return verdict, probs, highlighted, other


with gr.Blocks(title="Mother-Tongue Detector") as demo:
    gr.Markdown("# Mother-Tongue Detector · 母語偵測器\n"
                "Is a sentence **Mandarin, Taiwanese Hokkien, Hakka or Cantonese**? "
                "All four are written with the same Chinese characters.")
    with gr.Row():
        with gr.Column(scale=1):
            box = gr.Textbox(label="Sentence", lines=3, placeholder="Paste a sentence written in Chinese characters")
            button = gr.Button("Detect language", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=box,
                        label="Real test sentences the model never saw (pairs = the same meaning in two languages)")
        with gr.Column(scale=1):
            verdict = gr.Markdown("## …")
            probs = gr.Label(label="How likely each language is", num_top_classes=4)
            langdetect_box = gr.Markdown()
    evidence = gr.HighlightedText(
        label="Which characters decided it (red = pushed towards the answer, blue = pushed against it)",
        show_legend=False)
    gr.Markdown("*Prototype built for a 48-hour hackathon. Trained on example sentences from Taiwan's Ministry of "
                "Education Hokkien and Hakka dictionaries and on Tatoeba. On held-out test sentences of 4+ characters "
                "it is right about 94% of the time; it is weaker on very short texts and on Hakka written in "
                "non-standard characters.*")
    outputs = [verdict, probs, evidence, langdetect_box]
    button.click(run, box, outputs)
    box.submit(run, box, outputs)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
