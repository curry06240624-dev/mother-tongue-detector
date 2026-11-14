# Mother-Tongue Detector (母語偵測器)

**Track:** Languages
**Team:** Yu-Rui Huang

## The problem
In Hsinchu County, where nearly 7 in 10 people are Hakka, you hear Mandarin, Hakka (客語) and Taiwanese Hokkien (台語) every day. Hakka and Hokkien are national languages of Taiwan and are taught in schools, and all three languages are written with the same Chinese characters. Software can't tell them apart. The popular open-source detector `langdetect` labels 44–49% of our Hokkien and Hakka test sentences as **Korean**, and most of the rest as plain "Chinese". It even calls 49% of the Mandarin sentences Korean. Every language tool (reading text aloud, translating it, passing a message to the right helper) starts by asking "which language is this?", so Hokkien and Hakka text gets the wrong tool from the first step.

## What we built
A small AI model that reads a sentence written in Chinese characters and answers **Mandarin, Taiwanese Hokkien, Hakka or Cantonese**. It shows how sure it is and which characters decided it. If the text is under 4 characters or the model isn't confident, it says "too short" or "not sure" instead of guessing.

**Who it's for:** a volunteer at a help desk in Hsinchu who receives written messages and has to pass each one to someone who reads that language. A second user is the builder of a read-aloud tool, who must pick the right voice before reading a sentence out. Both need this tool at the same moment: when new text arrives.

Try it: the notebook's last section opens the app in Colab, or run `python app.py`. The demo video is linked on the submission form.

## What we added
We started from suggested project **4.4.1 Language identification** and added:
- **Our own languages and data.** We used Taiwan's languages, with example sentences from the Ministry of Education's Hokkien and Hakka dictionaries. Every example comes with its Mandarin translation. The model sees the same meaning in two languages, so it can't cheat by learning topics.
- **Our own question.** Can a model separate languages that share one writing system, and when does it break? We tested short texts, sentences from a different source (Tatoeba), and a language the model had never seen (Cantonese).
- **Our own fixes.** "Too short" and "not sure" answers, and a fourth language (Cantonese). We measured the cost of each change.

## How it works
**Data.**
- After cleaning, there are 58,845 unique dictionary texts: 14,960 Hokkien and 18,705 Hakka examples, plus 25,180 Mandarin translations.
- For the 4-language version we added 17,120 Cantonese and 14,878 Mandarin sentences from Tatoeba.

**Cleaning.**
- We keep only the Chinese characters, because punctuation differs between sources.
- We drop characters that have no standard code.
- We drop 979 texts that appear in more than one language (for example 了解, which is both Mandarin and Hokkien).
- We remove duplicates.

**Splits.** We split the data by **dictionary entry**: 70% train, 15% validation, 15% test. That way, a sentence and its translation always land in the same split.

**Model.** Each sentence becomes counts of single characters and character pairs, weighted by TF-IDF. A logistic regression learns how strongly each one points to each language. We chose it because it trains in seconds and every decision can be explained character by character. We tried pairs vs triples and different regularisation strengths, and chose using the validation set only.

## Results
These are on the held-out test set of 9,141 dictionary texts. "Recall" is the share of each language the method finds.

| Method | Accuracy | Macro F1 | Hokkien recall | Hakka recall |
|---|---|---|---|---|
| Always "Mandarin" | 42.2% | 19.8% | 0% | 0% |
| `langdetect` (off-the-shelf) | 21.4% | 15.4% | 0% | 0% |
| Unique-character rule (no ML) | 54.9% | 45.8% | 22.4% | 21.7% |
| **Our model** | **85.7%** | **84.6%** | **79.0%** | **81.6%** |

On texts of 4+ characters (the ones the tool answers), it is right **94.5%** of the time, and 97.9% for 7+ characters.

**Where it fails, and why we think so:**
- **Very short texts: 54% on 2–3 characters.** Many short words are the same in all three languages, so the tool now answers "too short" for them.
- **Shared grammar words.** 毋 ("not") is used in both Hokkien and Hakka. The Hokkien sentence 覺悟著過去做了毋著 was called Hakka with 90% confidence.
- **Text from another source.** On Tatoeba sentences of 4+ characters, Hokkien held at 92%, but Hakka fell to 64%. That set has only 61 sentences, so treat it with care. Many of them use spellings outside the dictionary standard, such as 涯 instead of 𠊎 ("I") and 唔 instead of 毋 ("not").
- **A language it never saw.** The 3-language model labelled Cantonese as Mandarin (65%) or Hakka (29%), with a median confidence of 83%. A "not sure" threshold caught only 3.5% of them.
  - Adding Cantonese as a fourth language fixed this: it finds 97.1% of Cantonese, and the dictionary test barely moved (94.4% vs 94.5%).
  - But non-standard Hakka got worse (64% → 38%). That's mostly because a fifth of those sentences are now called Cantonese: 唔 is also the Cantonese word for "not".

Numbers come from our run on 14 Nov 2026. Tatoeba is updated weekly, so a re-run may differ slightly.

## What we'd do next
1. Collect everyday Hakka and Hokkien writing (with permission), and teach the model common spelling variants such as 涯/𠊎 and 唔/毋.
2. Ask Hakka and Hokkien speakers in Hsinchu to check 100 predictions and to try the tool on messages they really write. Also handle messages that mix two languages.

## Sources
- **Data:**
  - [MOE Dictionary of Frequently-Used Taiwanese Hokkien](https://twblg.dict.edu.tw) and [MOE Dictionary of Frequently-Used Taiwan Hakka](https://hakkadict.moe.edu.tw). Both © Ministry of Education, CC BY-ND 3.0 TW, read through [g0v/moedict-data-twblg](https://github.com/g0v/moedict-data-twblg) and [g0v/moedict-data-hakka](https://github.com/g0v/moedict-data-hakka).
  - [Tatoeba](https://tatoeba.org/en/downloads), CC BY 2.0 FR.
  - The notebook downloads all data at run time; none of it is stored in this repository.
- **Libraries:** [scikit-learn](https://scikit-learn.org), [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [OpenCC](https://github.com/BYVoid/OpenCC), [langdetect](https://pypi.org/project/langdetect/), [Gradio](https://www.gradio.app), [matplotlib](https://matplotlib.org).
- **Tutorial:** Veritas AI Hackathon 2026 Project Briefs, project 4.4.1.
- **Background:**
  - [Ministry of Culture: national languages](https://www.moc.gov.tw/en/cp.aspx?n=412)
  - [Taipei Times on required local-language classes](https://taipeitimes.com/News/taiwan/archives/2021/01/18/2003750793)
  - [Hsinchu County Hakka population](https://n.yam.com/Article/20240327424880)
- **AI tools:** [Claude Code](https://claude.com/claude-code) (Anthropic, model Claude Opus 5.5) did the following:
  - found the datasets;
  - wrote the notebook and the demo app;
  - ran the experiments;
  - drafted this write-up and the video script.

  The video narration is an AI voice (Microsoft Edge text-to-speech).

---

### What's in this repository
| File | What it is |
|---|---|
| `mother_tongue_detector.ipynb` | The whole project, saved with its outputs. In Colab: Runtime → Run all (about 5 minutes) |
| `app.py` | The demo app (run the notebook first; it saves the model) |
| `results/` | The charts and score tables from the notebook |
| `tools/build_notebook.py` | The script that writes and runs the notebook |
| `requirements.txt` | Python packages |
