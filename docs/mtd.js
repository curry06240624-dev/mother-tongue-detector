// Mother-Tongue Detector: runs the trained model in the browser.
// It repeats scikit-learn's maths exactly: character 1-2-grams -> 1 + log(count) -> times IDF -> L2-normalise ->
// one score per language -> softmax. model.json stores each language's weights minus the average across languages,
// which gives the same probabilities and is also what the "why this answer" colours show.
(function (root) {
  "use strict";
  const HAN = /[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u{20000}-\u{3134F}]/gu;

  // Keep only the Chinese characters (the model was trained on text cleaned the same way).
  function clean(text) {
    return (text.normalize("NFKC").match(HAN) || []).join("");
  }

  function load(data) {
    const vocab = new Map();
    data.vocab.split("\u0001").forEach((gram, i) => vocab.set(gram, i));
    return Object.assign({}, data, { vocabMap: vocab });
  }

  function analyse(model, sentence) {
    const text = clean(sentence);
    const chars = Array.from(text);               // one entry per character, even outside the basic plane
    const [lo, hi] = model.ngram_range;
    const grams = [];
    const counts = new Map();
    for (let n = lo; n <= hi; n++) {
      for (let i = 0; i + n <= chars.length; i++) {
        const g = chars.slice(i, i + n).join("");
        grams.push({ i, n, g });
        counts.set(g, (counts.get(g) || 0) + 1);
      }
    }
    // TF-IDF vector (only n-grams the model knows), then L2 normalisation
    const x = new Map();
    let norm = 0;
    for (const [g, c] of counts) {
      const j = model.vocabMap.get(g);
      if (j === undefined) continue;
      const v = (1 + Math.log(c)) * model.idf[j];
      x.set(j, v);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (const [j, v] of x) x.set(j, v / norm);

    // One score per language, then softmax
    const K = model.classes.length;
    const scores = model.intercept.slice();
    for (const [j, v] of x) for (let k = 0; k < K; k++) scores[k] += v * model.weights[k][j];
    const top = Math.max(...scores);
    const exp = scores.map((s) => Math.exp(s - top));
    const total = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map((v) => v / total);
    let best = 0;
    for (let k = 1; k < K; k++) if (probs[k] > probs[best]) best = k;

    // How much each character pushed the model towards its answer
    const evidence = new Array(chars.length).fill(0);
    for (const { i, n, g } of grams) {
      const j = model.vocabMap.get(g);
      if (j === undefined) continue;
      const w = (x.get(j) * model.weights[best][j]) / counts.get(g) / n;
      for (let t = i; t < i + n; t++) evidence[t] += w;
    }

    let status = "ok";
    if (chars.length < model.min_chars) status = "too_short";
    else if (probs[best] < model.threshold) status = "not_sure";
    return { text, chars, probs, best, label: model.classes[best], confidence: probs[best], evidence, status };
  }

  const api = { clean, load, analyse };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MTD = api;
})(typeof self !== "undefined" ? self : this);
