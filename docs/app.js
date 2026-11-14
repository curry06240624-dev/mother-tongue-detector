// Page logic for the Mother-Tongue Detector demo. The model itself lives in mtd.js.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const NATIVE = { "Taiwanese Hokkien": "台語", Hakka: "客語", Mandarin: "華語", Cantonese: "粵語" };
  const ORDER = ["Taiwanese Hokkien", "Hakka", "Mandarin", "Cantonese"];
  let model = null;
  let examples = [];

  const input = $("input");
  const detectButton = $("detect");

  function pct(p) {
    return Math.round(p * 100) + "%";
  }

  function setPill(text, kind) {
    const pill = $("result-pill");
    pill.textContent = text;
    pill.className = "pill " + (kind === "result" ? "pill-result" : "pill-muted");
  }

  function renderBars(result) {
    const bars = $("bars");
    bars.innerHTML = "";
    if (!result) return;
    const rows = model.classes
      .map((name, k) => ({ name, p: result.probs[k], top: k === result.best }))
      .sort((a, b) => b.p - a.p || ORDER.indexOf(a.name) - ORDER.indexOf(b.name));
    for (const row of rows) {
      const el = document.createElement("div");
      el.className = "bar-row" + (row.top && result.status === "ok" ? " top" : "");
      el.innerHTML = `<span></span><div class="track"><div class="fill"></div></div><span class="pct"></span>`;
      el.children[0].textContent = row.name;
      el.querySelector(".fill").style.width = (row.p * 100).toFixed(1) + "%";
      el.querySelector(".pct").textContent = pct(row.p);
      bars.appendChild(el);
    }
  }

  function renderWhy(result, example) {
    const why = $("why");
    if (!result || result.status === "too_short") {
      why.hidden = true;
      return;
    }
    why.hidden = false;
    const tiles = $("tiles");
    tiles.innerHTML = "";
    const strongest = Math.max(...result.evidence, 1e-9);
    result.chars.forEach((ch, i) => {
      const t = document.createElement("div");
      t.className = "tile";
      t.textContent = ch;
      const s = Math.max(result.evidence[i], 0) / strongest;      // 0..1, only evidence *for* the answer
      if (s > 0.15) {
        t.style.background = `rgba(3, 199, 90, ${(0.1 + 0.3 * s).toFixed(3)})`;
        t.style.borderColor = `rgba(3, 199, 90, ${(0.25 + 0.5 * s).toFixed(3)})`;
      }
      tiles.appendChild(t);
    });
    const ranked = result.chars
      .map((ch, i) => ({ ch, e: result.evidence[i] }))
      .filter((c) => c.e > 0)
      .sort((a, b) => b.e - a.e)
      .filter((c, i, arr) => arr.findIndex((d) => d.ch === c.ch) === i)
      .slice(0, 3)
      .map((c) => c.ch);
    $("clues").innerHTML = ranked.length
      ? `Strongest clues for ${result.label}: <b>${ranked.join("、")}</b>`
      : "No single character stands out.";

    const truth = $("truth");
    const others = $("others");
    if (example && clean(input.value) === clean(example.text)) {
      const right = example.truth === result.label && result.status === "ok";
      truth.hidden = false;
      truth.innerHTML = `Correct answer: <b>${example.truth}</b> (${right ? "the model got it right" : "the model got it wrong"}). ` +
        `Meaning: ${example.meaning}`;
      const parts = Object.entries(example.others || {}).map(([tool, said]) => `${tool}: <b>${said}</b>`);
      parts.push(`langdetect: <b>${example.langdetect}</b>`);
      others.hidden = false;
      others.innerHTML = `What other tools said about this sentence${example.others ? " (checked 14 Nov 2026)" : ""}: ` + parts.join(" · ");
    } else {
      truth.hidden = true;
      others.hidden = true;
    }
  }

  function clean(s) {
    return window.MTD.clean(s || "");
  }

  function currentExample() {
    const c = clean(input.value);
    return examples.find((e) => clean(e.text) === c) || null;
  }

  function update() {
    const value = input.value;
    $("count").textContent = `${Array.from(value).length} / 500`;
    $("clear").hidden = value.length === 0;
    if (!model) return;
    detectButton.disabled = value.trim().length === 0;
    const big = $("big");
    const bigLabel = $("big-label");
    const message = $("message");

    if (clean(value).length === 0) {
      setPill("Result", "muted");
      big.textContent = "";
      bigLabel.textContent = "";
      message.textContent = value.trim() ? "No Chinese characters found." : "Type a sentence, or try one of the examples below.";
      renderBars(null);
      renderWhy(null);
      return;
    }
    const result = window.MTD.analyse(model, value);
    if (result.status === "too_short") {
      setPill("Too short to tell", "muted");
      big.textContent = "";
      bigLabel.textContent = "";
      message.textContent = `Please type at least ${model.min_chars} Chinese characters. Short words are often the same in all four languages.`;
      renderBars(null);
      renderWhy(null);
      return;
    }
    if (result.status === "not_sure") {
      setPill("Not sure", "muted");
      message.textContent = `Best guess: ${result.label} (${pct(result.confidence)}). It may be another language or a mix, so a person should check.`;
    } else {
      setPill(`${result.label} · ${NATIVE[result.label]}`, "result");
      message.textContent = "";
    }
    big.textContent = pct(result.confidence);
    bigLabel.textContent = "confidence";
    renderBars(result);
    renderWhy(result, currentExample());
  }

  function renderExamples(list) {
    const chips = $("chips");
    for (const e of list) {
      const b = document.createElement("button");
      b.className = "chip";
      b.type = "button";
      b.textContent = e.text;
      b.addEventListener("click", () => {
        input.value = e.text;
        update();
      });
      chips.appendChild(b);
    }
  }

  let timer = null;
  input.addEventListener("input", () => {
    $("count").textContent = `${Array.from(input.value).length} / 500`;
    $("clear").hidden = input.value.length === 0;
    clearTimeout(timer);
    timer = setTimeout(update, 120);
  });
  detectButton.addEventListener("click", update);
  $("clear").addEventListener("click", () => {
    input.value = "";
    update();
    input.focus();
  });

  fetch("model.json")
    .then((r) => r.json())
    .then((data) => {
      model = window.MTD.load(data);
      examples = data.examples;
      renderExamples(examples);
      update();
    })
    .catch(() => {
      $("message").textContent = "Could not load the model. Please reload the page.";
    });
})();
