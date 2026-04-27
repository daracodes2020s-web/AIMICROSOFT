<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MachineryLab</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #925296;
      color: #fff;
      margin: 0;
      padding: 20px;
    }

    h1 {
      margin-top: 0;
    }

    .card {
      background: rgba(0, 0, 0, 0.2);
      padding: 16px;
      border-radius: 8px;
      max-width: 720px;
    }

    .inputs {
      margin-top: 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .inputs input {
      padding: 6px 8px;
      border-radius: 4px;
      border: none;
      outline: none;
      width: 100%;
      box-sizing: border-box;
    }

    .buttons {
      margin-top: 12px;
      display: flex;
      gap: 8px;
    }

    button {
      padding: 8px 14px;
      border-radius: 4px;
      border: none;
      cursor: pointer;
      font-weight: bold;
    }

    #runBtn {
      background: #00c853;
      color: #000;
    }

    #clearBtn {
      background: #ffab00;
      color: #000;
    }

    pre {
      margin-top: 16px;
      padding: 12px;
      background: rgba(0, 0, 0, 0.35);
      border-radius: 6px;
      max-width: 720px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
  </style>
</head>
<body>
  <h1>MachineryLab</h1>
  <div class="card">
    <p>Where we experiment with training tiny neural networks (offline brain) and mix them with an internet brain.</p>
    <p>Type <strong>any number of words</strong>, then click <strong>Run AI</strong>.</p>

    <div class="inputs">
      <input
        id="sentenceInput"
        placeholder="Type your words here (any length)..."
      />
    </div>

    <div class="buttons">
      <button id="runBtn">Run AI (Offline + Online)</button>
      <button id="clearBtn">Clear Output</button>
    </div>

    <pre id="output">// Output will appear here and in the console.</pre>
  </div>

  <script>
    // MachineryTek Tiny Neural Word Brain
    // By Dara (CEO of MachineryTek) + Copilot assistant 🧠

    // Small helper to also log to the page
    const outputEl = document.getElementById("output");
    function logToPage(...args) {
      console.log(...args);
      outputEl.textContent += args.join(" ") + "\n";
    }

    // --------------------------------------------------
    // 1. Vocabulary
    // --------------------------------------------------
    const words = [
      "the", "cat", "was", "a", "dog", "machine", "data", "code", "learn", "train",
      "model", "input", "output", "vector", "value", "number", "text", "word", "node", "network",
      "compute", "system", "run", "process", "map", "predict", "classify", "analyze", "debug", "function",
      "logic", "state", "event", "server", "client", "cloud", "browser", "token", "embed", "feature",
      "activate", "weight", "bias", "layer", "neuron", "graph", "matrix", "array", "build", "create",
      "design", "develop", "test", "deploy", "update", "optimize", "cluster", "search", "index", "parse",
      "store", "load", "save", "open", "close", "engine", "message", "chat", "reply", "generate",
      "vectorize", "normalized", "scale", "batch", "epoch", "loss", "accuracy", "score", "intent", "response",
      "query", "string", "object", "value", "map", "key", "json", "api", "route", "request",
      "response", "connect", "session", "log", "trace", "error", "prompt", "tokenize", "analyze", "predictor",
      "classifier", "regression", "clustering", "dimensionality", "reduction", "visualize", "plot", "chart", "dashboard",
      "elephant", "giraffe", "zebra", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "squirrel", "did",
      "was", "and", "but", "or", "if", "then", "else", "when", "while", "for"
    ];

    // --------------------------------------------------
    // 2. Helpers
    // --------------------------------------------------

    // Random float in [min, max]
    function randomValue(min = 0.009, max = 0.99) {
      return Math.random() * (max - min) + min;
    }

    // Learning rate (fixed so training is stable)
    const learningRate = 0.05;

    // word -> numeric value between 0.01 and 0.99
    const wordValue = {};

    // Normalize a word (handle null/empty, spaces, casing)
    function normalizeWord(raw) {
      return (raw || "").trim().toLowerCase();
    }

    // Ensure a word is in the vocab (words + wordValue)
    function ensureInVocab(raw) {
      const w = normalizeWord(raw);
      if (!w) return null; // empty input

      if (!(w in wordValue)) {
        // New word: assign it a random value and push to vocab
        const v = randomValue();
        wordValue[w] = v;
        words.push(w); // make new words truly part of prediction vocab
        logToPage("Added new word to vocab:", w, "->", v);
      }
      return w;
    }

    // Get the numeric value for a word (ensures it's in vocab)
    function getWordValue(raw) {
      const w = ensureInVocab(raw);
      if (!w) return 0.01; // fallback for empty input
      return wordValue[w];
    }

    // Convert tanh output (-1 to 1) → valid index (0..words.length-1)
    function mapToIndex(a) {
      let idx = Math.floor((a + 1) / 2 * (words.length - 1));
      return Math.max(0, Math.min(words.length - 1, idx));
    }

    // Map a word's 0..1 value into approximately -1..1 (for target)
    function wordToTargetActivation(word) {
      const v = getWordValue(word); // ensures it exists, 0..1
      return v * 2 - 1; // 0..1 -> -1..1
    }

    // --------------------------------------------------
    // 3. Initialize base vocab numeric values
    // --------------------------------------------------

    // Preload values for the initial words array
    words.forEach(w => {
      const normalized = normalizeWord(w);
      if (!(normalized in wordValue)) {
        wordValue[normalized] = randomValue();
      }
    });

    // --------------------------------------------------
    // 4. Model parameters (variable length via aggregation)
    // --------------------------------------------------
    // Instead of 5 fixed weights, use a single weight + bias
    // and aggregate any number of word values (mean).
    let bigWeight = randomValue(0.13, 0.99);
    let bias      = randomValue();

    // --------------------------------------------------
    // 5. Forward pass (any number of words)
    // --------------------------------------------------

    // inputWords: array of raw words (string, any length)
    function forward(inputWords) {
      // This will auto-add any new words to vocab
      const inputValues = inputWords
        .map(getWordValue)
        .filter(v => v !== null && v !== undefined);

      if (inputValues.length === 0) {
        return { activatedResult: 0, inputValues: [] };
      }

      // Aggregate: mean of all word values
      const meanValue =
        inputValues.reduce((sum, v) => sum + v, 0) / inputValues.length;

      const weightedSum = meanValue * bigWeight + bias;
      const activatedResult = Math.tanh(weightedSum); // -1..1

      return { activatedResult, inputValues };
    }

    // --------------------------------------------------
    // 6. Training data (samples)
    // --------------------------------------------------

    const trainingSamples = [
      {
        input: ["the", "cat", "was", "a", "dog"],
        target: "machine"
      },
      {
        input: ["machine", "learn", "model", "train", "data"],
        target: "predictor"
      },
      {
        input: ["data", "analyze", "model", "predict", "output"],
        target: "score"
      },
      {
        input: ["debug", "test", "deploy", "update", "code"],
        target: "system"
      },
      {
        input: ["input", "process", "compute", "network", "node"],
        target: "graph"
      }
    ];

    // Train on ONE sample using gradient descent on tanh
    function trainOnSample(sample) {
      const { input, target } = sample;
      const targetActivation = wordToTargetActivation(target); // -1..1

      const { activatedResult, inputValues } = forward(input);
      if (inputValues.length === 0) return;

      const error = targetActivation - activatedResult; // how far off
      // derivative of tanh(x) = 1 - tanh(x)^2
      const derivative = 1 - activatedResult * activatedResult;
      const delta = error * derivative; // how much to change the sum

      // mean of inputValues for gradient
      const meanValue =
        inputValues.reduce((s, v) => s + v, 0) / inputValues.length;

      // Update parameters
      bigWeight += learningRate * delta * meanValue;
      bias      += learningRate * delta;
    }

    // Run many epochs of training
    function train(epochs = 500) {
      for (let e = 0; e < epochs; e++) {
        for (const sample of trainingSamples) {
          trainOnSample(sample);
        }
      }
      logToPage("Training complete.");
      logToPage("Weights:", JSON.stringify({
        bigWeight, bias
      }, null, 2));
    }

    // --------------------------------------------------
    // Helper to build a cleaner query for the online brain
    // --------------------------------------------------
    function buildOnlineQuery(inputWords) {
      // simple spelling fixes for demo
      const corrections = {
        thw: "the",
        eifel: "eiffel"
      };

      const stopwords = new Set(["the", "is", "a", "an", "of", "and", "or", "false", "true"]);

      const cleaned = inputWords
        .map(normalizeWord)
        .filter(Boolean)
        .map(w => corrections[w] || w)
        .filter(w => !stopwords.has(w));

      if (cleaned.length === 0) {
        return inputWords.join(" ");
      }
      return cleaned.join(" ");
    }

    // --------------------------------------------------
    // 7. Inference (user input → predicted words)
    // --------------------------------------------------

    // Now async so we can also call the internet
    async function predictNextWords(inputWords) {
      // Normalize & ensure vocab (fallback to "the" for weird empties)
      const safeWords = inputWords
        .map(w => ensureInVocab(w) || "the");

      // --- OFFLINE BRAIN (your neural net) ---
      const { activatedResult } = forward(safeWords);

      // FIRST predicted word
      let index1 = mapToIndex(activatedResult);
      let predicted1 = words[index1];

      // SECOND predicted word (slight variation)
      let activatedResult2 = Math.tanh(activatedResult + randomValue(-0.3, 0.3));
      let index2 = mapToIndex(activatedResult2);
      let predicted2 = words[index2];

      logToPage("🧠 Offline brain activated result:", activatedResult);
      logToPage("🧠 Offline predicted word 1:", predicted1);
      logToPage("🧠 Offline predicted word 2:", predicted2);
      logToPage("🧠 Offline result:", safeWords.join(" ") + " " + predicted1 + " " + predicted2);

      logToPage("Current vocab size:", words.length);
      logToPage("Last 10 vocab entries:", words.slice(-10).join(", "));

      // --- ONLINE BRAIN (internet search) ---
      await predictWordsFromInternet(safeWords);
    }

    // 7b. Online prediction using an Internet API (Datamuse)
    async function predictWordsFromInternet(inputWords) {
      const query = buildOnlineQuery(inputWords);
      logToPage("🌐 Online brain: searching for words related to:", `"${query}"`);

      const url =
        "https://api.datamuse.com/words?ml=" +
        encodeURIComponent(query) +
        "&max=10"; // top 10 related words

      try {
        const response = await fetch(url);
        const data = await response.json();

        if (!Array.isArray(data) || data.length === 0) {
          logToPage("🌐 Online brain: no related words found.");
          return;
        }

        const onlineWords = data.slice(0, 5).map(item => item.word);

        logToPage("🌐 Online brain suggestions:", onlineWords.join(", "));
        logToPage("🌐 Best guess:", onlineWords[0]);
      } catch (err) {
        logToPage("🌐 Online brain error:", err.message || err.toString());
      }
    }

    // --------------------------------------------------
    // 8. Wire up buttons & train model 🚀
    // --------------------------------------------------

    // Train once when the page loads
    train(800);

    const runBtn = document.getElementById("runBtn");
    const clearBtn = document.getElementById("clearBtn");
    const sentenceInput = document.getElementById("sentenceInput");

    runBtn.addEventListener("click", async () => {
      const text = sentenceInput.value || "";

      // Split on whitespace, normalize, drop empties
      const inputWords = text
        .split(/\s+/)
        .map(normalizeWord)
        .filter(Boolean);

      // If all fields are empty, do nothing
      if (inputWords.length === 0) {
        logToPage("Please enter at least one word before running the AI.");
        return;
      }

      logToPage("\n--- New Run ------------------------------------------------");
      logToPage("Input words:", inputWords.join(" "));

      runBtn.disabled = true;
      try {
        await predictNextWords(inputWords);
      } finally {
        runBtn.disabled = false;
      }
    });

    clearBtn.addEventListener("click", () => {
      outputEl.textContent = "// Output cleared.\n";
    });
  </script>
</body>
</html>
