// SPDX-License-Identifier: Apache-2.0
// Higgs Audio v3 TTS playground — vanilla JS frontend.

// ---------------------------------------------------------------------------
// Inline control tokens (mirror docs/cookbook/higgs_tts.md tables).
// ---------------------------------------------------------------------------

const TOKEN_CATEGORIES = {
  emotion: [
    ["elation", "Elation / joy"],
    ["amusement", "Amusement / playful laughter"],
    ["enthusiasm", "Enthusiasm / excitement"],
    ["determination", "Determination / firmness"],
    ["pride", "Pride / confidence"],
    ["contentment", "Calm satisfaction"],
    ["affection", "Warmth / affection"],
    ["relief", "Relief"],
    ["contemplation", "Thoughtful / reflective"],
    ["confusion", "Confused"],
    ["surprise", "Surprised"],
    ["awe", "Awe / wonder"],
    ["longing", "Longing / yearning"],
    ["arousal", "Heightened desire"],
    ["anger", "Anger"],
    ["fear", "Fear"],
    ["disgust", "Disgust"],
    ["bitterness", "Bitterness"],
    ["sadness", "Sadness"],
    ["shame", "Shame"],
    ["helplessness", "Helplessness"],
  ],
  style: [
    ["singing", "Singing"],
    ["shouting", "Shouting / projected voice"],
    ["whispering", "Whisper"],
  ],
  sfx: [
    ["cough", "Cough"],
    ["laughter", "Laughter"],
    ["crying", "Crying"],
    ["screaming", "Screaming"],
    ["burping", "Burping"],
    ["humming", "Humming"],
    ["sigh", "Sigh"],
    ["sniff", "Sniff"],
    ["sneeze", "Sneeze"],
  ],
  prosody: [
    ["speed_very_slow", "~0.65× speed"],
    ["speed_slow", "~0.85× speed"],
    ["speed_fast", "~1.2× speed"],
    ["speed_very_fast", "~1.4× speed"],
    ["pitch_low", "~−3 semitones"],
    ["pitch_high", "~+2.5 semitones"],
    ["pause", "~400–700 ms pause"],
    ["long_pause", "~700–1500 ms pause"],
    ["expressive_high", "More expressive delivery"],
    ["expressive_low", "Flatter delivery"],
  ],
};

const CATEGORY_ORDER = ["emotion", "style", "sfx", "prosody"];

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const textInput = $("#text-input");
const refAudio = $("#ref-audio");
const refAudioUrl = $("#ref-audio-url");
const refText = $("#ref-text");
const temperature = $("#temperature");
const topP = $("#top-p");
const topK = $("#top-k");
const maxNewTokens = $("#max-new-tokens");
const seed = $("#seed");

const synthButton = $("#synth-button");
const synthLabel = synthButton.querySelector(".primary-label");
const streamToggle = $("#stream-toggle");
const statusEl = $("#status");
const statusText = statusEl.querySelector(".status-text");
const finalAudio = $("#final-audio");
const liveAudio = $("#live-audio");
const speaking = $("#speaking");
const muteButton = $("#mute-button");
const refAudioName = $("#ref-audio-name");
const refAudioClear = $("#ref-audio-clear");
const themeToggle = $("#theme-toggle");
const envBadge = $("#env-badge");
const envText = envBadge.querySelector(".env-text");

const historyList = $("#history");

// --- mute toggle (persists across stream sessions) ----
let muted = false;
muteButton.addEventListener("click", () => {
  muted = !muted;
  liveAudio.muted = muted;
  muteButton.setAttribute("aria-pressed", muted ? "true" : "false");
  muteButton.title = muted ? "Unmute live playback" : "Mute live playback";
});

function setSpeaking(active) {
  speaking.classList.toggle("active", active);
  speaking.setAttribute("aria-hidden", active ? "false" : "true");
}

// --- theme toggle (light / dark, persisted) -----------
const THEME_KEY = "higgs-playground-theme";
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch {}
  themeToggle.title =
    theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
}
(function bootTheme() {
  let saved;
  try { saved = localStorage.getItem(THEME_KEY); } catch {}
  if (!saved) {
    saved = window.matchMedia &&
            window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light" : "dark";
  }
  applyTheme(saved);
})();
themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

// --- backend health check (one-shot on load) ----------
(async function checkBackend() {
  try {
    const resp = await fetch("/healthz", { cache: "no-store" });
    if (!resp.ok) throw new Error();
    const data = await resp.json();
    if (data.backend === "ok") {
      envBadge.classList.add("ok");
      envText.textContent = "backend · ready";
    } else {
      envBadge.classList.add("down");
      envText.textContent = "backend · down";
    }
  } catch {
    envBadge.classList.add("down");
    envText.textContent = "backend · n/a";
  }
})();

// --- file picker filename display ----------------------
refAudio.addEventListener("change", () => {
  const file = refAudio.files && refAudio.files[0];
  if (file) {
    refAudioName.textContent = file.name;
    refAudioName.classList.add("has-file");
  } else {
    refAudioName.textContent = "No file selected";
    refAudioName.classList.remove("has-file");
  }
});
refAudioClear.addEventListener("click", () => {
  refAudio.value = "";
  refAudio.dispatchEvent(new Event("change"));
});

// ---------------------------------------------------------------------------
// Token picker
// ---------------------------------------------------------------------------

function renderTokenTabs() {
  const tabsContainer = $("#token-tabs");
  tabsContainer.innerHTML = "";
  CATEGORY_ORDER.forEach((category, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "token-tab" + (i === 0 ? " active" : "");
    btn.dataset.category = category;
    btn.textContent = category;
    btn.addEventListener("click", () => {
      $$(".token-tab").forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
      renderTokenGrid(category);
    });
    tabsContainer.appendChild(btn);
  });
}

function renderTokenGrid(category) {
  const grid = $("#token-grid");
  grid.innerHTML = "";
  for (const [name, desc] of TOKEN_CATEGORIES[category]) {
    const literal = `<|${category}:${name}|>`;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "token-chip";
    chip.dataset.category = category;
    chip.title = `Insert ${literal}`;
    chip.innerHTML = `
      <span class="token-name">${literal}</span>
      <span class="token-desc">${desc}</span>
    `;
    chip.addEventListener("click", () => insertTokenAtCursor(literal));
    grid.appendChild(chip);
  }
}

function insertTokenAtCursor(token) {
  const start = textInput.selectionStart ?? textInput.value.length;
  const end = textInput.selectionEnd ?? textInput.value.length;
  const before = textInput.value.slice(0, start);
  const after = textInput.value.slice(end);

  // Add a leading space if there isn't whitespace already before the cursor,
  // and a trailing space so the next typed character doesn't merge into the
  // token visually. Avoid double-spacing.
  const leading = before && !/\s$/.test(before) ? " " : "";
  const trailing = after && !/^\s/.test(after) ? " " : "";
  const insert = `${leading}${token}${trailing}`;

  textInput.value = before + insert + after;
  const cursor = (before + insert).length;
  textInput.focus();
  textInput.setSelectionRange(cursor, cursor);
}

// ---------------------------------------------------------------------------
// Form helpers
// ---------------------------------------------------------------------------

function buildFormData() {
  const fd = new FormData();
  fd.append("text", textInput.value);
  if (refAudio.files && refAudio.files[0]) {
    fd.append("ref_audio", refAudio.files[0]);
  }
  fd.append("ref_audio_url", refAudioUrl.value || "");
  fd.append("ref_text", refText.value || "");
  fd.append("temperature", temperature.value || "");
  fd.append("top_p", topP.value || "");
  fd.append("top_k", topK.value || "");
  fd.append("max_new_tokens", maxNewTokens.value || "");
  fd.append("seed", seed.value || "");
  return fd;
}

function setStatus(text, kind = "") {
  statusText.textContent = text;
  statusEl.classList.remove("success", "error", "busy");
  if (kind) statusEl.classList.add(kind);
}

function lockButton(busy) {
  synthButton.disabled = busy;
  synthLabel.textContent = busy ? "Synthesizing…" : "Synthesize";
  streamToggle.disabled = busy;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

function renderHistoryEmpty() {
  historyList.innerHTML =
    '<li class="history-empty">No synthesis yet — your generated clips will appear here.</li>';
}

function appendHistory({ text, audioUrl, meta }) {
  // Drop the empty-state placeholder on first append.
  const placeholder = historyList.querySelector(".history-empty");
  if (placeholder) placeholder.remove();

  const li = document.createElement("li");
  li.className = "history-item";

  const textDiv = document.createElement("div");
  textDiv.className = "history-text";
  textDiv.textContent = text;
  li.appendChild(textDiv);

  if (audioUrl) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = audioUrl;
    li.appendChild(audio);
  }

  if (meta) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "history-meta";
    metaDiv.textContent = meta;
    li.appendChild(metaDiv);
  }

  historyList.prepend(li);
}

$("#clear-history").addEventListener("click", () => {
  renderHistoryEmpty();
});

// ---------------------------------------------------------------------------
// Synthesize — single button, streaming controlled by the toggle.
// ---------------------------------------------------------------------------

synthButton.addEventListener("click", async () => {
  if (!textInput.value.trim()) {
    setStatus("Please enter some text to synthesize.", "error");
    return;
  }
  finalAudio.removeAttribute("src");
  finalAudio.load();
  if (streamToggle.checked) {
    await runStreaming();
  } else {
    await runNonStreaming();
  }
});

async function runNonStreaming() {
  const inputText = textInput.value;
  lockButton(true);
  setStatus("Submitting request…", "busy");

  const started = performance.now();
  try {
    const resp = await fetch("/api/synthesize", {
      method: "POST",
      body: buildFormData(),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    finalAudio.src = url;
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    const sizeKb = (blob.size / 1024).toFixed(0);
    const meta = `${elapsed}s total | ${sizeKb} KB`;
    setStatus(meta, "success");
    appendHistory({ text: inputText, audioUrl: url, meta });
  } catch (exc) {
    setStatus(`Request failed: ${exc.message}`, "error");
  } finally {
    lockButton(false);
  }
}

async function runStreaming() {
  const inputText = textInput.value;
  lockButton(true);
  setStatus("Connecting to speech stream…", "busy");

  liveAudio.removeAttribute("src");
  liveAudio.muted = muted;
  liveAudio.load();

  const started = performance.now();
  let chunkCount = 0;
  let firstAudioAt = null;
  const wavChunks = [];
  let speakingActive = false;

  // The "speaking" indicator stays active whenever the hidden live <audio>
  // element is actually emitting sound (between play and ended/pause).
  const onPlay = () => {
    speakingActive = true;
    setSpeaking(true);
  };
  const onEnded = () => {
    speakingActive = false;
    setSpeaking(false);
  };
  liveAudio.addEventListener("play", onPlay);
  liveAudio.addEventListener("pause", onEnded);
  liveAudio.addEventListener("ended", onEnded);

  try {
    const resp = await fetch("/api/synthesize/stream", {
      method: "POST",
      body: buildFormData(),
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${err}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n")) !== -1) {
        const rawLine = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 1);
        const line = rawLine.replace(/\r$/, "");
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (!payload || payload === "[DONE]") continue;

        let event;
        try {
          event = JSON.parse(payload);
        } catch {
          continue;
        }
        if (event.error) {
          throw new Error(event.error);
        }
        const audio = event.audio;
        if (!audio || !audio.data) continue;

        chunkCount += 1;
        const wavBytes = base64ToUint8Array(audio.data);
        wavChunks.push(wavBytes);

        if (firstAudioAt === null) {
          firstAudioAt = (performance.now() - started) / 1000;
        }

        // Refresh the hidden live <audio> source with all chunks accumulated
        // so far, preserving the current playback position so the user hears
        // continuous audio rather than a restart on every chunk.
        const combined = combineWavChunks(wavChunks);
        const previewUrl = URL.createObjectURL(
          new Blob([combined], { type: "audio/wav" }),
        );
        const savedTime = liveAudio.currentTime;
        const wasPlaying = !liveAudio.paused;
        liveAudio.src = previewUrl;
        liveAudio.muted = muted;
        liveAudio.addEventListener(
          "loadedmetadata",
          () => {
            try {
              liveAudio.currentTime = savedTime;
            } catch {}
            if (wasPlaying || chunkCount === 1) {
              setSpeaking(true);
              liveAudio.play().catch(() => {});
            }
          },
          { once: true },
        );

        setStatus(
          `Streaming · chunk ${chunkCount} · first audio ${firstAudioAt.toFixed(2)}s`,
          "busy",
        );
      }
    }

    if (wavChunks.length === 0) {
      throw new Error("No audio was returned.");
    }

    const finalBytes = combineWavChunks(wavChunks);
    const finalBlob = new Blob([finalBytes], { type: "audio/wav" });
    const finalUrl = URL.createObjectURL(finalBlob);
    finalAudio.src = finalUrl;

    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    const ftf =
      firstAudioAt !== null ? ` | first audio ${firstAudioAt.toFixed(2)}s` : "";
    const meta = `${elapsed}s total | ${chunkCount} chunks${ftf}`;
    setStatus(meta, "success");
    appendHistory({ text: inputText, audioUrl: finalUrl, meta });
  } catch (exc) {
    setStatus(`Request failed: ${exc.message}`, "error");
  } finally {
    lockButton(false);
    liveAudio.removeEventListener("play", onPlay);
    liveAudio.removeEventListener("pause", onEnded);
    liveAudio.removeEventListener("ended", onEnded);
    // Indicator follows the hidden audio: if it's still draining, leave it
    // active; otherwise turn it off so the user sees the synthesis is done.
    if (!speakingActive || liveAudio.paused || liveAudio.ended) {
      setSpeaking(false);
    }
  }
}

// ---------------------------------------------------------------------------
// WAV utilities (browser-side)
// ---------------------------------------------------------------------------

function base64ToUint8Array(b64) {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

// Parse a WAV chunk into its header and PCM data so we can splice multiple
// streamed WAV chunks into a single re-headered WAV blob the browser can play.
function parseWav(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // RIFF header: 12 bytes
  // Walk subchunks to find "fmt " and "data".
  let offset = 12;
  let fmt = null;
  let dataOffset = null;
  let dataSize = 0;
  while (offset + 8 <= bytes.length) {
    const id = String.fromCharCode(
      bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3],
    );
    const size = view.getUint32(offset + 4, true);
    if (id === "fmt ") {
      fmt = {
        audioFormat: view.getUint16(offset + 8, true),
        channels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
        byteRate: view.getUint32(offset + 16, true),
        blockAlign: view.getUint16(offset + 20, true),
        bitsPerSample: view.getUint16(offset + 22, true),
      };
    } else if (id === "data") {
      dataOffset = offset + 8;
      dataSize = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  if (!fmt || dataOffset === null) {
    throw new Error("Invalid WAV chunk");
  }
  const pcm = bytes.subarray(dataOffset, dataOffset + dataSize);
  return { fmt, pcm };
}

function combineWavChunks(chunks) {
  let fmt = null;
  const pcms = [];
  let total = 0;
  for (const c of chunks) {
    const parsed = parseWav(c);
    if (!fmt) fmt = parsed.fmt;
    pcms.push(parsed.pcm);
    total += parsed.pcm.length;
  }
  return writeWav(fmt, pcms, total);
}

function writeWav(fmt, pcms, total) {
  const header = 44;
  const out = new Uint8Array(header + total);
  const view = new DataView(out.buffer);

  // RIFF
  out[0] = 0x52; out[1] = 0x49; out[2] = 0x46; out[3] = 0x46;
  view.setUint32(4, 36 + total, true);
  out[8] = 0x57; out[9] = 0x41; out[10] = 0x56; out[11] = 0x45;

  // fmt
  out[12] = 0x66; out[13] = 0x6d; out[14] = 0x74; out[15] = 0x20;
  view.setUint32(16, 16, true);
  view.setUint16(20, fmt.audioFormat, true);
  view.setUint16(22, fmt.channels, true);
  view.setUint32(24, fmt.sampleRate, true);
  view.setUint32(28, fmt.byteRate, true);
  view.setUint16(32, fmt.blockAlign, true);
  view.setUint16(34, fmt.bitsPerSample, true);

  // data
  out[36] = 0x64; out[37] = 0x61; out[38] = 0x74; out[39] = 0x61;
  view.setUint32(40, total, true);

  let offset = header;
  for (const pcm of pcms) {
    out.set(pcm, offset);
    offset += pcm.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

renderTokenTabs();
renderTokenGrid(CATEGORY_ORDER[0]);
