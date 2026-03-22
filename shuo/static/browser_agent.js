const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const clearBtn = document.getElementById("clearBtn");
const socketState = document.getElementById("socketState");
const phaseState = document.getElementById("phaseState");
const micState = document.getElementById("micState");
const userLive = document.getElementById("userLive");
const assistantLive = document.getElementById("assistantLive");
const transcriptList = document.getElementById("transcriptList");

const state = {
  ws: null,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  workletNode: null,
  playbackClock: 0,
  playbackNodes: new Set(),
  browserSampleRate: 16000,
  started: false,
};

function wsUrl() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/browser`;
}

function setStatus(target, text) {
  target.textContent = text;
}

function appendBubble(speaker, text) {
  if (!text.trim()) {
    return;
  }
  const bubble = document.createElement("article");
  bubble.className = `bubble ${speaker}`;
  bubble.innerHTML = `<span class="speaker label">${speaker}</span><p>${escapeHtml(text)}</p>`;
  transcriptList.appendChild(bubble);
  transcriptList.scrollTop = transcriptList.scrollHeight;
}

function clearTranscript() {
  transcriptList.innerHTML = "";
  resetLiveCard(userLive);
  resetLiveCard(assistantLive);
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function float32ToBase64Pcm16(floatBuffer) {
  const pcm = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, floatBuffer[i]));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  const bytes = new Uint8Array(pcm.buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToInt16(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
}

function playPcmChunk(base64, sampleRate) {
  if (!state.audioContext) {
    return;
  }
  const samples = base64ToInt16(base64);
  const audioBuffer = state.audioContext.createBuffer(1, samples.length, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) {
    channel[i] = samples[i] / 0x7fff;
  }

  const source = state.audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(state.audioContext.destination);

  const now = state.audioContext.currentTime;
  state.playbackClock = Math.max(state.playbackClock, now);
  source.start(state.playbackClock);
  state.playbackClock += audioBuffer.duration;
  source.onended = () => state.playbackNodes.delete(source);
  state.playbackNodes.add(source);
}

function clearPlayback() {
  state.playbackClock = state.audioContext ? state.audioContext.currentTime : 0;
  for (const node of state.playbackNodes) {
    try {
      node.stop();
    } catch (_err) {
      // ignore nodes that already ended
    }
  }
  state.playbackNodes.clear();
}

function resetLiveCard(target) {
  target.textContent = "...";
  target.dataset.final = "false";
}

function renderLiveTranscript(target, text, final) {
  const value = text && text.trim() ? text : "...";
  target.textContent = value;
  target.dataset.final = final ? "true" : "false";
}

async function shutdownAudioPipeline() {
  clearPlayback();
  if (state.workletNode) {
    try {
      state.workletNode.port.onmessage = null;
      state.workletNode.disconnect();
    } catch (_err) {
      // ignore disconnect races
    }
  }
  if (state.sourceNode) {
    try {
      state.sourceNode.disconnect();
    } catch (_err) {
      // ignore disconnect races
    }
  }
  if (state.mediaStream) {
    for (const track of state.mediaStream.getTracks()) {
      track.stop();
    }
  }
  if (state.audioContext) {
    await state.audioContext.close();
  }

  state.audioContext = null;
  state.mediaStream = null;
  state.sourceNode = null;
  state.workletNode = null;
  setStatus(micState, "off");
}

async function ensureAudioPipeline() {
  if (state.audioContext) {
    return;
  }

  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  state.audioContext = new AudioContext();
  await state.audioContext.audioWorklet.addModule("/static/pcm-capture-processor.js");
  state.browserSampleRate = state.audioContext.sampleRate;
  state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
  state.workletNode = new AudioWorkletNode(state.audioContext, "pcm-capture-processor");
  state.workletNode.port.onmessage = (event) => {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN || !state.started) {
      return;
    }
    const audio_b64 = float32ToBase64Pcm16(event.data);
    state.ws.send(JSON.stringify({
      type: "audio",
      encoding: "pcm16",
      sample_rate: state.browserSampleRate,
      audio_b64,
    }));
  };

  const silentGain = state.audioContext.createGain();
  silentGain.gain.value = 0;
  state.sourceNode.connect(state.workletNode);
  state.workletNode.connect(silentGain).connect(state.audioContext.destination);
  setStatus(micState, "on");
}

async function connect() {
  connectBtn.disabled = true;
  setStatus(socketState, "connecting");

  const ws = new WebSocket(wsUrl());
  state.ws = ws;

  ws.onopen = () => {
    disconnectBtn.disabled = false;
    setStatus(socketState, "open");
  };

  ws.onmessage = async (event) => {
    const message = JSON.parse(event.data);

    if (message.type === "ready") {
      await ensureAudioPipeline();
      await state.audioContext.resume();
      state.started = true;
      state.playbackClock = state.audioContext.currentTime;
      ws.send(JSON.stringify({ type: "start", sample_rate: state.browserSampleRate }));
      return;
    }

    if (message.type === "state") {
      setStatus(phaseState, message.phase.toLowerCase());
      return;
    }

    if (message.type === "audio") {
      playPcmChunk(message.audio_b64, message.sample_rate);
      return;
    }

    if (message.type === "clear") {
      clearPlayback();
      return;
    }

    if (message.type === "transcript") {
      const liveTarget = message.speaker === "user" ? userLive : assistantLive;
      renderLiveTranscript(liveTarget, message.text, message.final);
      if (message.final) {
        appendBubble(message.speaker, message.text);
      }
      return;
    }

    if (message.type === "error") {
      setStatus(socketState, `error: ${message.message}`);
    }
  };

  ws.onclose = () => {
    disconnectBtn.disabled = true;
    connectBtn.disabled = false;
    state.started = false;
    setStatus(socketState, "closed");
    setStatus(phaseState, "offline");
    resetLiveCard(userLive);
    resetLiveCard(assistantLive);
    shutdownAudioPipeline().catch((error) => console.error(error));
  };

  ws.onerror = () => {
    setStatus(socketState, "error");
  };
}

async function disconnect() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "stop" }));
    state.ws.close();
  }
  state.ws = null;
  state.started = false;
  await shutdownAudioPipeline();
}

connectBtn.addEventListener("click", () => {
  connect().catch((error) => {
    console.error(error);
    setStatus(socketState, `error: ${error.message}`);
    connectBtn.disabled = false;
  });
});

disconnectBtn.addEventListener("click", () => {
  disconnect().catch((error) => {
    console.error(error);
  });
});

clearBtn.addEventListener("click", clearTranscript);
