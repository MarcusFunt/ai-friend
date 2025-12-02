const recordButton = document.getElementById("recordButton");
const statusText = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const indicator = document.getElementById("indicator");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let permissionPromise;

async function ensureRecorder() {
  if (mediaRecorder) return mediaRecorder;
  if (!permissionPromise) {
    permissionPromise = navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
        mediaRecorder.onstop = handleStop;
        return mediaRecorder;
      })
      .catch((error) => {
        setStatus(`Microphone permission denied: ${error.message}`, "idle");
        throw error;
      });
  }
  return permissionPromise;
}

function setStatus(message, mode) {
  statusText.textContent = message;
  indicator.className = `indicator ${mode}`;
}

function bindPressToTalk() {
  document.addEventListener("keydown", async (event) => {
    if (event.code !== "Space" || event.repeat) return;
    event.preventDefault();
    if (isRecording) return;
    await startRecording();
  });

  document.addEventListener("keyup", (event) => {
    if (event.code !== "Space") return;
    event.preventDefault();
    if (isRecording) stopRecording();
  });

  recordButton.addEventListener("mousedown", startRecording);
  document.addEventListener("mouseup", () => {
    if (isRecording) stopRecording();
  });
}

async function startRecording() {
  try {
    const recorder = await ensureRecorder();
    audioChunks = [];
    recorder.start();
    isRecording = true;
    setStatus("Recording… release to transcribe", "recording");
  } catch (error) {
    console.error(error);
  }
}

function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") return;
  mediaRecorder.stop();
  isRecording = false;
  setStatus("Processing audio…", "processing");
}

async function handleStop() {
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("file", blob, "speech.webm");

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (result.error) {
      transcriptEl.textContent = `Error: ${result.error}`;
      setStatus("Idle – hold space to start recording.", "idle");
      return;
    }

    transcriptEl.textContent = result.text || "(no text returned)";
    setStatus("Idle – hold space to start recording.", "idle");
  } catch (error) {
    transcriptEl.textContent = `Network error: ${error.message}`;
    setStatus("Idle – hold space to start recording.", "idle");
  }
}

bindPressToTalk();
