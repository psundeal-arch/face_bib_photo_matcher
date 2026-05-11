const form = document.getElementById("matchForm");
const bibInput = document.getElementById("bibInput");
const embeddingThresholdInput = document.getElementById("embeddingThresholdInput");
const fileInput = document.getElementById("fileInput");
const uploadBlock = document.getElementById("uploadBlock");
const cameraBlock = document.getElementById("cameraBlock");
const sourceRadios = document.querySelectorAll('input[name="photo_source"]');
const statusEl = document.getElementById("status");
const matchesEl = document.getElementById("matches");

const startCamBtn = document.getElementById("startCamBtn");
const captureBtn = document.getElementById("captureBtn");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");

let stream = null;
let capturedBlob = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function clearMatches() {
  matchesEl.innerHTML = "";
}

function getSelectedSource() {
  const selected = document.querySelector('input[name="photo_source"]:checked');
  return selected ? selected.value : "upload";
}

function stopCamera() {
  if (!stream) {
    return;
  }
  for (const track of stream.getTracks()) {
    track.stop();
  }
  stream = null;
  captureBtn.disabled = true;
}

function updateSourceUI() {
  const source = getSelectedSource();
  const useUpload = source === "upload";

  uploadBlock.classList.toggle("hidden", !useUpload);
  cameraBlock.classList.toggle("hidden", useUpload);

  if (useUpload) {
    stopCamera();
  } else {
    fileInput.value = "";
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderSourceUrl(url) {
  if (!url) {
    return "(none)";
  }
  const safe = escapeHtml(url);
  return `<a href="${safe}" target="_blank" rel="noopener noreferrer">${safe}</a>`;
}

async function startCamera() {
  if (stream) {
    return;
  }
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  captureBtn.disabled = false;
}

function captureFrame() {
  if (!stream) {
    return;
  }
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);

  canvas.toBlob(
    (blob) => {
      capturedBlob = blob;
      const url = URL.createObjectURL(blob);
      preview.src = url;
      preview.classList.remove("hidden");
      setStatus("Captured photo ready.");
    },
    "image/jpeg",
    0.95,
  );
}

sourceRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    updateSourceUI();
    setStatus(`Photo source: ${getSelectedSource()}.`);
  });
});

updateSourceUI();

startCamBtn.addEventListener("click", async () => {
  try {
    await startCamera();
    setStatus("Camera started.");
  } catch (err) {
    setStatus(`Camera error: ${err}`);
  }
});

captureBtn.addEventListener("click", () => {
  captureFrame();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearMatches();

  const bib = bibInput.value.trim();
  if (!bib) {
    setStatus("Please enter bib number.");
    return;
  }

  const source = getSelectedSource();
  let photoFile = null;

  if (source === "upload") {
    if (fileInput.files && fileInput.files.length > 0) {
      photoFile = fileInput.files[0];
    }
  } else if (capturedBlob) {
    photoFile = new File([capturedBlob], "captured.jpg", { type: "image/jpeg" });
    const dt = new DataTransfer();
    dt.items.add(photoFile);
    fileInput.files = dt.files;
  }

  if (!photoFile) {
    setStatus(source === "upload" ? "Please upload a photo." : "Please capture a photo from camera.");
    return;
  }

  setStatus("Submitting query and opening result page...");
  form.method = "POST";
  form.action = "results";
  form.enctype = "multipart/form-data";
  form.target = "_blank";
  form.submit();
  setStatus("Opened result in a new page.");
});
