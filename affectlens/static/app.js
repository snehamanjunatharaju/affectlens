import { FaceLandmarker, FilesetResolver } from './vendor/mediapipe/vision_bundle.mjs';

const fileInput = document.querySelector('#file');
const previewImage = document.querySelector('#preview-image');
const previewVideo = document.querySelector('#camera-preview');
const canvas = document.querySelector('#capture-canvas');
const placeholder = document.querySelector('#placeholder');
const placeholderText = document.querySelector('#placeholder-text');
const analyzeButton = document.querySelector('#analyze');
const uploadControls = document.querySelector('#upload-controls');
const cameraControls = document.querySelector('#camera-controls');
const startCameraButton = document.querySelector('#start-camera');
const liveButton = document.querySelector('#toggle-live');
const stopCameraButton = document.querySelector('#stop-camera');
const sourceNote = document.querySelector('#source-note');
const result = document.querySelector('#result');

let selectedFile;
let cameraStream;
let liveTimer;
let requestInFlight = false;
let modelReady = false;
let cameraRequestId = 0;
let liveProbabilityWindow = [];
const LIVE_WINDOW_SIZE = 7;
let faceLandmarkerPromise;
let serviceLabel = 'CHECKING SERVICE';
let facePipelineLabel = 'LOADING MEDIAPIPE';

function renderServiceStatus() {
  const status = document.querySelector('#service-status');
  status.textContent = `${facePipelineLabel} · ${serviceLabel}`;
  status.style.color = modelReady ? '#c9f36b' : '#ffd385';
}

function loadFaceLandmarker() {
  if (!faceLandmarkerPromise) {
    faceLandmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm');
      return FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/static/vendor/mediapipe/face_landmarker.task',
          delegate: 'CPU',
        },
        runningMode: 'IMAGE',
        numFaces: 3,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });
    })().catch(error => {
      console.warn('MediaPipe failed to load; the server-side OpenCV detector will be used.', error);
      facePipelineLabel = 'OPENCV FALLBACK';
      renderServiceStatus();
      return null;
    });
  }
  return faceLandmarkerPromise;
}

loadFaceLandmarker().then(landmarker => {
  if (landmarker) facePipelineLabel = 'MEDIAPIPE';
  renderServiceStatus();
});

fetch('/api/health').then(response => response.json()).then(data => {
  modelReady = data.model_loaded;
  serviceLabel = modelReady
    ? (data.uncertainty_mode === 'split_conformal' ? 'CONFORMAL' : 'SCORE THRESHOLD')
    : 'MODEL NOT CONFIGURED';
  renderServiceStatus();
  liveButton.disabled = !modelReady || !cameraStream;
}).catch(() => {
  serviceLabel = 'SERVICE OFFLINE';
  renderServiceStatus();
});

document.querySelector('#upload-tab').addEventListener('click', () => setSource('upload'));
document.querySelector('#camera-tab').addEventListener('click', () => setSource('camera'));

function setSource(source) {
  const uploadMode = source === 'upload';
  document.querySelector('#upload-tab').classList.toggle('active', uploadMode);
  document.querySelector('#camera-tab').classList.toggle('active', !uploadMode);
  document.querySelector('#upload-tab').setAttribute('aria-selected', String(uploadMode));
  document.querySelector('#camera-tab').setAttribute('aria-selected', String(!uploadMode));
  uploadControls.hidden = !uploadMode;
  cameraControls.hidden = uploadMode;
  previewImage.hidden = true;
  previewVideo.hidden = true;

  if (uploadMode) {
    stopCamera();
    previewImage.hidden = !selectedFile;
    placeholder.hidden = Boolean(selectedFile);
    placeholderText.textContent = 'Choose a face image to preview';
    sourceNote.textContent = 'JPEG, PNG or WebP · max 8 MB · image is not saved';
  } else {
    placeholder.hidden = Boolean(cameraStream);
    placeholderText.textContent = 'Click “Start camera” to enable your camera';
    previewVideo.hidden = !cameraStream;
    sourceNote.textContent = 'Camera starts on request · MediaPipe runs locally · no audio or recording';
  }
}

fileInput.addEventListener('change', () => {
  const nextFile = fileInput.files?.[0];
  if (!nextFile) return;
  selectedFile = nextFile;
  if (previewImage.src.startsWith('blob:')) URL.revokeObjectURL(previewImage.src);
  previewImage.src = URL.createObjectURL(selectedFile);
  previewImage.hidden = false;
  previewVideo.hidden = true;
  placeholder.hidden = true;
  analyzeButton.disabled = false;
});

analyzeButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.innerHTML = 'Analyzing…';
  try {
    await previewImage.decode();
    const prepared = await prepareFaceInput(previewImage, selectedFile);
    sourceNote.textContent = prepared.faceCrop
      ? 'MediaPipe aligned the face locally · only the crop goes to the local API · not saved'
      : 'MediaPipe found no face; trying the local OpenCV fallback · image is not saved';
    const data = await requestAnalysis(prepared);
    showPrediction(data);
  } catch (error) {
    showError(error.message);
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.innerHTML = 'Analyze image <span>↗</span>';
  }
});

startCameraButton.addEventListener('click', startCamera);
stopCameraButton.addEventListener('click', stopCamera);
liveButton.addEventListener('click', () => {
  if (liveTimer) stopLiveAnalysis();
  else startLiveAnalysis();
});

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError('Camera access is unavailable. Open this page on localhost or in a secure browser context.');
    return;
  }
  const requestId = ++cameraRequestId;
  startCameraButton.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    });
    if (requestId !== cameraRequestId || !document.querySelector('#camera-tab').classList.contains('active')) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    cameraStream = stream;
    previewVideo.srcObject = cameraStream;
    await previewVideo.play();
    previewVideo.hidden = false;
    previewImage.hidden = true;
    placeholder.hidden = true;
    stopCameraButton.disabled = false;
    liveButton.disabled = !modelReady;
    if (!modelReady) {
      sourceNote.textContent = 'Camera preview is local; add a trained model to enable live analysis';
    }
  } catch (error) {
    if (requestId !== cameraRequestId) return;
    cameraStream?.getTracks().forEach(track => track.stop());
    cameraStream = undefined;
    previewVideo.srcObject = null;
    startCameraButton.disabled = false;
    showError(error.name === 'NotAllowedError'
      ? 'Camera permission was not granted.'
      : `Could not start camera: ${error.message}`);
  }
}

function startLiveAnalysis() {
  if (!cameraStream || !modelReady || liveTimer) return;
  liveProbabilityWindow = [];
  liveButton.textContent = 'Pause live analysis';
  sourceNote.textContent = 'MediaPipe aligns locally · one face crop sent every 700 ms · no video is recorded';
  captureFrame();
  liveTimer = window.setInterval(captureFrame, 700);
}

function stopLiveAnalysis() {
  if (liveTimer) window.clearInterval(liveTimer);
  liveTimer = undefined;
  liveProbabilityWindow = [];
  liveButton.textContent = 'Start live analysis ↗';
  liveButton.disabled = !cameraStream || !modelReady;
  if (cameraStream) sourceNote.textContent = 'Camera preview is local · live analysis paused';
}

function stopCamera() {
  cameraRequestId += 1;
  stopLiveAnalysis();
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = undefined;
  previewVideo.pause();
  previewVideo.srcObject = null;
  previewVideo.hidden = true;
  startCameraButton.disabled = false;
  stopCameraButton.disabled = true;
  liveButton.disabled = true;
  if (!document.querySelector('#camera-tab').classList.contains('active')) return;
  placeholder.hidden = false;
  placeholderText.textContent = 'Click “Start camera” to enable your camera';
  sourceNote.textContent = 'Camera stopped · no frame or recording was saved';
}

async function captureFrame() {
  if (!cameraStream || requestInFlight || !previewVideo.videoWidth || !previewVideo.videoHeight) return;
  requestInFlight = true;
  canvas.width = previewVideo.videoWidth;
  canvas.height = previewVideo.videoHeight;
  const context = canvas.getContext('2d', { alpha: false });
  context.drawImage(previewVideo, 0, 0, canvas.width, canvas.height);
  try {
    const frame = await canvasToBlob(canvas, 'image/jpeg', 0.88);
    const prepared = await prepareFaceInput(canvas, frame);
    const data = await requestAnalysis(prepared);
    showPrediction(data, { live: true });
  } catch (error) {
    showError(error.message);
    if (/No model configured/i.test(error.message)) stopLiveAnalysis();
  } finally {
    requestInFlight = false;
  }
}

async function prepareFaceInput(source, fallbackBlob) {
  const landmarker = await loadFaceLandmarker();
  if (!landmarker) return { blob: fallbackBlob, faceCrop: false };
  let detections;
  try {
    detections = landmarker.detect(source).faceLandmarks;
  } catch (error) {
    console.warn('MediaPipe detection failed; trying the server-side OpenCV detector.', error);
    facePipelineLabel = 'OPENCV FALLBACK';
    faceLandmarkerPromise = Promise.resolve(null);
    renderServiceStatus();
    return { blob: fallbackBlob, faceCrop: false };
  }
  if (detections.length > 1) throw new Error('More than one face detected. Use an image with one face.');
  if (detections.length === 0) return { blob: fallbackBlob, faceCrop: false };
  return { blob: await alignFaceCrop(source, detections[0]), faceCrop: true };
}

function alignFaceCrop(source, landmarks) {
  const sourceWidth = source.videoWidth || source.naturalWidth || source.width;
  const sourceHeight = source.videoHeight || source.naturalHeight || source.height;
  if (!sourceWidth || !sourceHeight || landmarks.length <= 263) {
    throw new Error('Could not align this face. Try a clear, front-facing image.');
  }
  const points = landmarks.map(point => ({ x: point.x * sourceWidth, y: point.y * sourceHeight }));
  const xs = points.map(point => point.x);
  const ys = points.map(point => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const faceWidth = maxX - minX, faceHeight = maxY - minY;
  const side = Math.ceil(Math.max(faceWidth, faceHeight) * 1.18);
  if (side < 32) throw new Error('Face is too small. Move closer to the camera.');
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2 - faceHeight * 0.02;
  const eyeLeft = points[33], eyeRight = points[263];
  const angle = Math.atan2(eyeRight.y - eyeLeft.y, eyeRight.x - eyeLeft.x);
  const outputSize = 256;
  const aligned = document.createElement('canvas');
  aligned.width = outputSize;
  aligned.height = outputSize;
  const context = aligned.getContext('2d', { alpha: false });
  context.fillStyle = '#777';
  context.fillRect(0, 0, outputSize, outputSize);
  context.translate(outputSize / 2, outputSize / 2);
  context.scale(outputSize / side, outputSize / side);
  context.rotate(-angle);
  context.translate(-centerX, -centerY);
  context.drawImage(source, 0, 0);
  return canvasToBlob(aligned, 'image/jpeg', 0.92);
}

function canvasToBlob(source, type, quality) {
  return new Promise((resolve, reject) => {
    source.toBlob(blob => blob ? resolve(blob) : reject(new Error('Could not prepare the face crop.')), type, quality);
  });
}

async function requestAnalysis({ blob, faceCrop }) {
  const headers = { 'Content-Type': blob.type || 'image/jpeg' };
  if (faceCrop) headers['X-AffectLens-Face-Crop'] = 'mediapipe-landmarks';
  const response = await fetch('/api/analyze', { method: 'POST', headers, body: blob });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Analysis failed');
  return data;
}

function showPrediction(data, { live = false } = {}) {
  const currentScores = data.probabilities || {};
  const labels = Object.keys(currentScores);
  if (!labels.length) {
    liveProbabilityWindow = [];
    const message = data.reason === 'no_face'
      ? 'No face detected. Center one face in the frame and try again.'
      : data.reason === 'multiple_faces'
        ? 'More than one face detected. Use a frame with one face.'
        : 'No usable face scores were returned.';
    result.innerHTML = `<p class="abstain">${message}</p>`;
    return;
  }

  let scores = currentScores;
  if (live) {
    liveProbabilityWindow.push(currentScores);
    liveProbabilityWindow = liveProbabilityWindow.slice(-LIVE_WINDOW_SIZE);
    scores = Object.fromEntries(labels.map(label => [
      label,
      liveProbabilityWindow.reduce((sum, frame) => sum + (frame[label] || 0), 0) / liveProbabilityWindow.length,
    ]));
  }
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestLabel, bestScore] = ranked[0];
  const rows = ranked.map(([label, score]) =>
    `<div class="score"><span>${escapeHtml(label)}</span><div class="track"><div class="fill" style="width:${Math.round(score * 100)}%"></div></div><span>${(score * 100).toFixed(1)}%</span></div>`
  ).join('');
  const uncertainty = data.abstained
    ? `<p class="abstain">Low certainty: the model cannot narrow this frame to one class.</p>`
    : '<p class="microcopy">The model selected one class; this is still a guess, not a guarantee.</p>';
  const choices = data.prediction_set?.length
    ? `<details class="candidate-list"><summary>Conformal candidates for this frame (${data.prediction_set.length})</summary><p>${data.prediction_set.map(escapeHtml).join(', ')}</p></details>`
    : '';
  const rolling = live
    ? `<p class="microcopy">Rolling average of ${liveProbabilityWindow.length} recent camera frames. Scores are model outputs, not calibrated probabilities.</p>`
    : '<p class="microcopy">Scores are model outputs, not calibrated probabilities.</p>';
  result.innerHTML = `<div class="prediction-title"><strong>Best guess: ${escapeHtml(bestLabel)}</strong><span class="confidence">MODEL SCORE ${(bestScore * 100).toFixed(1)}%</span></div>${uncertainty}${choices}${rolling}${rows}`;
}

function showError(message) {
  result.innerHTML = `<p class="abstain">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
}
