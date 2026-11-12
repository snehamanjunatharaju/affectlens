import { FaceLandmarker, FilesetResolver } from './vendor/mediapipe/vision_bundle.mjs';

const fileInput = document.querySelector('#file');
const previewImage = document.querySelector('#preview-image');
const previewVideo = document.querySelector('#camera-preview');
const videoFilePreview = document.querySelector('#video-file-preview');
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
const videoInput = document.querySelector('#video-file');
const analyzeVideoButton = document.querySelector('#analyze-video');
const result = document.querySelector('#result');

let selectedFile;
let cameraStream;
let liveTimer;
let requestInFlight = false;
let modelReady = false;
let cameraRequestId = 0;
let liveProbabilityWindows = new Map();
let selectedVideoFile;
let selectedVideoUrl;
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
        outputFaceBlendshapes: true,
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
document.querySelector('#video-tab').addEventListener('click', () => setSource('video'));

function setSource(source) {
  const uploadMode = source === 'upload';
  const cameraMode = source === 'camera';
  const videoMode = source === 'video';
  document.querySelector('#upload-tab').classList.toggle('active', uploadMode);
  document.querySelector('#camera-tab').classList.toggle('active', cameraMode);
  document.querySelector('#video-tab').classList.toggle('active', videoMode);
  document.querySelector('#upload-tab').setAttribute('aria-selected', String(uploadMode));
  document.querySelector('#camera-tab').setAttribute('aria-selected', String(cameraMode));
  document.querySelector('#video-tab').setAttribute('aria-selected', String(videoMode));
  uploadControls.hidden = !uploadMode;
  cameraControls.hidden = !cameraMode;
  document.querySelector('#video-controls').hidden = !videoMode;
  previewImage.hidden = true;
  previewVideo.hidden = true;
  videoFilePreview.hidden = true;

  if (uploadMode) {
    stopCamera();
    previewImage.hidden = !selectedFile;
    placeholder.hidden = Boolean(selectedFile);
    placeholderText.textContent = 'Choose a face image to preview';
    sourceNote.textContent = 'JPEG, PNG or WebP · max 8 MB · image is not saved';
  } else if (cameraMode) {
    stopCamera();
    placeholder.hidden = Boolean(cameraStream);
    placeholderText.textContent = 'Click “Start camera” to enable your camera';
    previewVideo.hidden = !cameraStream;
    sourceNote.textContent = 'Camera starts on request · MediaPipe runs locally · no audio or recording';
  } else {
    stopCamera();
    placeholder.hidden = Boolean(selectedVideoFile);
    placeholderText.textContent = 'Choose a short video to preview';
    videoFilePreview.hidden = !selectedVideoFile;
    sourceNote.textContent = 'MP4 or WebM · up to 30 seconds · frames processed locally · no audio';
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
    const prepared = await prepareFaceInputs(previewImage, selectedFile);
    sourceNote.textContent = prepared[0].faceCrop
      ? `MediaPipe aligned ${prepared.length} face(s) locally · crops go to the local API · not saved`
      : 'MediaPipe found no face; trying the local OpenCV fallback · image is not saved';
    const predictions = await Promise.all(prepared.map(requestAnalysis));
    if (predictions.length > 1) showMultiplePredictions(predictions);
    else showPrediction(predictions[0]);
  } catch (error) {
    showError(error.message);
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.innerHTML = 'Analyze image <span>↗</span>';
  }
});

videoInput.addEventListener('change', () => {
  const nextFile = videoInput.files?.[0];
  if (!nextFile) return;
  if (!['video/mp4', 'video/webm'].includes(nextFile.type)) {
    showError('Choose an MP4 or WebM video.');
    videoInput.value = '';
    return;
  }
  if (nextFile.size > 25 * 1024 * 1024) {
    showError('Video exceeds the 25 MB limit.');
    videoInput.value = '';
    return;
  }
  selectedVideoFile = nextFile;
  if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
  selectedVideoUrl = URL.createObjectURL(nextFile);
  videoFilePreview.src = selectedVideoUrl;
  videoFilePreview.hidden = false;
  placeholder.hidden = true;
  analyzeVideoButton.disabled = false;
});

analyzeVideoButton.addEventListener('click', analyzeVideoTimeline);

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
  liveProbabilityWindows = new Map();
  liveButton.textContent = 'Pause live analysis';
  sourceNote.textContent = 'MediaPipe tracks up to three faces locally · face crops sent every 700 ms · no video is recorded';
  captureFrame();
  liveTimer = window.setInterval(captureFrame, 700);
}

function stopLiveAnalysis() {
  if (liveTimer) window.clearInterval(liveTimer);
  liveTimer = undefined;
  liveProbabilityWindows = new Map();
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
    const prepared = await prepareFaceInputs(canvas, frame);
    const predictions = await Promise.all(prepared.map(requestAnalysis));
    if (predictions.length > 1) showMultiplePredictions(predictions, { live: true });
    else showPrediction(predictions[0], { live: true, faceIndex: 0 });
  } catch (error) {
    showError(error.message);
    if (/No model configured/i.test(error.message)) stopLiveAnalysis();
  } finally {
    requestInFlight = false;
  }
}

async function prepareFaceInputs(source, fallbackBlob) {
  const landmarker = await loadFaceLandmarker();
  if (!landmarker) return [{ blob: fallbackBlob, faceCrop: false }];
  let detections;
  let blendshapes = [];
  try {
    const result = landmarker.detect(source);
    detections = result.faceLandmarks;
    blendshapes = result.faceBlendshapes || [];
  } catch (error) {
    console.warn('MediaPipe detection failed; trying the server-side OpenCV detector.', error);
    facePipelineLabel = 'OPENCV FALLBACK';
    faceLandmarkerPromise = Promise.resolve(null);
    renderServiceStatus();
    return [{ blob: fallbackBlob, faceCrop: false }];
  }
  if (detections.length === 0) return [{ blob: fallbackBlob, faceCrop: false, landmarkCues: [] }];
  return Promise.all(detections.slice(0, 3).map(async (landmarks, index) => ({
    blob: await alignFaceCrop(source, landmarks),
    faceCrop: true,
    landmarkCues: (blendshapes[index]?.categories || []).map(({ categoryName, score }) => ({ categoryName, score })),
  })));
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

async function requestAnalysis({ blob, faceCrop, landmarkCues = [] }) {
  const headers = { 'Content-Type': blob.type || 'image/jpeg' };
  if (faceCrop) headers['X-AffectLens-Face-Crop'] = 'mediapipe-landmarks';
  const response = await fetch('/api/analyze', { method: 'POST', headers, body: blob });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Analysis failed');
  return { ...data, landmarkCues };
}

function showPrediction(data, options = {}) {
  result.innerHTML = predictionMarkup(data, options);
}

function showMultiplePredictions(predictions, { live = false } = {}) {
  result.innerHTML = predictions.map((data, index) =>
    `<section class="face-result"><h3>Face ${index + 1}</h3>${predictionMarkup(data, { live, faceIndex: index })}</section>`
  ).join('');
}

async function analyzeVideoTimeline() {
  if (!selectedVideoFile || !modelReady) return;
  analyzeVideoButton.disabled = true;
  analyzeVideoButton.textContent = 'Analyzing…';
  try {
    const video = videoFilePreview;
    if (video.readyState < 1) await new Promise((resolve, reject) => {
      video.addEventListener('loadedmetadata', resolve, { once: true });
      video.addEventListener('error', () => reject(new Error('Could not read this video.')), { once: true });
    });
    if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error('This video has no readable duration.');
    if (video.duration > 30) throw new Error('Video is longer than the 30-second limit.');
    const frameCount = Math.min(30, Math.max(1, Math.ceil(video.duration)));
    const frames = [];
    for (let index = 0; index < frameCount; index += 1) {
      const time = frameCount === 1 ? 0 : index * video.duration / (frameCount - 1);
      await seekVideo(video, time);
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d', { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameBlob = await canvasToBlob(canvas, 'image/jpeg', 0.86);
      const inputs = await prepareFaceInputs(canvas, frameBlob);
      const predictions = await Promise.all(inputs.map(requestAnalysis));
      frames.push({ time, predictions });
      sourceNote.textContent = `Processed frame ${index + 1} of ${frameCount} locally · no audio analyzed`;
    }
    showVideoTimeline(frames);
    sourceNote.textContent = `Analyzed ${frames.length} sampled frames · only detected face crops went to the local API · video is not saved`;
  } catch (error) {
    showError(error.message);
  } finally {
    analyzeVideoButton.disabled = !selectedVideoFile;
    analyzeVideoButton.innerHTML = 'Analyze timeline <span>↗</span>';
  }
}

function seekVideo(video, time) {
  return new Promise((resolve, reject) => {
    const target = Math.min(Math.max(0, time), Math.max(0, video.duration - 0.01));
    if (video.readyState >= 2 && Math.abs(video.currentTime - target) < 0.005) return resolve();
    const onSeeked = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error('Could not decode a video frame.')); };
    const cleanup = () => {
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
    };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.currentTime = target;
  });
}

function showVideoTimeline(frames) {
  const faceCount = Math.max(1, ...frames.map(frame => frame.predictions.length));
  result.innerHTML = Array.from({ length: faceCount }, (_, faceIndex) => {
    const points = frames.map(frame => {
      const prediction = frame.predictions[faceIndex];
      const scores = prediction?.probabilities || {};
      const [label, score] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0] || ['no face', 0];
      const height = Math.max(3, score * 100);
      return `<div class="timeline-point${prediction?.abstained ? ' abstained' : ''}" style="height:${height}%" title="${frame.time.toFixed(1)}s · ${escapeHtml(label)} · ${(score * 100).toFixed(1)}%"><span>${frame.time.toFixed(0)}s</span></div>`;
    }).join('');
    const guesses = frames.map(frame => {
      const scores = frame.predictions[faceIndex]?.probabilities || {};
      return Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0];
    }).filter(Boolean);
    const summary = guesses.length ? `Best guesses in detector order: ${guesses.map(escapeHtml).join(' → ')}.` : 'No face was detected in the sampled frames.';
    return `<section class="face-result"><h3>Face ${faceIndex + 1} · sampled frames</h3><div class="timeline-chart">${points}</div><p class="timeline-summary">${summary} Gray bars mean the model abstained. Detector order can change between frames.</p></section>`;
  }).join('');
}

function predictionMarkup(data, { live = false, faceIndex = 0 } = {}) {
  const currentScores = data.probabilities || {};
  const labels = Object.keys(currentScores);
  if (!labels.length) {
    liveProbabilityWindows.delete(faceIndex);
    const message = data.reason === 'no_face'
      ? 'No face detected. Center one face in the frame and try again.'
      : data.reason === 'multiple_faces'
        ? 'Several faces were detected. Each face is analyzed separately.'
        : 'No usable face scores were returned.';
    return `<p class="abstain">${message}</p>`;
  }

  let scores = currentScores;
  if (live) {
    const window = [...(liveProbabilityWindows.get(faceIndex) || []), currentScores].slice(-LIVE_WINDOW_SIZE);
    liveProbabilityWindows.set(faceIndex, window);
    scores = Object.fromEntries(labels.map(label => [
      label,
      window.reduce((sum, frame) => sum + (frame[label] || 0), 0) / window.length,
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
  const cues = (data.landmarkCues || []).filter(item => item.score >= 0.1).sort((a, b) => b.score - a.score).slice(0, 5);
  const landmarkPanel = cues.length
    ? `<details class="candidate-list"><summary>MediaPipe shape cues · separate geometric signal</summary><p>${cues.map(item => `${escapeHtml(item.categoryName)} ${(item.score * 100).toFixed(0)}%`).join(' · ')}</p><p class="microcopy">Blendshape coefficients are not emotion labels and are not combined with the classifier score.</p></details>`
    : '';
  const rolling = live
    ? `<p class="microcopy">Rolling average of ${(liveProbabilityWindows.get(faceIndex) || []).length} recent frames. Temperature scaling can fail under shift.</p>`
    : '<p class="microcopy">Temperature-calibrated score on the training distribution; it is not a measure of inner feelings.</p>';
  return `<div class="prediction-title"><strong>Best guess: ${escapeHtml(bestLabel)}</strong><span class="confidence">CALIBRATED SCORE ${(bestScore * 100).toFixed(1)}%</span></div>${uncertainty}${choices}${landmarkPanel}${rolling}${rows}`;
}

function showError(message) {
  result.innerHTML = `<p class="abstain">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
}
