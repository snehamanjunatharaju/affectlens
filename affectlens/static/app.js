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
let selectedVideoFile;
let selectedVideoUrl;
let liveFaceTracks = { tracks: new Map(), nextId: 1 };
let liveProbabilityHistories = new Map();
let faceLandmarkerPromise;
let serviceLabel = 'CHECKING SERVICE';
let facePipelineLabel = 'LOADING MEDIAPIPE';

function isHeifFile(file) {
  const mimeType = (file.type || '').toLowerCase();
  return ['image/heic', 'image/heif'].includes(mimeType) || /\.(heic|heif)$/i.test(file.name || '');
}

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
    ? (data.fusion_loaded ? 'MESH FUSION · CONFORMAL' : data.uncertainty_mode === 'split_conformal' ? 'CONFORMAL' : 'SCORE THRESHOLD')
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
    sourceNote.textContent = 'JPEG, PNG, WebP, HEIC or HEIF · max 8 MB · sent to server for analysis · not saved by the app';
  } else if (cameraMode) {
    stopCamera();
    placeholder.hidden = Boolean(cameraStream);
    placeholderText.textContent = 'Click “Start camera” to enable your camera';
    previewVideo.hidden = !cameraStream;
    sourceNote.textContent = 'Camera starts on request · face detection runs in browser · crops sent to server · no audio or recording';
  } else {
    stopCamera();
    placeholder.hidden = Boolean(selectedVideoFile);
    placeholderText.textContent = 'Choose a short video to preview';
    videoFilePreview.hidden = !selectedVideoFile;
    sourceNote.textContent = 'MP4 or WebM · up to 30 seconds · only sampled face crops sent to server · no audio';
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
    try {
      await previewImage.decode();
    } catch (decodeError) {
      if (!isHeifFile(selectedFile)) {
        throw new Error('This image cannot be previewed by your browser. Save it as JPEG or PNG and try again.');
      }
      const heifBlob = selectedFile.type ? selectedFile : new Blob([selectedFile], { type: 'image/heic' });
      sourceNote.textContent = 'Browser cannot preview this HEIC/HEIF file; sending it to the server for face detection.';
      showPrediction(await requestAnalysis({ blob: heifBlob, faceCrop: false }));
      return;
    }
    const prepared = await prepareFaceInputs(previewImage, selectedFile);
    sourceNote.textContent = prepared[0].faceCrop
      ? `MediaPipe aligned ${prepared.length} face(s) in browser · crops sent to server · not saved by app`
      : 'MediaPipe found no face; sending image to server-side face detector · not saved by app';
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
  liveFaceTracks = { tracks: new Map(), nextId: 1 };
  liveProbabilityHistories = new Map();
  liveButton.textContent = 'Pause live analysis';
  sourceNote.textContent = 'Faces are matched by mesh position between frames · crops sent to server every 700 ms · no video is recorded';
  captureFrame();
  liveTimer = window.setInterval(captureFrame, 700);
}

function stopLiveAnalysis() {
  if (liveTimer) window.clearInterval(liveTimer);
  liveTimer = undefined;
  liveFaceTracks = { tracks: new Map(), nextId: 1 };
  liveProbabilityHistories = new Map();
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
    const rawPredictions = await Promise.all(prepared.map(async input => ({
      ...await requestAnalysis(input), faceCenter: input.faceCenter,
    })));
    const predictions = assignFaceTracks(rawPredictions, liveFaceTracks, performance.now(), 3000);
    if (predictions.length > 1) showMultiplePredictions(predictions, { live: true });
    else showPrediction(predictions[0], { live: true, faceIndex: predictions[0].trackId ?? 0 });
  } catch (error) {
    showError(error.message);
    if (/No model configured/i.test(error.message)) stopLiveAnalysis();
  } finally {
    requestInFlight = false;
  }
}

async function prepareFaceInputs(source, fallbackBlob) {
  const landmarker = await loadFaceLandmarker();
  if (!landmarker) return [{ blob: fallbackBlob, faceCrop: false, faceCenter: { x: 0.5, y: 0.5 } }];
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
    return [{ blob: fallbackBlob, faceCrop: false, faceCenter: { x: 0.5, y: 0.5 } }];
  }
  if (detections.length === 0) return [{ blob: fallbackBlob, faceCrop: false, landmarkCues: [], faceCenter: { x: 0.5, y: 0.5 } }];
  return Promise.all(detections.slice(0, 3).map(async (landmarks, index) => ({
    blob: await alignFaceCrop(source, landmarks),
    faceCrop: true,
    faceCenter: landmarkCenter(landmarks),
    landmarkCues: (blendshapes[index]?.categories || []).map(({ categoryName, score }) => ({ categoryName, score })),
  })));
}

function landmarkCenter(landmarks) {
  const xs = landmarks.map(point => point.x);
  const ys = landmarks.map(point => point.y);
  return {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
}

function assignFaceTracks(predictions, state, timestamp, maxAge = Infinity) {
  for (const [id, track] of state.tracks) {
    if (timestamp - track.lastSeen > maxAge) state.tracks.delete(id);
  }
  const available = new Set(state.tracks.keys());
  return predictions.map(prediction => {
    if (!prediction.faceCenter || !Object.keys(prediction.probabilities || {}).length) return prediction;
    let nearestId;
    let nearestDistance = 0.18;
    for (const id of available) {
      const center = state.tracks.get(id).center;
      const distance = Math.hypot(prediction.faceCenter.x - center.x, prediction.faceCenter.y - center.y);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestId = id;
      }
    }
    const id = nearestId ?? state.nextId++;
    available.delete(id);
    state.tracks.set(id, { center: prediction.faceCenter, lastSeen: timestamp });
    return { ...prediction, trackId: id };
  });
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
  if (landmarkCues.length) {
    headers['X-AffectLens-Blendshapes'] = JSON.stringify(Object.fromEntries(
      landmarkCues.map(({ categoryName, score }) => [categoryName, score]),
    ));
  }
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
    `<section class="face-result"><h3>${data.trackId ? `Track ${data.trackId}` : `Face ${index + 1}`}</h3>${predictionMarkup(data, { live, faceIndex: data.trackId ?? index })}</section>`
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
      const predictions = await Promise.all(inputs.map(async input => ({
        ...await requestAnalysis(input), faceCenter: input.faceCenter,
      })));
      frames.push({ time, predictions });
      sourceNote.textContent = `Processed frame ${index + 1} of ${frameCount} in browser · no audio analyzed`;
    }
    showVideoTimeline(frames);
    sourceNote.textContent = `Analyzed ${frames.length} sampled frames · detected face crops sent to server · video is not uploaded or saved by the app`;
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
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error('Timed out while decoding a video frame.'));
    }, 8000);
    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('loadeddata', onSeeked);
      video.removeEventListener('error', onError);
    };
    video.addEventListener('error', onError, { once: true });
    if (Math.abs(video.currentTime - target) < 0.005) {
      video.addEventListener('loadeddata', onSeeked, { once: true });
      return;
    }
    video.addEventListener('seeked', onSeeked, { once: true });
    video.currentTime = target;
  });
}

function showVideoTimeline(frames) {
  const tracker = { tracks: new Map(), nextId: 1 };
  frames.forEach(frame => {
    frame.predictions = assignFaceTracks(frame.predictions, tracker, frame.time * 1000, 6000);
  });
  const trackIds = [...new Set(frames.flatMap(frame => frame.predictions.map(prediction => prediction.trackId).filter(Boolean)))];
  if (!trackIds.length) {
    result.innerHTML = '<p class="abstain">No face with usable scores was found in the sampled frames.</p>';
    return;
  }
  result.innerHTML = trackIds.map(trackId => {
    const points = frames.map(frame => {
      const prediction = frame.predictions.find(item => item.trackId === trackId);
      const scores = prediction?.probabilities || {};
      const [label, score] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0] || ['no face', 0];
      const height = Math.max(3, score * 100);
      const status = !prediction ? 'no face detected' : prediction.abstained ? 'uncertain; no single label' : `selected ${label}`;
      return `<div class="timeline-point${prediction?.abstained ? ' abstained' : ''}" style="height:${height}%" title="${frame.time.toFixed(1)}s · ${escapeHtml(status)} · leading score ${(score * 100).toFixed(1)}%"><span>${frame.time.toFixed(0)}s</span></div>`;
    }).join('');
    const sampled = frames.map(frame => frame.predictions.find(item => item.trackId === trackId)).filter(Boolean);
    const uncertain = sampled.filter(prediction => prediction.abstained).length;
    const summary = sampled.length
      ? `${uncertain} of ${sampled.length} sampled frames were uncertain and had no single selected label.`
      : 'No face was detected in the sampled frames.';
    return `<section class="face-result"><h3>Approximate track ${trackId} · sampled frames</h3><div class="timeline-chart">${points}</div><p class="timeline-summary">${summary} Gray bars indicate abstention. Tracks are matched by face position and may split or switch when people cross or move quickly; they are not identity recognition.</p></section>`;
  }).join('');
}

function predictionMarkup(data, { live = false, faceIndex = 0 } = {}) {
  const currentScores = data.probabilities || {};
  const labels = Object.keys(currentScores);
  if (!labels.length) {
    const message = data.reason === 'no_face'
      ? 'No face detected. Center one face in the frame and try again.'
      : data.reason === 'multiple_faces'
        ? 'Several faces were detected. Each face is analyzed separately.'
        : 'No usable face scores were returned.';
    return `<p class="abstain">${message}</p>`;
  }

  const scores = currentScores;
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestLabel, bestScore] = ranked[0];
  let liveTrend = '';
  if (live) {
    const history = [...(liveProbabilityHistories.get(faceIndex) || []), currentScores].slice(-5);
    liveProbabilityHistories.set(faceIndex, history);
    if (history.length >= 3) {
      const meanScores = Object.fromEntries(labels.map(label => [
        label,
        history.reduce((sum, frame) => sum + (frame[label] || 0), 0) / history.length,
      ]));
      const [trendLabel] = Object.entries(meanScores).sort((a, b) => b[1] - a[1])[0];
      liveTrend = `<p class="microcopy">Recent frame trend: ${escapeHtml(trendLabel)} · ${history.length} frames · smoothing is descriptive and does not change this frame’s calibrated result.</p>`;
    }
  }
  const rows = ranked.map(([label, score]) =>
    `<div class="score"><span>${escapeHtml(label)}</span><div class="track"><div class="fill" style="width:${Math.round(score * 100)}%"></div></div><span>${(score * 100).toFixed(1)}%</span></div>`
  ).join('');
  const uncertainty = data.abstained
    ? `<p class="abstain">Uncertain: no single label selected for this frame. The leading score is ${escapeHtml(bestLabel)}, but it is not a reliable classification.</p>`
    : '<p class="microcopy">The model selected one class; this is still a guess, not a guarantee.</p>';
  const choices = data.prediction_set?.length
    ? `<details class="candidate-list"><summary>Conformal candidates for this frame (${data.prediction_set.length})</summary><p>${data.prediction_set.map(escapeHtml).join(', ')}</p></details>`
    : '';
  const cues = (data.landmarkCues || []).filter(item => item.score >= 0.1).sort((a, b) => b.score - a.score).slice(0, 5);
  const landmarkPanel = cues.length
    ? `<details class="candidate-list"><summary>MediaPipe blendshape features</summary><p>${cues.map(item => `${escapeHtml(item.categoryName)} ${(item.score * 100).toFixed(0)}%`).join(' · ')}</p><p class="microcopy">${data.fusion_applied ? 'These movement features were combined with image scores by the calibrated fusion model.' : 'These movement coefficients were not used in the class score because the fusion model was unavailable.'} They are not emotion labels by themselves.</p></details>`
    : '';
  const disagreement = data.modality_disagreement
    ? `<p class="abstain">Image and mesh fusion disagree: the image-only model favored ${escapeHtml(data.base_label || 'another class')}. This frame is marked uncertain.</p>`
    : '';
  const rolling = live
    ? '<p class="microcopy">Scores are for this frame only; live predictions can vary between frames. Temperature scaling can fail under shift.</p>'
    : '<p class="microcopy">Temperature-calibrated score on the training distribution; it is not a measure of inner feelings.</p>';
  const heading = data.abstained ? `Leading class (uncertain): ${bestLabel}` : `Selected class: ${bestLabel}`;
  const scoreLabel = data.abstained ? 'LEADING SCORE' : 'CALIBRATED SCORE';
  return `<div class="prediction-title"><strong>${escapeHtml(heading)}</strong><span class="confidence">${scoreLabel} ${(bestScore * 100).toFixed(1)}%</span></div>${uncertainty}${disagreement}${choices}${landmarkPanel}${rolling}${liveTrend}${rows}`;
}

function showError(message) {
  result.innerHTML = `<p class="abstain">${escapeHtml(message)}</p>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
}
