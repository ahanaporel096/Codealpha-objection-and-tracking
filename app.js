/**
 * app.js — VisionTrack AI Browser Engine
 * Handles webcam access, COCO-SSD detection, IoU object tracking,
 * HUD drawing, CSV logging, Supabase sync, and Groq AI reports.
 */

// ── COCO Class Groupings (matching detector.py) ─────────────────────────────
const PERSON_CLASSES = new Set(["person"]);
const VEHICLE_CLASSES = new Set([
  "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"
]);
const ANIMAL_CLASSES = new Set([
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]);

const CLASS_COLORS = {
  person: "rgba(57, 255, 20, 1)",      // Neon green
  car: "rgba(0, 200, 255, 1)",         // Cyan
  truck: "rgba(255, 140, 0, 1)",       // Orange
  bus: "rgba(255, 60, 0, 1)",          // Red-orange
  motorcycle: "rgba(180, 60, 255, 1)", // Purple
  bicycle: "rgba(255, 200, 0, 1)",     // Yellow
  default: "rgba(0, 240, 255, 1)"      // Cyan fallback
};

function getClassGroup(label) {
  if (PERSON_CLASSES.has(label)) return "person";
  if (VEHICLE_CLASSES.has(label)) return "vehicle";
  if (ANIMAL_CLASSES.has(label)) return "animal";
  return "other";
}

function getClassColor(label) {
  return CLASS_COLORS[label] || CLASS_COLORS.default;
}

// ── IoU (Intersection over Union) Tracker ───────────────────────────────────
class IoUTracker {
  constructor(iouThreshold = 0.35, maxLostFrames = 20) {
    this.iouThreshold = iouThreshold;
    this.maxLostFrames = maxLostFrames;
    this.nextId = 1;
    this.tracks = []; // Array of { id, bbox, label, group, color, lostFrames }
  }

  calculateIoU(b1, b2) {
    // bbox format: [x, y, width, height]
    const x1 = Math.max(b1[0], b2[0]);
    const y1 = Math.max(b1[1], b2[1]);
    const x2 = Math.min(b1[0] + b1[2], b2[0] + b2[2]);
    const y2 = Math.min(b1[1] + b1[3], b2[1] + b2[3]);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const area1 = b1[2] * b1[3];
    const area2 = b2[2] * b2[3];
    const union = area1 + area2 - intersection;

    return union > 0 ? intersection / union : 0;
  }

  update(detections) {
    const matchedTrackIndices = new Set();
    const matchedDetectionIndices = new Set();

    // Match existing tracks with detections based on highest IoU
    for (let i = 0; i < this.tracks.length; i++) {
      let bestIoU = 0;
      let bestDetIdx = -1;

      for (let j = 0; j < detections.length; j++) {
        if (matchedDetectionIndices.has(j)) continue;
        if (this.tracks[i].label !== detections[j].class) continue; // Same class

        const iou = this.calculateIoU(this.tracks[i].bbox, detections[j].bbox);
        if (iou > bestIoU && iou >= this.iouThreshold) {
          bestIoU = iou;
          bestDetIdx = j;
        }
      }

      if (bestDetIdx !== -1) {
        matchedTrackIndices.add(i);
        matchedDetectionIndices.add(bestDetIdx);

        // Update track position and reset lost frames
        this.tracks[i].bbox = detections[bestDetIdx].bbox;
        this.tracks[i].confidence = detections[bestDetIdx].score;
        this.tracks[i].lostFrames = 0;
        detections[bestDetIdx].trackId = this.tracks[i].id;
      } else {
        this.tracks[i].lostFrames += 1;
      }
    }

    // Create new tracks for unmatched detections
    for (let j = 0; j < detections.length; j++) {
      if (!matchedDetectionIndices.has(j)) {
        const id = this.nextId++;
        const group = getClassGroup(detections[j].class);
        const color = getClassColor(detections[j].class);

        this.tracks.push({
          id,
          bbox: detections[j].bbox,
          label: detections[j].class,
          group,
          color,
          confidence: detections[j].score,
          lostFrames: 0
        });

        detections[j].trackId = id;
      }
    }

    // Remove stale tracks that have been lost for too long
    this.tracks = this.tracks.filter(t => t.lostFrames < this.maxLostFrames);

    return detections;
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
  }
}

// ── Application State ────────────────────────────────────────────────────────
const state = {
  running: false,
  model: null,
  modelLoaded: false,
  stream: null,
  tracker: new IoUTracker(),
  sessionId: Math.random().toString(36).substring(2, 10),
  sourceType: "webcam", // "webcam" | "video"
  uploadedVideoUrl: null,

  // Settings
  confidence: 0.50,
  showIds: true,
  showConf: true,
  showHud: true,
  logCsv: false,
  syncSupabase: false,

  // Metrics
  total: 0,
  people: 0,
  vehicles: 0,
  animals: 0,
  framesCount: 0,
  fps: 0,

  // FPS Rolling window
  timestamps: [],

  // In-memory detection logs
  detectionLogs: [],
  lastAnnotatedCanvas: null,

  // Supabase buffer
  supabaseBuffer: [],
  lastSupabaseFlush: Date.now()
};

// ── DOM Elements ─────────────────────────────────────────────────────────────
const elements = {
  sidebar: document.getElementById("sidebar"),
  menuToggleBtn: document.getElementById("menuToggleBtn"),
  closeSidebarBtn: document.getElementById("closeSidebarBtn"),

  pillWebcam: document.getElementById("pillWebcam"),
  pillVideo: document.getElementById("pillVideo"),
  videoUploadGroup: document.getElementById("videoUploadGroup"),
  videoFileInput: document.getElementById("videoFileInput"),
  fileNameLabel: document.getElementById("fileNameLabel"),

  confSlider: document.getElementById("confSlider"),
  confValue: document.getElementById("confValue"),
  showIdsCheck: document.getElementById("showIdsCheck"),
  showConfCheck: document.getElementById("showConfCheck"),
  showHudCheck: document.getElementById("showHudCheck"),
  logCsvCheck: document.getElementById("logCsvCheck"),
  syncSupabaseCheck: document.getElementById("syncSupabaseCheck"),

  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  modelStatusIndicator: document.getElementById("modelStatusIndicator"),
  modelStatusText: document.getElementById("modelStatusText"),

  errorBanner: document.getElementById("errorBanner"),
  valObjects: document.getElementById("valObjects"),
  valPeople: document.getElementById("valPeople"),
  valVehicles: document.getElementById("valVehicles"),
  valAnimals: document.getElementById("valAnimals"),
  valFps: document.getElementById("valFps"),
  valFrames: document.getElementById("valFrames"),

  inputVideo: document.getElementById("inputVideo"),
  outputCanvas: document.getElementById("outputCanvas"),
  emptyStateOverlay: document.getElementById("emptyStateOverlay"),

  exportBar: document.getElementById("exportBar"),
  downloadCsvBtn: document.getElementById("downloadCsvBtn"),
  downloadSnapshotBtn: document.getElementById("downloadSnapshotBtn"),

  generateAiReportBtn: document.getElementById("generateAiReportBtn"),
  aiSpinner: document.getElementById("aiSpinner"),
  aiReportOutput: document.getElementById("aiReportOutput")
};

// ── Model Initialization ─────────────────────────────────────────────────────
async function loadVisionModel() {
  try {
    elements.modelStatusText.textContent = "Loading COCO-SSD Model...";
    // Load MobileNet v2 based COCO-SSD (fast, 80 COCO classes)
    state.model = await cocoSsd.load({ base: "mobilenet_v2" });
    state.modelLoaded = true;

    elements.modelStatusText.textContent = "Vision Model Ready";
    elements.modelStatusIndicator.querySelector(".status-indicator-dot").classList.add("ready");
  } catch (err) {
    console.error("Failed to load model:", err);
    elements.modelStatusText.textContent = "Model load failed (check connection)";
    showError("Could not load browser vision model. Please verify your internet connection.");
  }
}

// ── Video & Stream Setup ─────────────────────────────────────────────────────
async function startWebcam() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Camera access is not supported by your browser or requires HTTPS/localhost.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    },
    audio: false
  });

  state.stream = stream;
  elements.inputVideo.srcObject = stream;
  await elements.inputVideo.play();
}

async function startVideoFile() {
  if (!state.uploadedVideoUrl) {
    throw new Error("Please select a video file first.");
  }

  elements.inputVideo.srcObject = null;
  elements.inputVideo.src = state.uploadedVideoUrl;
  elements.inputVideo.loop = true;
  await elements.inputVideo.play();
}

function stopStream() {
  state.running = false;
  elements.startBtn.disabled = false;
  elements.stopBtn.disabled = true;

  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
  }

  if (elements.inputVideo) {
    elements.inputVideo.pause();
  }

  // Show export options if data exists
  if (state.detectionLogs.length > 0 || state.framesCount > 0) {
    elements.exportBar.style.display = "flex";
  }

  // Final flush to Supabase
  if (state.syncSupabase) {
    flushSupabaseBuffer(true);
    sendSessionSummaryToSupabase();
  }
}

// ── Detection & Inference Loop ───────────────────────────────────────────────
async function runDetectionLoop() {
  const video = elements.inputVideo;
  const canvas = elements.outputCanvas;
  const ctx = canvas.getContext("2d");

  // Canvas sizing to match video aspect ratio
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
  }

  const renderFrame = async () => {
    if (!state.running) return;

    if (video.readyState >= 2) {
      // 1. Draw raw video frame onto canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // 2. Run object detection
      const rawDetections = await state.model.detect(video, 40, state.confidence);

      // 3. Update IoU Tracking IDs
      const trackedDetections = state.tracker.update(rawDetections);

      // 4. Compute statistics
      let peopleCount = 0;
      let vehiclesCount = 0;
      let animalsCount = 0;

      const currentClassCounts = {};
      const nowIso = new Date().toISOString();

      trackedDetections.forEach(det => {
        const group = getClassGroup(det.class);
        if (group === "person") peopleCount++;
        else if (group === "vehicle") vehiclesCount++;
        else if (group === "animal") animalsCount++;

        currentClassCounts[det.class] = (currentClassCounts[det.class] || 0) + 1;

        // Draw bounding box, corner accents, and label
        drawDetectionBox(ctx, det);

        // Record detection log if enabled
        if (state.logCsv || state.syncSupabase) {
          const record = {
            timestamp: nowIso,
            session_id: state.sessionId,
            track_id: det.trackId || -1,
            label: det.class,
            confidence: Number(det.score.toFixed(4)),
            x1: Math.round(det.bbox[0]),
            y1: Math.round(det.bbox[1]),
            x2: Math.round(det.bbox[0] + det.bbox[2]),
            y2: Math.round(det.bbox[1] + det.bbox[3]),
            group_name: group
          };

          if (state.logCsv) state.detectionLogs.push(record);
          if (state.syncSupabase) bufferSupabaseRecord(record);
        }
      });

      // 5. Update FPS
      const now = performance.now();
      state.timestamps.push(now);
      if (state.timestamps.length > 30) state.timestamps.shift();

      if (state.timestamps.length >= 2) {
        const elapsed = (state.timestamps[state.timestamps.length - 1] - state.timestamps[0]) / 1000;
        state.fps = (state.timestamps.length - 1) / elapsed;
      }

      state.total = trackedDetections.length;
      state.people = peopleCount;
      state.vehicles = vehiclesCount;
      state.animals = animalsCount;
      state.framesCount++;

      // 6. Draw HUD overlay
      if (state.showHud) {
        drawHUD(ctx, canvas.width, canvas.height);
      }

      // 7. Update UI metrics
      updateMetricsDisplay();

      // Check periodic Supabase sync
      if (state.syncSupabase && (Date.now() - state.lastSupabaseFlush > 2500 || state.supabaseBuffer.length >= 40)) {
        flushSupabaseBuffer();
      }
    }

    if (state.running) {
      requestAnimationFrame(renderFrame);
    }
  };

  requestAnimationFrame(renderFrame);
}

// ── Canvas Drawing Helpers ───────────────────────────────────────────────────
function drawDetectionBox(ctx, det) {
  const [x, y, w, h] = det.bbox;
  const color = getClassColor(det.class);

  // 1. Semi-transparent fill
  ctx.fillStyle = color.replace("1)", "0.08)");
  ctx.fillRect(x, y, w, h);

  // 2. Main border
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);

  // 3. Corner accent marks
  const cl = Math.min(14, w / 4, h / 4);
  ctx.lineWidth = 3;
  ctx.beginPath();
  // Top-left
  ctx.moveTo(x, y + cl); ctx.lineTo(x, y); ctx.lineTo(x + cl, y);
  // Top-right
  ctx.moveTo(x + w - cl, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cl);
  // Bottom-left
  ctx.moveTo(x, y + h - cl); ctx.lineTo(x, y + h); ctx.lineTo(x + cl, y + h);
  // Bottom-right
  ctx.moveTo(x + w - cl, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cl);
  ctx.stroke();

  // 4. Label banner
  let labelText = det.class;
  if (state.showIds && det.trackId) labelText = `#${det.trackId} ${labelText}`;
  if (state.showConf) labelText += ` ${(det.score * 100).toFixed(0)}%`;

  ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
  const textWidth = ctx.measureText(labelText).width;
  const bannerHeight = 22;
  const bannerY = Math.max(0, y - bannerHeight);

  ctx.fillStyle = color;
  ctx.fillRect(x, bannerY, textWidth + 12, bannerHeight);

  ctx.fillStyle = "#0a0a0a";
  ctx.fillText(labelText, x + 6, bannerY + 16);
}

function drawHUD(ctx, width, height) {
  ctx.save();
  const hudWidth = 260;
  const hudHeight = 110;
  const hudX = 16;
  const hudY = 16;

  // Frosted dark background
  ctx.fillStyle = "rgba(11, 15, 25, 0.82)";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(hudX, hudY, hudWidth, hudHeight, 10);
  ctx.fill();
  ctx.stroke();

  // Title
  ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.fillText("🎯 VISION HUD  •  ACTIVE", hudX + 14, hudY + 24);

  // Status stats
  ctx.font = "12px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(`FPS: ${state.fps.toFixed(1)}  |  Total: ${state.total}`, hudX + 14, hudY + 46);
  ctx.fillText(`People: ${state.people}  |  Vehicles: ${state.vehicles}`, hudX + 14, hudY + 68);
  ctx.fillText(`Animals: ${state.animals}  |  Model: COCO-SSD`, hudX + 14, hudY + 90);

  ctx.restore();
}

function updateMetricsDisplay() {
  elements.valObjects.textContent = state.total;
  elements.valPeople.textContent = state.people;
  elements.valVehicles.textContent = state.vehicles;
  elements.valAnimals.textContent = state.animals;
  elements.valFps.textContent = state.fps.toFixed(1);
  elements.valFrames.textContent = state.framesCount;
}

// ── Supabase Cloud Sync ──────────────────────────────────────────────────────
function bufferSupabaseRecord(record) {
  state.supabaseBuffer.push(record);
}

async function flushSupabaseBuffer(force = false) {
  if (state.supabaseBuffer.length === 0) return;

  const recordsToSend = [...state.supabaseBuffer];
  state.supabaseBuffer = [];
  state.lastSupabaseFlush = Date.now();

  try {
    const res = await fetch("/api/sync_supabase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "detections", records: recordsToSend })
    });
    if (!res.ok) {
      console.warn("Supabase batch sync returned:", res.status);
    }
  } catch (err) {
    console.warn("Supabase sync request failed:", err);
  }
}

async function sendSessionSummaryToSupabase() {
  try {
    await fetch("/api/sync_supabase", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "session",
        data: {
          id: state.sessionId,
          total_objects: state.total,
          people_count: state.people,
          vehicle_count: state.vehicles,
          animal_count: state.animals,
          frames_count: state.framesCount,
          model_name: "coco-ssd"
        }
      })
    });
  } catch (err) {
    console.warn("Could not save session summary to Supabase:", err);
  }
}

// ── AI Scene Intelligence ────────────────────────────────────────────────────
async function generateAiReport() {
  elements.aiSpinner.style.display = "flex";
  elements.aiReportOutput.style.display = "none";
  elements.generateAiReportBtn.disabled = true;

  const stats = {
    total: state.total,
    person: state.people,
    vehicle: state.vehicles,
    animal: state.animals,
    frames: state.framesCount
  };

  try {
    const res = await fetch("/api/ai_summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stats, session_id: state.sessionId })
    });

    const data = await res.json();
    elements.aiSpinner.style.display = "none";
    elements.generateAiReportBtn.disabled = false;

    if (data.summary) {
      elements.aiReportOutput.textContent = data.summary;
      elements.aiReportOutput.style.display = "block";
    } else if (data.error) {
      showError(data.error);
    }
  } catch (err) {
    elements.aiSpinner.style.display = "none";
    elements.generateAiReportBtn.disabled = false;
    showError("Failed to reach AI service endpoint: " + err.message);
  }
}

// ── Export Tools (CSV & Snapshot) ────────────────────────────────────────────
function downloadCsv() {
  if (state.detectionLogs.length === 0) {
    alert("No detections recorded yet. Enable 'Log to CSV' and start detection.");
    return;
  }

  const headers = ["timestamp", "session_id", "track_id", "label", "confidence", "x1", "y1", "x2", "y2", "group"];
  const rows = state.detectionLogs.map(r => [
    r.timestamp, r.session_id, r.track_id, r.label, r.confidence, r.x1, r.y1, r.x2, r.y2, r.group_name
  ]);

  const csvContent = [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `detections_${state.sessionId}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadSnapshot() {
  const canvas = elements.outputCanvas;
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `snapshot_${state.sessionId}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, "image/png");
}

function showError(msg) {
  elements.errorBanner.textContent = msg;
  elements.errorBanner.style.display = "block";
  setTimeout(() => {
    elements.errorBanner.style.display = "none";
  }, 7000);
}

// ── Event Listeners ──────────────────────────────────────────────────────────
function attachEventListeners() {
  // Mobile drawer toggle
  elements.menuToggleBtn.addEventListener("click", () => elements.sidebar.classList.add("open"));
  elements.closeSidebarBtn.addEventListener("click", () => elements.sidebar.classList.remove("open"));

  // Input source radio pills
  elements.pillWebcam.addEventListener("click", () => {
    elements.pillWebcam.classList.add("active");
    elements.pillVideo.classList.remove("active");
    elements.videoUploadGroup.style.display = "none";
    state.sourceType = "webcam";
  });

  elements.pillVideo.addEventListener("click", () => {
    elements.pillVideo.classList.add("active");
    elements.pillWebcam.classList.remove("active");
    elements.videoUploadGroup.style.display = "flex";
    state.sourceType = "video";
  });

  // Video file upload
  elements.videoFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      elements.fileNameLabel.textContent = file.name;
      if (state.uploadedVideoUrl) URL.revokeObjectURL(state.uploadedVideoUrl);
      state.uploadedVideoUrl = URL.createObjectURL(file);
    }
  });

  // Controls & Settings
  elements.confSlider.addEventListener("input", (e) => {
    state.confidence = parseFloat(e.target.value);
    elements.confValue.textContent = state.confidence.toFixed(2);
  });

  elements.showIdsCheck.addEventListener("change", (e) => state.showIds = e.target.checked);
  elements.showConfCheck.addEventListener("change", (e) => state.showConf = e.target.checked);
  elements.showHudCheck.addEventListener("change", (e) => state.showHud = e.target.checked);
  elements.logCsvCheck.addEventListener("change", (e) => state.logCsv = e.target.checked);
  elements.syncSupabaseCheck.addEventListener("change", (e) => state.syncSupabase = e.target.checked);

  // Start button
  elements.startBtn.addEventListener("click", async () => {
    if (!state.modelLoaded) {
      showError("Please wait for the vision model to finish loading.");
      return;
    }

    try {
      elements.emptyStateOverlay.style.display = "none";
      elements.startBtn.disabled = true;
      elements.stopBtn.disabled = false;
      state.sessionId = Math.random().toString(36).substring(2, 10);
      state.tracker.reset();
      state.framesCount = 0;
      state.detectionLogs = [];

      if (state.sourceType === "webcam") {
        await startWebcam();
      } else {
        await startVideoFile();
      }

      state.running = true;
      runDetectionLoop();

      // Close mobile drawer if open
      elements.sidebar.classList.remove("open");
    } catch (err) {
      console.error(err);
      elements.emptyStateOverlay.style.display = "flex";
      elements.startBtn.disabled = false;
      elements.stopBtn.disabled = true;
      showError(err.message || "Failed to start video source.");
    }
  });

  // Stop button
  elements.stopBtn.addEventListener("click", stopStream);

  // AI Summary button
  elements.generateAiReportBtn.addEventListener("click", generateAiReport);

  // Export buttons
  elements.downloadCsvBtn.addEventListener("click", downloadCsv);
  elements.downloadSnapshotBtn.addEventListener("click", downloadSnapshot);
}

// ── Application Boot ─────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  attachEventListeners();
  loadVisionModel();
});
