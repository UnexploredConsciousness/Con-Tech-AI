/**
 * Guardian AI — Frontend Application
 * Handles tab switching, file uploads, API calls, and result rendering.
 */

const API_BASE = "http://127.0.0.1:5000/api"; // ✅ FIXED

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
    activeTab: "audio",
    files: { audio: null, image: null, video: null },
    loading: false,
};

// ─── Example Messages ─────────────────────────────────────────────────────────
const EXAMPLES = {
    scam: `URGENT: Your bank account has been SUSPENDED due to suspicious activity! Verify your identity immediately or your account will be permanently closed. Click here NOW: http://bit.ly/secure-verify-2024\n\nDo NOT ignore this message. You have 24 HOURS to act.`,
    phishing: `Dear Customer,\n\nWe detected unauthorized access to your PayPal account from an unrecognized device.\n\nTo secure your account, please verify your information immediately:\nhttp://paypa1-secure-login.tk/verify\n\nFailure to verify within 48 hours will result in permanent account suspension.\n\n— PayPal Security Team`,
    legit: `Hi! Just a reminder that your dentist appointment is scheduled for tomorrow, March 5th at 2:30 PM with Dr. Smith. Please arrive 10 minutes early. Reply YES to confirm or NO to reschedule. Call us at (555) 123-4567 if you have any questions.`,
};

// ─── DOM Ready ────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initNav();
    initTabs();
    initFileUploads();
    initTextInput();
    initExampleButtons();
    initFeatureCards();
    initResults();
    checkApiHealth();
});

// ─── Navigation ───────────────────────────────────────────────────────────────
function initNav() {
    const navbar = document.getElementById("navbar");
    const hamburger = document.getElementById("hamburger");

    window.addEventListener("scroll", () => {
        navbar?.classList.toggle("scrolled", window.scrollY > 20);
    });

    hamburger?.addEventListener("click", () => {
        navbar?.classList.toggle("nav-open");
    });
}

// ─── Tabs ─────────────────────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });
}

function switchTab(tabName) {
    state.activeTab = tabName;

    document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.tab === tabName);
    });

    document.querySelectorAll(".tab-panel").forEach((p) => {
        p.classList.toggle("active", p.id === `panel-${tabName}`);
    });

    hideResults();
}

// ─── Feature Cards ────────────────────────────────────────────────────────────
function initFeatureCards() {
    document.querySelectorAll(".feature-card").forEach((card) => {
        card.addEventListener("click", () => {
            const tab = card.dataset.tab;
            if (tab) {
                switchTab(tab);
                document.getElementById("analyzer")
                    ?.scrollIntoView({ behavior: "smooth" });
            }
        });
    });
}

// ─── File Uploads ─────────────────────────────────────────────────────────────
function initFileUploads() {
    ["audio", "image", "video"].forEach((type) => {
        const dropZone = document.getElementById(`drop-${type}`);
        const fileInput = document.getElementById(`input-${type}`);
        const analyzeBtn = document.getElementById(`btn-${type}`);

        if (!dropZone || !fileInput) return;

        dropZone.addEventListener("click", (e) => {
            if (e.target.tagName !== "BUTTON") fileInput.click();
        });

        fileInput.addEventListener("change", () =>
            handleFileSelect(type, fileInput.files[0])
        );

        ["dragover", "dragenter"].forEach((evt) => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropZone.classList.add("drag-over");
            });
        });

        ["dragleave", "dragend"].forEach((evt) => {
            dropZone.addEventListener(evt, () =>
                dropZone.classList.remove("drag-over")
            );
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            const file = e.dataTransfer?.files?.[0];
            if (file) handleFileSelect(type, file);
        });

        analyzeBtn?.addEventListener("click", () => runAnalysis(type));
    });
}

function handleFileSelect(type, file) {
    if (!file) return;
    state.files[type] = file;

    const infoEl = document.getElementById(`file-info-${type}`);
    if (infoEl) {
        infoEl.textContent = `📎 ${file.name} (${formatBytes(file.size)})`;
        infoEl.classList.remove("hidden");
    }

    if (type === "image") {
        const preview = document.getElementById("img-preview");
        const wrap = document.getElementById("img-preview-wrap");
        if (preview && wrap) {
            preview.src = URL.createObjectURL(file);
            wrap.classList.remove("hidden");
        }
    }

    const btn = document.getElementById(`btn-${type}`);
    if (btn) btn.disabled = false;

    hideResults();
}

// ─── Text Input ───────────────────────────────────────────────────────────────
function initTextInput() {
    const textarea = document.getElementById("text-input");
    const charCount = document.getElementById("char-count");
    const analyzeBtn = document.getElementById("btn-text");
    const clearBtn = document.getElementById("btn-clear-text");

    textarea?.addEventListener("input", () => {
        const len = textarea.value.length;
        charCount.textContent = `${len.toLocaleString()} / 10,000 characters`;
        charCount.style.color = len > 9000 ? "#ff4444" : "";
        analyzeBtn.disabled = len === 0;
        hideResults();
    });

    clearBtn?.addEventListener("click", () => {
        textarea.value = "";
        charCount.textContent = "0 / 10,000 characters";
        analyzeBtn.disabled = true;
        hideResults();
        textarea.focus();
    });

    analyzeBtn?.addEventListener("click", () => runAnalysis("text"));
}

// ─── Example Buttons ──────────────────────────────────────────────────────────
function initExampleButtons() {
    document.querySelectorAll(".example-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.dataset.example;
            const textarea = document.getElementById("text-input");

            if (textarea && EXAMPLES[key]) {
                textarea.value = EXAMPLES[key];

                // Trigger input event so counter + button state updates
                textarea.dispatchEvent(new Event("input", { bubbles: true }));

                textarea.focus();
            }
        });
    });
}

// ─── Analysis Dispatcher ─────────────────────────────────────────────────────
async function runAnalysis(type) {
    if (state.loading) return;

    if (type !== "text" && !state.files[type]) {
        showError("Please select a file first.");
        return;
    }

    if (type === "text") {
        const text = document.getElementById("text-input")?.value?.trim();
        if (!text) {
            showError("Please enter some text to analyze.");
            return;
        }
    }

    setLoading(type, true);
    hideResults();

    try {
        let result;
        if (type === "text") {
            result = await analyzeText();
        } else {
            result = await analyzeFile(type);
        }
        renderResults(type, result);
    } catch (err) {
        showError(
            err.message || "Analysis failed. Make sure the backend is running."
        );
    } finally {
        setLoading(type, false);
    }
}

async function analyzeFile(type) {
    const file = state.files[type];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetchWithTimeout(`${API_BASE}/analyze/${type}`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Server error: ${res.status}`);
    }
    return res.json();
}

async function analyzeText() {
    const text = document.getElementById("text-input").value.trim();

    const res = await fetchWithTimeout(`${API_BASE}/analyze/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Server error: ${res.status}`);
    }
    return res.json();
}

// ─── Health Check ─────────────────────────────────────────────────────────────
async function checkApiHealth() {
    try {
        const res = await fetchWithTimeout(`${API_BASE}/health`, {}, 3000);
        if (res.ok) {
            console.log("✅ Guardian AI API connected.");
        }
    } catch {
        console.warn(
            "⚠️ Cannot reach Guardian AI API. Is the backend running?"
        );
    }
}

// ─── Results Rendering ────────────────────────────────────────────────────────
function renderResults(type, response) {
    const resultData = response.result || response;
    const container = document.getElementById("results-content");
    const section = document.getElementById("results");

    if (!container || !section) return;

    const threatLevel = resultData.threat_level || resultData.classification || "UNKNOWN";
    const score = resultData.threat_score ?? resultData.ai_probability ?? 0;
    const summary = resultData.summary || "";
    const recommendations = resultData.recommendations || [];

    const levelClass = threatLevelClass(threatLevel);

    let html = `
    <div class="result-card ${levelClass}">
      <div class="result-header">
        <div class="result-badge ${levelClass}">${threatLevelIcon(threatLevel)} ${threatLevel}</div>
        <div class="result-score">
          <div class="score-circle ${levelClass}">
            <svg viewBox="0 0 36 36" class="score-svg">
              <path class="score-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
              <path class="score-fill" stroke-dasharray="${score}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
            </svg>
            <div class="score-text">
              <span class="score-num">${Math.round(score)}</span>
              <span class="score-unit">/ 100</span>
            </div>
          </div>
        </div>
      </div>

      <p class="result-summary">${escHtml(summary)}</p>

      ${renderStageScores(resultData)}
      ${renderExtraInfo(type, resultData)}
      ${renderRecommendations(recommendations)}
    </div>
  `;

    container.innerHTML = html;
    section.classList.remove("hidden");
    section.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderStageScores(data) {
    const stages = data.stage_scores || data.scores;
    if (!stages || Object.keys(stages).length === 0) return "";

    const labels = {
        keyword: "Keyword Detection",
        behavioral: "Behavioral Patterns",
        ml_model: "ML Model",
        ml: "ML Model",
        structural: "Structural Patterns",
        url: "URL Analysis",
        linguistic: "Linguistic Analysis",
        metadata: "Metadata Forensics",
        noise: "Noise Analysis",
        face: "Face Artifacts",
        compression: "Compression Analysis",
        deep_learning: "Deep Learning",
        temporal: "Temporal Consistency",
        face_tracking: "Face Tracking",
        frame_analysis: "Frame Analysis",
        audio_sync: "Audio Sync",
    };

    const bars = Object.entries(stages).map(([key, val]) => {
        const label = labels[key] || key;
        const pct = Math.round(Math.min(100, Math.max(0, val)));
        const barClass = pct >= 70 ? "bar-critical" : pct >= 50 ? "bar-high" : pct >= 30 ? "bar-medium" : "bar-low";
        return `
      <div class="score-bar-row">
        <span class="score-bar-label">${label}</span>
        <div class="score-bar-track">
          <div class="score-bar-fill ${barClass}" style="width:${pct}%"></div>
        </div>
        <span class="score-bar-val">${pct}</span>
      </div>`;
    }).join("");

    return `<div class="stage-scores"><h4>Stage Breakdown</h4>${bars}</div>`;
}

function renderExtraInfo(type, data) {
    let items = [];

    if (type === "audio") {
        if (data.transcription) items.push(["Transcription", `"${data.transcription.substring(0, 200)}${data.transcription.length > 200 ? "…" : ""}"`]);
        if (data.detected_keywords?.length) items.push(["Keywords Found", data.detected_keywords.slice(0, 5).map(k => k.keyword).join(", ")]);
        if (data.behavioral_patterns?.length) items.push(["Patterns Found", data.behavioral_patterns.map(p => p.pattern).join(", ")]);
    } else if (type === "text") {
        if (data.scam_type) items.push(["Scam Type", data.scam_type]);
        if (data.word_count) items.push(["Word Count", data.word_count]);
        const urls = data.analysis_details?.url?.urls_found;
        if (urls?.length) items.push(["URLs Detected", urls.length]);
        const kwFlags = data.analysis_details?.keyword?.flags;
        if (kwFlags?.length) items.push(["Scam Keywords", kwFlags.slice(0, 4).map(f => f.keyword).join(", ")]);
    } else if (type === "image") {
        if (data.classification) items.push(["Classification", data.classification]);
        if (data.confidence) items.push(["Confidence", `${data.confidence}%`]);
    } else if (type === "video") {
        if (data.classification) items.push(["Classification", data.classification]);
        if (data.frames_analyzed) items.push(["Frames Analyzed", data.frames_analyzed]);
        if (data.video_metadata?.duration_seconds) items.push(["Duration", `${data.video_metadata.duration_seconds}s`]);
    }

    if (!items.length) return "";

    return `
    <div class="extra-info">
      ${items.map(([k, v]) => `<div class="extra-row"><span class="extra-key">${k}:</span><span class="extra-val">${escHtml(String(v))}</span></div>`).join("")}
    </div>`;
}

function renderRecommendations(recs) {
    if (!recs?.length) return "";
    return `
    <div class="recommendations">
      <h4>Recommendations</h4>
      <ul>${recs.map(r => `<li>${escHtml(r)}</li>`).join("")}</ul>
    </div>`;
}

// ─── Results Control ──────────────────────────────────────────────────────────
function initResults() {
    document.getElementById("btn-close-results")?.addEventListener("click", hideResults);
}

function hideResults() {
    document.getElementById("results")?.classList.add("hidden");
}

function showError(msg) {
    const container = document.getElementById("results-content");
    const section = document.getElementById("results");
    if (container && section) {
        container.innerHTML = `<div class="result-error"><span>⚠️</span> ${escHtml(msg)}</div>`;
        section.classList.remove("hidden");
    }
}

// ─── Loading State ────────────────────────────────────────────────────────────
function setLoading(type, isLoading) {
    state.loading = isLoading;
    const btn = document.getElementById(`btn-${type}`);
    if (!btn) return;

    const text = btn.querySelector(".btn-text");
    const spinner = btn.querySelector(".btn-spinner");

    btn.disabled = isLoading;
    text?.classList.toggle("hidden", isLoading);
    spinner?.classList.toggle("hidden", !isLoading);
}

// ─── API Health Check ─────────────────────────────────────────────────────────
async function checkApiHealth() {
    try {
        const res = await fetchWithTimeout(`${API_BASE}/health`, {}, 3000);
        if (res.ok) {
            console.log("✅ Guardian AI API connected.");
        }
    } catch {
        console.warn("⚠️ Cannot reach Guardian AI API. Is the backend running?");
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function threatLevelClass(level) {
    const map = {
        CRITICAL: "level-critical", HIGH: "level-high", MEDIUM: "level-medium", LOW: "level-low",
        AI_GENERATED: "level-critical", DEEPFAKE: "level-critical", SUSPICIOUS: "level-high",
        GENUINE: "level-low", ERROR: "level-medium"
    };
    return map[level] || "level-low";
}

function threatLevelIcon(level) {
    const map = {
        CRITICAL: "🚨", HIGH: "⚠️", MEDIUM: "⚡", LOW: "✅",
        AI_GENERATED: "🤖", DEEPFAKE: "🎭", SUSPICIOUS: "🔍", GENUINE: "✅", ERROR: "❌"
    };
    return map[level] || "🔍";
}

function escHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

async function fetchWithTimeout(url, options = {}, timeout = 30000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    try {
        const res = await fetch(url, { ...options, signal: controller.signal });
        return res;
    } finally {
        clearTimeout(id);
    }
}