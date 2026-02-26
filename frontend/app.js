/**
 * Guardian AI — Frontend Application Logic
 * ==========================================
 * Handles tab switching, drag-and-drop file upload, API communication,
 * and dynamic results rendering with animated threat gauges.
 */

// ─── Configuration ───────────────────────────────────────────
const API_BASE = window.location.origin;

// Track selected files per tab
const selectedFiles = { audio: null, image: null, video: null };

// ─── Tab Switching ───────────────────────────────────────────

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Update panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
        panel.classList.remove('active');
    });
    const panel = document.getElementById(`panel-${tabName}`);
    panel.classList.remove('hidden');
    panel.classList.add('active');
}

// ─── Drag & Drop Handling ────────────────────────────────────

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragging');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragging');
}

function handleDrop(e, type) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragging');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        setFile(type, files[0]);
    }
}

function handleFileSelect(e, type) {
    const files = e.target.files;
    if (files.length > 0) {
        setFile(type, files[0]);
    }
}

// ─── File Management ─────────────────────────────────────────

function setFile(type, file) {
    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showToast('File too large. Maximum size is 50MB.', 'error');
        return;
    }

    selectedFiles[type] = file;
    showFilePreview(type, file);
    enableAnalyzeButton(type);
}

function removeFile(type) {
    selectedFiles[type] = null;
    hideFilePreview(type);
    disableAnalyzeButton(type);
    // Reset the input
    document.getElementById(`input-${type}`).value = '';
}

function showFilePreview(type, file) {
    const dropzone = document.getElementById(`dropzone-${type}`);
    const content = document.getElementById(`dropzone-${type}-content`);
    const preview = document.getElementById(`preview-${type}`);

    content.classList.add('hidden');
    preview.classList.remove('hidden');
    dropzone.classList.add('has-file');

    const sizeStr = formatFileSize(file.size);
    const iconColor = { audio: 'from-rose-500 to-orange-500', image: 'from-cyan-500 to-blue-500', video: 'from-violet-500 to-purple-600' }[type];
    const iconSvg = {
        audio: '<path stroke-linecap="round" stroke-linejoin="round" d="M19.114 5.636a9 9 0 010 12.728M16.463 8.288a5.25 5.25 0 010 7.424M6.75 8.25l4.72-4.72a.75.75 0 011.28.53v15.88a.75.75 0 01-1.28.53l-4.72-4.72H4.51c-.88 0-1.704-.507-1.938-1.354A9.01 9.01 0 012.25 12c0-.83.112-1.633.322-2.396C2.806 8.756 3.63 8.25 4.51 8.25H6.75z"/>',
        image: '<path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z"/>',
        video: '<path stroke-linecap="round" stroke-linejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"/>'
    }[type];

    let thumbHtml = '';
    if (type === 'image') {
        const url = URL.createObjectURL(file);
        thumbHtml = `<img src="${url}" class="image-thumb" alt="Preview">`;
    }

    preview.innerHTML = `
        <div class="file-preview">
            ${thumbHtml || `
            <div class="file-preview-icon bg-gradient-to-br ${iconColor}">
                <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">${iconSvg}</svg>
            </div>`}
            <div class="file-preview-info">
                <div class="file-preview-name">${escapeHtml(file.name)}</div>
                <div class="file-preview-size">${sizeStr}</div>
            </div>
            <div class="file-preview-remove" onclick="event.stopPropagation(); removeFile('${type}')" title="Remove file">
                <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
                </svg>
            </div>
        </div>
    `;
}

function hideFilePreview(type) {
    const dropzone = document.getElementById(`dropzone-${type}`);
    const content = document.getElementById(`dropzone-${type}-content`);
    const preview = document.getElementById(`preview-${type}`);

    content.classList.remove('hidden');
    preview.classList.add('hidden');
    preview.innerHTML = '';
    dropzone.classList.remove('has-file');
}

function enableAnalyzeButton(type) {
    const btn = document.getElementById(`btn-analyze-${type}`);
    btn.disabled = false;
    btn.classList.remove('opacity-50', 'cursor-not-allowed');
}

function disableAnalyzeButton(type) {
    const btn = document.getElementById(`btn-analyze-${type}`);
    btn.disabled = true;
    btn.classList.add('opacity-50', 'cursor-not-allowed');
}

// ─── API Communication ───────────────────────────────────────

async function analyzeFile(type) {
    const file = selectedFiles[type];
    if (!file) {
        showToast('Please select a file first', 'error');
        return;
    }

    const btn = document.getElementById(`btn-analyze-${type}`);
    btn.classList.add('loading');
    btn.disabled = true;

    // Show loading state in results panel
    showLoadingResults(type);

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/api/analyze/${type}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            renderResults(type, result.data);
            showToast('Analysis complete!', 'success');
        } else {
            showErrorResults(type, result.error || 'Analysis failed');
            showToast(result.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('API Error:', error);
        showErrorResults(type, 'Could not connect to the server. Make sure the backend is running on port 5000.');
        showToast('Connection failed — is the backend running?', 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// ─── Results Rendering ───────────────────────────────────────

function showLoadingResults(type) {
    const panel = document.getElementById(`results-${type}`);
    panel.innerHTML = `
        <div class="text-center space-y-4">
            <div class="spinner w-12 h-12 mx-auto" style="border-width: 3px;"></div>
            <div>
                <p class="text-gray-300 font-medium">Analyzing ${type}...</p>
                <p class="text-gray-600 text-xs mt-1">This may take a few seconds</p>
            </div>
        </div>
    `;
}

function showErrorResults(type, error) {
    const panel = document.getElementById(`results-${type}`);
    panel.innerHTML = `
        <div class="text-center space-y-3 result-animate">
            <div class="w-16 h-16 mx-auto rounded-2xl bg-red-500/10 flex items-center justify-center">
                <svg class="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/>
                </svg>
            </div>
            <p class="text-red-300 text-sm font-medium">${escapeHtml(error)}</p>
        </div>
    `;
}

function renderResults(type, data) {
    const panel = document.getElementById(`results-${type}`);
    const score = data.threat_score || 0;
    const level = (data.threat_level || 'LOW').toLowerCase();
    const color = data.threat_color || '#22c55e';
    const circumference = 2 * Math.PI * 60;
    const offset = circumference - (score / 100) * circumference;

    let html = `<div class="result-animate w-full space-y-1" style="display: block;">`;

    // ── Threat Gauge ──
    html += `
        <div class="text-center pb-4">
            <div class="threat-gauge mx-auto">
                <svg viewBox="0 0 136 136" width="100%" height="100%">
                    <circle class="threat-gauge-bg" cx="68" cy="68" r="60"/>
                    <circle class="threat-gauge-fill" cx="68" cy="68" r="60"
                        stroke="${color}"
                        stroke-dasharray="${circumference}"
                        stroke-dashoffset="${offset}"/>
                </svg>
                <div class="threat-gauge-text">
                    <span class="threat-score-value" style="color: ${color}">${Math.round(score)}</span>
                    <span class="threat-score-label">Threat Score</span>
                </div>
            </div>
            <div class="mt-3">
                <span class="level-badge ${level}">
                    ${{ critical: '🚨', high: '⚠️', medium: '⚡', low: '✅' }[level] || '✅'} ${data.threat_level || 'LOW'}
                </span>
            </div>
            <p class="text-gray-400 text-xs mt-2 max-w-xs mx-auto">${escapeHtml(data.threat_description || '')}</p>
            ${data.processing_time ? `<div class="processing-time justify-center mt-2"><svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> ${data.processing_time}s</div>` : ''}
        </div>
    `;

    // ── Analysis Breakdown ──
    if (data.analyses && data.analyses.length > 0) {
        html += `<div class="result-section"><div class="result-section-title">Analysis Breakdown</div>`;
        data.analyses.forEach((a, i) => {
            const barColor = a.score >= 60 ? '#ef4444' : a.score >= 35 ? '#eab308' : '#22c55e';
            html += `
                <div class="analysis-bar-item">
                    <div class="analysis-bar-label">
                        <span>${escapeHtml(a.name)}</span>
                        <span>${Math.round(a.score)}%</span>
                    </div>
                    <div class="analysis-bar-track">
                        <div class="analysis-bar-fill" style="background: ${barColor}; width: 0;" data-width="${a.score}%"></div>
                    </div>
                    ${a.details ? `<p class="text-gray-600 text-xs mt-1">${escapeHtml(a.details)}</p>` : ''}
                </div>
            `;
        });
        html += `</div>`;
    }

    // ── Transcript (Audio only) ──
    if (type === 'audio' && data.transcript && data.transcript !== '[Transcription unavailable]') {
        html += `
            <div class="result-section">
                <div class="result-section-title">Transcript</div>
                <div class="transcript-box">"${escapeHtml(data.transcript)}"</div>
            </div>
        `;
    }

    // ── Detected Keywords (Audio only) ──
    if (type === 'audio' && data.analyses) {
        const keywordAnalysis = data.analyses.find(a => a.found_keywords && a.found_keywords.length > 0);
        if (keywordAnalysis) {
            html += `<div class="result-section"><div class="result-section-title">Detected Scam Keywords</div><div class="flex flex-wrap gap-1">`;
            keywordAnalysis.found_keywords.slice(0, 20).forEach(kw => {
                html += `<span class="keyword-tag">${escapeHtml(kw.keyword)} ${kw.count > 1 ? `(×${kw.count})` : ''}</span>`;
            });
            html += `</div></div>`;
        }
    }

    // ── Detected Indicators (Image/Video) ──
    if (data.detected_indicators && data.detected_indicators.length > 0) {
        html += `<div class="result-section"><div class="result-section-title">Detected Indicators</div>`;
        data.detected_indicators.slice(0, 8).forEach(ind => {
            html += `
                <div class="pattern-item">
                    <div class="flex items-center justify-between mb-1">
                        <span class="text-xs font-semibold text-gray-300">${escapeHtml(ind.type)}</span>
                        <span class="badge ${ind.severity || 'low'}">${ind.severity || 'info'}</span>
                    </div>
                    <p class="text-xs text-gray-500">${escapeHtml(ind.detail)}</p>
                </div>
            `;
        });
        html += `</div>`;
    }

    // ── Detected Patterns (Audio) ──
    if (data.detected_patterns && data.detected_patterns.length > 0) {
        html += `<div class="result-section"><div class="result-section-title">Behavioral Patterns</div>`;
        data.detected_patterns.forEach(p => {
            html += `
                <div class="pattern-item">
                    <div class="flex items-center justify-between mb-1">
                        <span class="text-xs font-semibold text-gray-300">${escapeHtml(p.name)}</span>
                        <span class="badge ${p.severity || 'medium'}">${p.severity || 'medium'}</span>
                    </div>
                    <p class="text-xs text-gray-500">${escapeHtml(p.description)}</p>
                </div>
            `;
        });
        html += `</div>`;
    }

    // ── Classification (Image/Video) ──
    if (data.classification) {
        const classLabels = {
            'AI_GENERATED': { text: 'AI-Generated', icon: '🤖', cls: 'critical' },
            'DEEPFAKE': { text: 'Deepfake Detected', icon: '🚨', cls: 'critical' },
            'SUSPICIOUS': { text: 'Suspicious', icon: '⚠️', cls: 'medium' },
            'LIKELY_GENUINE': { text: 'Likely Genuine', icon: '✅', cls: 'low' }
        };
        const cl = classLabels[data.classification] || { text: data.classification, icon: '❓', cls: 'medium' };
        html += `
            <div class="result-section text-center">
                <div class="result-section-title">Classification</div>
                <span class="level-badge ${cl.cls}" style="font-size: 0.85rem;">${cl.icon} ${cl.text}</span>
            </div>
        `;
    }

    // ── Recommendations ──
    if (data.recommendations && data.recommendations.length > 0) {
        html += `<div class="result-section"><div class="result-section-title">Recommendations</div>`;
        data.recommendations.forEach(rec => {
            html += `<div class="recommendation-item">${escapeHtml(rec)}</div>`;
        });
        html += `</div>`;
    }

    // ── Media Info ──
    if (data.image_info) {
        html += `
            <div class="result-section">
                <div class="result-section-title">Image Info</div>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div class="text-gray-500">Resolution</div><div class="text-gray-300">${data.image_info.width}×${data.image_info.height}</div>
                    <div class="text-gray-500">Format</div><div class="text-gray-300">${data.image_info.format || 'N/A'}</div>
                    <div class="text-gray-500">Size</div><div class="text-gray-300">${formatFileSize(data.image_info.size_bytes)}</div>
                </div>
            </div>
        `;
    }
    if (data.video_info) {
        html += `
            <div class="result-section">
                <div class="result-section-title">Video Info</div>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div class="text-gray-500">Resolution</div><div class="text-gray-300">${data.video_info.width}×${data.video_info.height}</div>
                    <div class="text-gray-500">Duration</div><div class="text-gray-300">${data.video_info.duration_seconds}s</div>
                    <div class="text-gray-500">FPS</div><div class="text-gray-300">${data.video_info.fps}</div>
                    <div class="text-gray-500">Frames</div><div class="text-gray-300">${data.video_info.total_frames}</div>
                </div>
            </div>
        `;
    }

    html += `</div>`;
    panel.innerHTML = html;

    // Animate bars after render
    requestAnimationFrame(() => {
        setTimeout(() => {
            panel.querySelectorAll('.analysis-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 100);
    });
}

// ─── Toast Notifications ─────────────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(60px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ─── Utilities ───────────────────────────────────────────────

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ─── Initialize ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Prevent default drag behavior on the whole page
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => e.preventDefault());

    console.log('🛡️ Guardian AI Frontend Loaded');
});
