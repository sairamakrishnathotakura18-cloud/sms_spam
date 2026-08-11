/* ==========================================================================
   SpamGuard AI — Main JavaScript Application Engine
   ========================================================================== */

const API_BASE_URL = 'http://127.0.0.1:5000';

// Global application state
let currentMetrics = null;
let lastBatchResults = [];

// Sample test messages for quick testing
const SAMPLES = {
    spam1: "WINNER! You have won a $1000 Walmart gift card. Claim now at http://win-now.com or call 0800-123-456",
    spam2: "URGENT! Your bank account has been locked due to suspicious activity. Verify credentials at http://secure-bank-login.net immediately.",
    ham1: "Hey! Are we still on for dinner tonight at 7 PM? Let me know if you want me to pick up anything.",
    ham2: "Hi Team, the project documentation has been updated. Please review the PR before tomorrow's standup."
};

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initEventListeners();
    checkApiHealth();
    fetchModelMetrics();
    loadPredictionHistory();
});

/* ==========================================================================
   1. Theme Toggle & Mobile Navbar
   ========================================================================== */
function initTheme() {
    const savedTheme = localStorage.getItem('spamguard_theme') || 'light';
    if (savedTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        document.documentElement.removeAttribute('data-theme');
    }
}

function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('spamguard_theme', 'light');
        showToast("Switched to Light mode", "info");
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('spamguard_theme', 'dark');
        showToast("Switched to Dark mode", "info");
    }
}

function initEventListeners() {
    // Theme toggle button
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

    // Mobile Navigation Hamburger
    const mobileBtn = document.getElementById('mobile-menu-btn');
    const navMenu = document.getElementById('nav-menu');
    if (mobileBtn && navMenu) {
        mobileBtn.addEventListener('click', () => {
            navMenu.classList.toggle('show');
        });
    }

    // Textarea input length counter
    const textarea = document.getElementById('sms-input');
    if (textarea) {
        textarea.addEventListener('input', () => {
            const len = textarea.value.length;
            document.getElementById('char-counter').textContent = `${len} / 5000 chars`;
        });

        // Ctrl + Enter shortcut
        textarea.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                analyzeSMS();
            }
        });
    }

    // Drag & Drop for Batch CSV Dropzone
    const dropzone = document.getElementById('dropzone');
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = 'var(--primary)';
                dropzone.style.background = 'var(--primary-light)';
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.style.borderColor = '';
                dropzone.style.background = '';
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                processCSVFile(files[0]);
            }
        });
    }
}

/* ==========================================================================
   2. API Integration & Health Check
   ========================================================================== */
async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        if (!res.ok || !data.model_loaded) {
            console.warn("API Server health check warning:", data);
        }
    } catch (err) {
        showToast("Backend Flask server unavailable. Make sure app.py is running.", "error");
    }
}

async function fetchModelMetrics() {
    try {
        const res = await fetch(`${API_BASE_URL}/metrics`);
        if (!res.ok) return;

        const data = await res.json();
        currentMetrics = data;
        updateDashboardMetrics(data);
    } catch (err) {
        console.warn("Could not fetch metrics from API:", err);
    }
}

function updateDashboardMetrics(data) {
    if (!data || !data.primary_metrics) return;

    const pm = data.primary_metrics;
    document.getElementById('metric-accuracy').textContent = `${pm.accuracy}%`;
    document.getElementById('bar-accuracy').style.width = `${pm.accuracy}%`;

    document.getElementById('metric-precision').textContent = `${pm.precision}%`;
    document.getElementById('bar-precision').style.width = `${pm.precision}%`;

    document.getElementById('metric-recall').textContent = `${pm.recall}%`;
    document.getElementById('bar-recall').style.width = `${pm.recall}%`;

    document.getElementById('metric-f1').textContent = `${pm.f1_score}%`;
    document.getElementById('bar-f1').style.width = `${pm.f1_score}%`;
}

/* ==========================================================================
   3. Single SMS Message Analyzer
   ========================================================================== */
function fillSample(sampleKey) {
    if (SAMPLES[sampleKey]) {
        const textarea = document.getElementById('sms-input');
        textarea.value = SAMPLES[sampleKey];
        document.getElementById('char-counter').textContent = `${textarea.value.length} / 5000 chars`;
        showToast("Sample message loaded into analyzer", "info");
    }
}

function clearInput() {
    document.getElementById('sms-input').value = '';
    document.getElementById('char-counter').textContent = '0 / 5000 chars';
    document.getElementById('result-container').classList.add('hidden');
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('result-card').classList.add('hidden');
}

async function analyzeSMS() {
    const textarea = document.getElementById('sms-input');
    const inputVal = textarea.value.trim();

    if (!inputVal) {
        showToast("Please enter an SMS message to analyze.", "error");
        textarea.focus();
        return;
    }

    // Toggle UI States
    const resultContainer = document.getElementById('result-container');
    const loadingState = document.getElementById('loading-state');
    const resultCard = document.getElementById('result-card');
    const analyzeBtn = document.getElementById('analyze-btn');

    resultContainer.classList.remove('hidden');
    resultCard.classList.add('hidden');
    loadingState.classList.remove('hidden');

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing with AI...`;

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: inputVal })
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Prediction request failed.");
        }

        displayResult(data);
        savePredictionToHistory(data);
        showToast(`Analysis complete: Classified as ${data.prediction}`, data.is_spam ? "error" : "success");

    } catch (err) {
        showToast(`Error: ${err.message}`, "error");
        resultContainer.classList.add('hidden');
    } finally {
        loadingState.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<span>Analyze Message →</span>`;
    }
}

function displayResult(data) {
    const resultCard = document.getElementById('result-card');
    resultCard.classList.remove('hidden');
    resultCard.className = `card result-card ${data.is_spam ? 'spam' : 'ham'}`;

    // Header Verdict
    const iconBox = document.getElementById('result-icon');
    const riskBadge = document.getElementById('risk-badge');
    const heading = document.getElementById('result-heading');
    const recText = document.getElementById('rec-text');

    if (data.is_spam) {
        iconBox.className = 'fa-solid fa-triangle-exclamation';
        riskBadge.textContent = 'HIGH RISK';
        heading.textContent = '⚠ Suspicious Message Detected';
        recText.textContent = 'High Risk: Message contains strong indicators of spam, scam, or phishing. Do not click links or reply.';
    } else {
        iconBox.className = 'fa-solid fa-circle-check';
        riskBadge.textContent = 'LOW RISK';
        heading.textContent = '✓ Message Looks Safe';
        recText.textContent = 'Safe Message: Message displays normal conversational patterns with no suspicious spam triggers.';
    }

    // Confidence & Progress
    document.getElementById('conf-val').textContent = `${data.confidence}%`;
    document.getElementById('prob-val').textContent = `${data.confidence}%`;
    document.getElementById('progress-fill').style.width = `${data.confidence}%`;

    // Cleaned Tokens
    document.getElementById('cleaned-tokens-val').textContent = data.cleaned_message || '[No feature tokens found]';

    // Why Flagged Feature Tags
    const grid = document.getElementById('pattern-tags-grid');
    grid.innerHTML = '';

    if (data.explanation && data.explanation.length > 0) {
        data.explanation.forEach(item => {
            const isSpamWord = item.category === 'spam';
            const pill = document.createElement('div');
            pill.className = `pattern-pill ${isSpamWord ? 'spam' : 'ham'}`;
            pill.innerHTML = `
                <i class="fa-solid ${isSpamWord ? 'fa-triangle-exclamation' : 'fa-check'}"></i>
                <span>${escapeHtml(item.word)}</span>
                <small>(${item.score})</small>
            `;
            grid.appendChild(pill);
        });
    } else {
        grid.innerHTML = '<span style="font-size:0.85rem; color:var(--text-muted)">No distinct trigger words highlighted.</span>';
    }
}

/* ==========================================================================
   4. Batch CSV Processor
   ========================================================================== */
function handleCSVFileSelect(e) {
    const file = e.target.files[0];
    if (file) processCSVFile(file);
}

async function processCSVFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast("Invalid file type. Please upload a .csv file.", "error");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    showToast("Processing CSV batch dataset...", "info");

    try {
        const res = await fetch(`${API_BASE_URL}/batch-predict`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Batch classification failed.");

        renderBatchResults(data);
        showToast(`Processed ${data.total} messages successfully!`, "success");
    } catch (err) {
        showToast(`CSV Error: ${err.message}`, "error");
    }
}

function renderBatchResults(data) {
    lastBatchResults = data.results || [];
    const area = document.getElementById('batch-results-area');
    area.classList.remove('hidden');

    document.getElementById('batch-total-count').textContent = data.total;
    document.getElementById('batch-spam-count').textContent = data.spam_count;
    document.getElementById('batch-ham-count').textContent = data.ham_count;

    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    lastBatchResults.forEach((row, idx) => {
        const tr = document.createElement('tr');
        const isSpam = row.is_spam;
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td style="max-width: 250px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(row.raw_message)}</td>
            <td>
                <span class="badge ${isSpam ? 'spam' : 'ham'}">
                    ${isSpam ? '🚨 Spam' : '✅ Safe'}
                </span>
            </td>
            <td><strong>${row.confidence}%</strong></td>
        `;
        tbody.appendChild(tr);
    });
}

function downloadSampleCSV() {
    const content = "data:text/csv;charset=utf-8,message\n" +
        "\"Congratulations! You won a $1000 gift card. Call 0800 now!\"\n" +
        "\"Hey mom, will be late for dinner today.\"\n" +
        "\"URGENT: Your account access is restricted. Click link http://verify-now.org\"\n";
    
    const encoded = encodeURI(content);
    const link = document.createElement("a");
    link.setAttribute("href", encoded);
    link.setAttribute("download", "sample_sms_messages.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Downloaded sample SMS test CSV", "info");
}

function downloadBatchResultsCSV() {
    if (lastBatchResults.length === 0) return;

    let csv = "ID,SMS Message,Prediction,Confidence %,Spam Probability %\n";
    lastBatchResults.forEach((r, i) => {
        const cleanMsg = (r.raw_message || "").replace(/"/g, '""');
        csv += `${i + 1},"${cleanMsg}","${r.prediction}",${r.confidence},${r.spam_probability}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `spamguard_batch_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("Exported prediction results CSV", "success");
}

/* ==========================================================================
   5. Local Prediction History
   ========================================================================== */
function savePredictionToHistory(item) {
    let history = JSON.parse(localStorage.getItem('spamguard_history') || '[]');
    const record = {
        id: Date.now(),
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        message: item.message,
        prediction: item.prediction,
        is_spam: item.is_spam,
        confidence: item.confidence
    };

    history.unshift(record);
    if (history.length > 15) history = history.slice(0, 15);
    localStorage.setItem('spamguard_history', JSON.stringify(history));
    renderHistory();
}

function loadPredictionHistory() {
    renderHistory();
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('spamguard_history') || '[]');
    const tbody = document.getElementById('history-table-body');
    const emptyMsg = document.getElementById('empty-history-msg');

    if (history.length === 0) {
        tbody.innerHTML = '';
        emptyMsg.classList.remove('hidden');
        return;
    }

    emptyMsg.classList.add('hidden');
    tbody.innerHTML = '';

    history.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="white-space: nowrap;"><i class="fa-regular fa-clock"></i> ${item.time}</td>
            <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(item.message)}</td>
            <td>
                <span class="badge ${item.is_spam ? 'spam' : 'ham'}">
                    ${item.is_spam ? '🚨 SPAM' : '✅ SAFE'}
                </span>
            </td>
            <td><strong>${item.confidence}%</strong></td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="retestHistory('${escapeHtml(item.message)}')">
                    <i class="fa-solid fa-rotate-right"></i> Retest
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function retestHistory(msg) {
    document.getElementById('sms-input').value = unescapeHtml(msg);
    document.getElementById('char-counter').textContent = `${msg.length} / 5000 chars`;
    window.location.hash = '#analyzer';
    analyzeSMS();
}

function clearHistory() {
    localStorage.removeItem('spamguard_history');
    renderHistory();
    showToast("History cleared", "info");
}

/* ==========================================================================
   6. Utilities & Security Escaping
   ========================================================================== */
function copyCode(btn) {
    const codeBlock = btn.parentElement.nextElementSibling.querySelector('code');
    if (!codeBlock) return;

    navigator.clipboard.writeText(codeBlock.textContent).then(() => {
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = "Copy"; }, 2000);
        showToast("Code snippet copied to clipboard", "success");
    });
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let icon = 'fa-circle-info';
    if (type === 'success') icon = 'fa-circle-check';
    if (type === 'error') icon = 'fa-triangle-exclamation';

    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${escapeHtml(message)}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function unescapeHtml(str) {
    if (!str) return '';
    return str.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#039;/g, "'");
}
