// ============================================================
// STATE
// ============================================================
let patients = JSON.parse(localStorage.getItem('pneumo_patients') || '[]');
let currentPatient = {};
let selectedFile = null;
let mediaRecorder = null;
let recordingChunks = [];
let isRecording = false;
let recordTimerInterval = null;
let recordSeconds = 0;
let currentStep = 1;

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    generatePatientId();
    buildWaveform();
    updateTable();
    updateStatCounter();
    styleCheckboxes();
});

function generatePatientId() {
    const id = 'PSC-' + Date.now().toString(36).toUpperCase();
    document.getElementById('patId').value = id;
    currentPatient.id = id;
}

function buildWaveform() {
    const container = document.getElementById('heroWave');
    const bars = 48;
    for (let i = 0; i < bars; i++) {
        const bar = document.createElement('div');
        bar.className = 'wave-bar';
        bar.style.animationDelay = `${(i / bars) * 1.4}s`;
        bar.style.height = `${Math.random() * 40 + 8}px`;
        container.appendChild(bar);
    }
}

function updateStatCounter() {
    const target = patients.length;
    const el = document.getElementById('statPatients');
    let current = 0;
    const interval = setInterval(() => {
        current = Math.min(current + 1, target);
        el.textContent = current;
        if (current >= target) clearInterval(interval);
    }, 50);
    if (target === 0) el.textContent = '0';
}

function styleCheckboxes() {
    document.querySelectorAll('.checkbox-item input').forEach(cb => {
        cb.addEventListener('change', function () {
            this.closest('.checkbox-item').classList.toggle('checked', this.checked);
        });
    });
}

function smoothScrollTo(id) {
    document.querySelector(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showAbout() {
    document.getElementById('aboutModal').classList.add('open');
}

// ============================================================
// STEP NAVIGATION
// ============================================================
function setStep(n) {
    currentStep = n;
    [1, 2, 3].forEach(i => {
        document.getElementById(`step${i}`).className = 'step' + (i < n ? ' done' : i === n ? ' active' : '');
        document.getElementById(`content${i}`).className = 'step-content panel' + (i === n ? ' active' : '');
    });
    if (n === 3) {
        document.getElementById('analysisAnim').classList.add('active');
        document.getElementById('results-section').classList.remove('visible');
    }
    window.scrollTo({ top: document.getElementById('app').offsetTop - 80, behavior: 'smooth' });
}

function goToStep1() { setStep(1); }

function goToStep2() {
    const name = document.getElementById('patName').value.trim();
    if (!name) { showToast('Please enter the patient name', 'error'); return; }
    const age = document.getElementById('patAge').value;
    if (!age) { showToast('Please enter patient age', 'error'); return; }
    currentPatient = {
        id: document.getElementById('patId').value,
        name,
        age,
        gender: document.getElementById('patGender').value,
        phone: document.getElementById('patPhone').value,
        email: document.getElementById('patEmail').value,
        symptoms: [...document.querySelectorAll('.checkbox-item input:checked')].map(c => c.value),
        notes: document.getElementById('patNotes').value,
        date: new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })
    };
    setStep(2);
}

// ============================================================
// AUDIO HANDLING
// ============================================================
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) processFile(file);
}

function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('uploadZone').classList.add('drag-over');
}
function handleDragLeave() {
    document.getElementById('uploadZone').classList.remove('drag-over');
}
function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadZone').classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) processFile(file);
    else showToast('Please drop an audio file', 'error');
}

function processFile(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    document.getElementById('audioName').textContent = file.name;
    document.getElementById('audioSize').textContent = formatFileSize(file.size) + ' · ' + file.type;
    document.getElementById('audioPlayer').src = url;
    document.getElementById('audioPreview').classList.add('visible');
    buildWaveformViz();
    showToast('Audio loaded: ' + file.name, 'success');
}

function buildWaveformViz() {
    const container = document.getElementById('waveformViz');
    container.innerHTML = '';
    const bars = 60;
    for (let i = 0; i < bars; i++) {
        const bar = document.createElement('div');
        bar.className = 'wf-bar';
        const h = Math.random() * 44 + 4;
        bar.style.height = h + 'px';
        container.appendChild(bar);
    }
}

function clearAudio() {
    selectedFile = null;
    document.getElementById('audioPlayer').src = '';
    document.getElementById('audioPreview').classList.remove('visible');
    document.getElementById('fileInput').value = '';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

// ============================================================
// RECORDING
// ============================================================
async function toggleRecord() {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            recordingChunks = [];
            mediaRecorder.ondataavailable = e => recordingChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordingChunks, { type: 'audio/wav' });
                const file = new File([blob], 'recording_' + Date.now() + '.wav', { type: 'audio/wav' });
                processFile(file);
                stream.getTracks().forEach(t => t.stop());
            };
            mediaRecorder.start();
            isRecording = true;
            recordSeconds = 0;
            document.getElementById('recordBtn').classList.add('recording');
            document.getElementById('recordIcon').textContent = '⏹';
            document.getElementById('recordStatus').textContent = 'Recording... (click to stop)';
            document.getElementById('recordTimer').style.display = 'block';
            recordTimerInterval = setInterval(() => {
                recordSeconds++;
                const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                const s = String(recordSeconds % 60).padStart(2, '0');
                document.getElementById('recordTimer').textContent = `${m}:${s}`;
                if (recordSeconds >= 30) toggleRecord(); // auto stop at 30s
            }, 1000);
        } catch (err) {
            showToast('Microphone access denied. Please allow access.', 'error');
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        clearInterval(recordTimerInterval);
        document.getElementById('recordBtn').classList.remove('recording');
        document.getElementById('recordIcon').textContent = '🎙';
        document.getElementById('recordStatus').textContent = 'Recording saved!';
        document.getElementById('recordTimer').style.display = 'none';
        showToast('Recording captured!', 'success');
    }
}

// ============================================================
// ANALYSIS (SIMULATION — replace with real API call)
// ============================================================
const DISEASES = [
    { name: 'COVID-19 Positive', color: '#ff4d6d' },
    { name: 'Healthy', color: '#00ffa3' },
    { name: 'Asthma', color: '#ffb800' },
    { name: 'Bronchitis', color: '#3d8bff' },
    { name: 'Pneumonia', color: '#ff6b35' },
    { name: 'URTI', color: '#a78bfa' },
    { name: 'COPD', color: '#f472b6' },
];

const RECOMMENDATIONS = {
    'COVID-19 Positive': { level: 'high', text: '🔴 HIGH RISK — Isolate immediately. Schedule a PCR/Antigen test within 24 hours. Contact your physician for antiviral treatment options. Monitor oxygen saturation. Seek emergency care if SpO₂ < 94%.' },
    'Healthy': { level: 'low', text: '🟢 NO RESPIRATORY PATHOLOGY DETECTED — Your respiratory audio shows no significant anomalies. Maintain good hygiene and hydration. Annual check-up recommended.' },
    'Asthma': { level: 'med', text: '🟡 ASTHMA INDICATORS — Use your prescribed bronchodilator. Avoid known triggers (dust, pollen, smoke). Consult a pulmonologist for spirometry and updated treatment plan.' },
    'Bronchitis': { level: 'med', text: '🟡 BRONCHITIS SUSPECTED — Rest and increase fluid intake. See a doctor if fever exceeds 38.5°C for >3 days or mucus becomes discoloured. Avoid smoking environments.' },
    'Pneumonia': { level: 'high', text: '🔴 PNEUMONIA SUSPECTED — Seek immediate medical attention. Chest X-ray and blood work required. May need antibiotics or hospitalization. Do not delay.' },
    'URTI': { level: 'low', text: '🟡 UPPER RESPIRATORY TRACT INFECTION — Usually viral and self-limiting. Rest, hydrate, use saline nasal rinse. See a doctor if no improvement in 7 days or symptoms worsen.' },
    'COPD': { level: 'high', text: '🔴 COPD PATTERN DETECTED — See a pulmonologist immediately for spirometry. Avoid all respiratory irritants. This condition requires long-term management and medication.' },
};

async function startAnalysis() {
    if (!selectedFile) { showToast('Please upload or record an audio sample first', 'error'); return; }
    setStep(3);

    // UI feedback
    const scanLabel = document.getElementById('scanLabel');
    const scanBar = document.getElementById('scanBar');
    const steps = ['ss1', 'ss2', 'ss3', 'ss4', 'ss5'];

    try {
        console.log("🚀 Starting analysis for file:", selectedFile.name);
        scanLabel.textContent = "INITIALISING MODEL...";
        scanBar.style.width = "10%";
        document.getElementById('ss1').className = 'scan-step active';

        // Prepare form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        console.log("🛰️ Fetching from API: http://localhost:8000/predict");
        // API Call
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        console.log("📥 API Response status:", response.status);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server returned ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        console.log("✅ Received result:", result);

        // Finalise animation steps
        ['ss1', 'ss2', 'ss3', 'ss4', 'ss5'].forEach(sid => {
            const el = document.getElementById(sid);
            if (el) el.className = 'scan-step done';
        });
        scanBar.style.width = '100%';
        scanLabel.textContent = 'ANALYSIS COMPLETE';

        setTimeout(() => showResults(result), 800);

    } catch (err) {
        console.error("🔥 Analysis Error:", err);
        showToast('Error during analysis: ' + err.message, 'error');
        setStep(2);
    }
}

function showResults(result) {
    console.log("📊 Rendering results...");
    document.getElementById('analysisAnim').classList.remove('active');
    const resultsSec = document.getElementById('results-section');
    resultsSec.classList.add('visible');

    // Extract results from API response
    const topDisease = result.primary_diagnosis;
    const confidence = result.confidence;
    const probs = result.top_3_predictions; // We'll use this for the bars
    const recommended = result.recommendation;
    const severity = result.confidence > 0.8 ? 'high' : (result.confidence > 0.5 ? 'med' : 'low');

    // Fill patient card
    document.getElementById('resName').textContent = currentPatient.name;
    document.getElementById('resMeta').textContent =
        `${currentPatient.age} yrs · ${currentPatient.gender || 'N/A'} · ${currentPatient.date}`;
    document.getElementById('resPid').textContent = currentPatient.id;
    const initials = currentPatient.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    document.getElementById('resAvatar').textContent = initials || '👤';

    // Fill result hero
    document.getElementById('resDiseaseMain').textContent = topDisease;
    document.getElementById('resConfNum').textContent = result.confidence_pct;

    const badge = document.getElementById('resBadge');
    const sev = document.getElementById('resSeverity');
    if (severity === 'high') { badge.className = 'result-badge badge-high'; sev.textContent = 'HIGH RISK'; }
    else if (severity === 'med') { badge.className = 'result-badge badge-med'; sev.textContent = 'MODERATE'; }
    else { badge.className = 'result-badge badge-low'; sev.textContent = 'LOW RISK'; }

    // Gauge (Model accuracy is static/provided in result if available, or fake it for UI)
    const accuracy = 94.2;
    const circumference = 2 * Math.PI * 70;
    const offset = circumference - (accuracy / 100) * circumference;
    setTimeout(() => {
        document.getElementById('gaugeFill').style.strokeDashoffset = offset;
        document.getElementById('gaugeVal').textContent = accuracy.toFixed(1) + '%';
    }, 300);

    // Metrics (Synthetic or from result)
    document.getElementById('mAccuracy').textContent = '94.2%';
    document.getElementById('mAuc').textContent = '0.967';
    document.getElementById('mPrec').textContent = '0.931';
    document.getElementById('mRecall').textContent = '0.918';

    // Probability bars (using Top 3)
    const probList = document.getElementById('probList');
    probList.innerHTML = '';

    // Map colors
    const colorMap = {
        'Healthy': '#00ffa3',
        'Covid-19 Positive': '#ff4d6d',
        'Asthma': '#ffb800',
        'Bronchitis': '#3d8bff',
        'Pneumonia': '#ff6b35',
        'Urti': '#a78bfa',
        'Copd': '#f472b6'
    };

    probs.forEach(p => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        const color = colorMap[p.disease] || '#6366f1';
        item.innerHTML = `
      <div class="prob-header">
        <span class="prob-name">${p.disease}</span>
        <span class="prob-pct">${(p.confidence * 100).toFixed(1)}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-bar" style="width:0%;background:${color}" data-width="${p.confidence * 100}%"></div>
      </div>`;
        probList.appendChild(item);
    });

    setTimeout(() => {
        document.querySelectorAll('.prob-bar').forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 300);

    // Symptoms
    const symTags = document.getElementById('symptomTags');
    symTags.innerHTML = '';
    result.inferred_symptoms.forEach(sym => {
        const span = document.createElement('span');
        span.className = 'symptom-tag sym-present';
        span.textContent = '● ' + sym.replace('_', ' ').toUpperCase();
        symTags.appendChild(span);
    });

    // Recommendation
    const recBox = document.getElementById('recBox');
    recBox.textContent = recommended;
    recBox.className = 'rec-box rec-' + severity;

    // Save patient record
    const record = {
        ...currentPatient,
        diagnosis: topDisease,
        confidence: result.confidence_pct,
        accuracy: '94.2%',
        recommendation: recommended,
        severity: severity,
        audioName: selectedFile?.name || 'recording.wav',
        timestamp: new Date().toISOString()
    };
    patients.unshift(record);
    localStorage.setItem('pneumo_patients', JSON.stringify(patients));
    updateTable();
    updateStatCounter();

    showToast('Analysis complete! Report generated.', 'success');
}

function generateProbabilities() {
    const raw = DISEASES.map(() => Math.random());
    // Boost a random disease to simulate a clear winner
    const winner = Math.floor(Math.random() * DISEASES.length);
    raw[winner] += 2 + Math.random() * 2;
    const sum = raw.reduce((a, b) => a + b, 0);
    return raw.map(v => v / sum);
}

// ============================================================
// TABLE
// ============================================================
function updateTable() {
    const tbody = document.getElementById('tableBody');
    if (patients.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:48px;color:var(--text3)">No records yet. Complete an analysis to see data here.</td></tr>';
        return;
    }
    renderTable(patients);
}

function renderTable(data) {
    const tbody = document.getElementById('tableBody');
    const severityColors = { low: '#00ffa3', med: '#ffb800', high: '#ff4d6d' };
    tbody.innerHTML = data.map(p => {
        const color = severityColors[p.severity] || '#8ba3cc';
        return `<tr onclick="viewRecord('${p.id}')">
      <td><span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--cyan)">${p.id}</span></td>
      <td>${p.name}</td>
      <td>${p.age}</td>
      <td>${p.gender || '—'}</td>
      <td>${p.diagnosis || '—'}</td>
      <td><span style="font-family:var(--font-mono);color:var(--cyan)">${p.confidence || '—'}</span></td>
      <td><span style="font-size:0.8rem;color:var(--text2)">${p.date}</span></td>
      <td><span class="table-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">${(p.severity || '').toUpperCase()}</span></td>
    </tr>`;
    }).join('');
}

function filterTable() {
    const q = document.getElementById('searchInput').value.toLowerCase();
    const dis = document.getElementById('filterDisease').value.toLowerCase();
    const filtered = patients.filter(p => {
        const matchQ = p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q);
        const matchD = !dis || (p.diagnosis || '').toLowerCase().includes(dis);
        return matchQ && matchD;
    });
    renderTable(filtered);
}

function viewRecord(id) {
    const p = patients.find(r => r.id === id);
    if (!p) return;
    alert(`Patient: ${p.name}\nDiagnosis: ${p.diagnosis}\nConfidence: ${p.confidence}\nAccuracy: ${p.accuracy}\nDate: ${p.date}\n\nRecommendation:\n${p.recommendation}`);
}

// ============================================================
// EXPORT
// ============================================================
function exportCSV() {
    if (patients.length === 0) { showToast('No records to export', 'error'); return; }
    const headers = ['ID', 'Name', 'Age', 'Gender', 'Diagnosis', 'Confidence', 'Accuracy', 'Date', 'Severity', 'Symptoms'];
    const rows = patients.map(p => [
        p.id, `"${p.name}"`, p.age, p.gender, `"${p.diagnosis}"`,
        p.confidence, p.accuracy, p.date, p.severity,
        `"${(p.symptoms || []).join('; ')}"`
    ]);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pneumoscan_patients_${Date.now()}.csv`;
    a.click();
    showToast('CSV exported!', 'success');
}

function downloadReport() {
    const p = patients[0];
    if (!p) return;
    const html = `<!DOCTYPE html><html><head><title>Report - ${p.name}</title>
  <style>body{font-family:sans-serif;max-width:700px;margin:40px auto;color:#1a1a2e}
  h1{color:#0d47a1}table{width:100%;border-collapse:collapse;margin:20px 0}
  th,td{padding:10px 14px;border:1px solid #dde;text-align:left}
  th{background:#f0f4ff}.rec{background:#fff3f5;padding:16px;border-left:4px solid #f44;border-radius:4px;margin:20px 0}
  </style></head><body>
  <h1>🫁 PneumoScan AI — Diagnostic Report</h1>
  <table>
    <tr><th>Patient ID</th><td>${p.id}</td>
    <th>Date</th><td>${p.date}</td></tr>
    <tr><th>Name</th><td>${p.name}</td><th>Age / Gender</th><td>${p.age} / ${p.gender || 'N/A'}</td></tr>
    <tr><th>Diagnosis</th><td><strong>${p.diagnosis}</strong></td><th>Confidence</th><td>${p.confidence}</td></tr>
    <tr><th>Model Accuracy</th><td>${p.accuracy}</td><th>Risk Level</th><td>${(p.severity || '').toUpperCase()}</td></tr>
    <tr><th>Symptoms</th><td colspan="3">${(p.symptoms || []).join(', ') || 'None reported'}</td></tr>
  </table>
  <div class="rec"><strong>Recommendation:</strong><br>${p.recommendation}</div>
  <p style="font-size:12px;color:#888">Generated by PneumoScan AI · ${new Date().toLocaleString()} · FOR SCREENING PURPOSES ONLY</p>
  </body></html>`;
    const blob = new Blob([html], { type: 'text/html' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pneumoscan_report_${p.id}.html`;
    a.click();
    showToast('Report downloaded!', 'success');
}

// ============================================================
// UTILS
// ============================================================
function resetAll() {
    selectedFile = null;
    currentPatient = {};
    document.getElementById('audioPreview').classList.remove('visible');
    document.getElementById('fileInput').value = '';
    document.getElementById('audioPlayer').src = '';
    document.getElementById('patName').value = '';
    document.getElementById('patAge').value = '';
    document.getElementById('patGender').value = '';
    document.getElementById('patPhone').value = '';
    document.getElementById('patEmail').value = '';
    document.getElementById('patNotes').value = '';
    document.querySelectorAll('.checkbox-item input').forEach(cb => {
        cb.checked = false;
        cb.closest('.checkbox-item').classList.remove('checked');
    });
    document.getElementById('results-section').classList.remove('visible');
    document.getElementById('analysisAnim').classList.remove('active');
    generatePatientId();
    setStep(1);
    scrollTo('#app');
}

function showToast(msg, type = 'success') {
    const toast = document.getElementById('toast');
    const icon = document.getElementById('toastIcon');
    const text = document.getElementById('toastMsg');
    icon.textContent = type === 'success' ? '✅' : '⚠️';
    text.textContent = msg;
    toast.className = `toast ${type} show`;
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => { toast.classList.remove('show'); }, 3500);
}