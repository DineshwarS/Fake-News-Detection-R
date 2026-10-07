// ==============================================================================
// Fake News Detection Dashboard - Frontend Application Logic
// ==============================================================================

const API_BASE_URL = 'http://127.0.0.1:8000';

// Sample news articles for quick demonstration
const SAMPLE_NEWS = {
  real: `U.S. military to accept transgender recruits on Monday: Pentagon. Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said, after President Donald Trump's administration decided not to appeal rulings that blocked the ban. Two federal appeals courts, one in Washington and one in Virginia, had rejected the administration's request to put on hold orders by lower court judges.`,
  fake: `Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing. Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shoutout to his enemies, haters, and the very dishonest fake news media. The president posted an utterly bizarre and unhinged tweet from his private Mar-a-Lago club ranting about his political opponents while ignoring basic holiday cheer.`
};

// DOM Elements
const newsTextInput = document.getElementById('newsTextInput');
const charCount = document.getElementById('charCount');
const wordCount = document.getElementById('wordCount');
const predictBtn = document.getElementById('predictBtn');
const btnSpinner = document.getElementById('btnSpinner');
const btnLabel = document.getElementById('btnLabel');
const loadRealSampleBtn = document.getElementById('loadRealSampleBtn');
const loadFakeSampleBtn = document.getElementById('loadFakeSampleBtn');
const clearInputBtn = document.getElementById('clearInputBtn');

const backendStatus = document.getElementById('backendStatus');
const statusText = document.getElementById('statusText');

const resultBox = document.getElementById('resultBox');
const verdictBadge = document.getElementById('verdictBadge');
const verdictIcon = document.getElementById('verdictIcon');
const verdictLabel = document.getElementById('verdictLabel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const fakeProbVal = document.getElementById('fakeProbVal');
const fakeProbBar = document.getElementById('fakeProbBar');
const realProbVal = document.getElementById('realProbVal');
const realProbBar = document.getElementById('realProbBar');
const resultTimestamp = document.getElementById('resultTimestamp');
const analysisMeta = document.getElementById('analysisMeta');

// Metrics DOM Elements
const metricAccuracy = document.getElementById('metricAccuracy');
const metricPrecision = document.getElementById('metricPrecision');
const metricRecall = document.getElementById('metricRecall');
const metricF1 = document.getElementById('metricF1');

const cmTP = document.getElementById('cmTP');
const cmFP = document.getElementById('cmFP');
const cmTN = document.getElementById('cmTN');
const cmFN = document.getElementById('cmFN');
const cmTotalCount = document.getElementById('cmTotalCount');

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  checkBackendHealth();
  fetchModelMetrics();
});

function setupEventListeners() {
  // Real-time character & word counters
  newsTextInput.addEventListener('input', updateInputCounters);

  // Quick preset sample buttons
  loadRealSampleBtn.addEventListener('click', () => {
    newsTextInput.value = SAMPLE_NEWS.real;
    updateInputCounters();
    newsTextInput.focus();
  });

  loadFakeSampleBtn.addEventListener('click', () => {
    newsTextInput.value = SAMPLE_NEWS.fake;
    updateInputCounters();
    newsTextInput.focus();
  });

  clearInputBtn.addEventListener('click', () => {
    newsTextInput.value = '';
    updateInputCounters();
    resultBox.classList.add('hidden');
    newsTextInput.focus();
  });

  // Predict button
  predictBtn.addEventListener('click', handlePredict);

  // Keyboard shortcut: Ctrl/Cmd + Enter to trigger prediction
  newsTextInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handlePredict();
    }
  });
}

function updateInputCounters() {
  const text = newsTextInput.value;
  charCount.textContent = `${text.length} characters`;
  
  const words = text.trim().split(/\s+/).filter(w => w.length > 0);
  wordCount.textContent = `${words.length} words`;
}

// 1. Backend Health Check
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      backendStatus.className = 'status-pill online';
      statusText.textContent = data.model_loaded ? 'R Backend Online' : 'Backend Ready (Model Loading)';
      return true;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    backendStatus.className = 'status-pill offline';
    statusText.textContent = 'R Backend Offline';
    return false;
  }
}

// 2. Fetch Model Evaluation Metrics
async function fetchModelMetrics() {
  try {
    const res = await fetch(`${API_BASE_URL}/metrics`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    
    const data = await res.json();
    renderMetrics(data);
  } catch (err) {
    console.warn('Metrics endpoint not ready yet:', err);
    // Keep placeholder or fallback message
    cmTotalCount.textContent = 'Metrics will appear once model training finishes.';
  }
}

function renderMetrics(data) {
  if (data.accuracy !== undefined) {
    metricAccuracy.textContent = `${data.accuracy}%`;
    metricPrecision.textContent = `${data.precision}%`;
    metricRecall.textContent = `${data.recall}%`;
    metricF1.textContent = `${data.f1_score}%`;
  }

  if (data.confusion_matrix) {
    const cm = data.confusion_matrix;
    cmTP.textContent = Number(cm.true_positive).toLocaleString();
    cmFP.textContent = Number(cm.false_positive).toLocaleString();
    cmTN.textContent = Number(cm.true_negative).toLocaleString();
    cmFN.textContent = Number(cm.false_negative).toLocaleString();

    if (cm.total_tested) {
      cmTotalCount.textContent = `Total Tested: ${Number(cm.total_tested).toLocaleString()} articles`;
    }
  }
}

// 3. Handle News Prediction
async function handlePredict() {
  const text = newsTextInput.value.trim();

  if (!text) {
    alert('Please enter or paste a news article headline or text.');
    newsTextInput.focus();
    return;
  }

  // Set loading state
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || errorData.message || `Server error (status ${response.status})`);
    }

    const data = await response.json();
    displayPredictionResult(data);

  } catch (error) {
    console.error('Prediction failed:', error);
    alert(`Prediction Failed: ${error.message}\n\nPlease verify that the R Plumber server is running on http://127.0.0.1:8000.`);
  } finally {
    setLoading(false);
  }
}

function displayPredictionResult(data) {
  const rawPred = Array.isArray(data.prediction) ? data.prediction[0] : data.prediction;
  const isFake = String(rawPred).toUpperCase() === 'FAKE';
  
  const rawConf = Array.isArray(data.confidence) ? data.confidence[0] : data.confidence;
  const confidence = Number(rawConf) || 0;

  const rawFakeProb = Array.isArray(data.fake_probability) ? data.fake_probability[0] : data.fake_probability;
  const fakeProb = Number(rawFakeProb) || 0;

  const rawRealProb = Array.isArray(data.real_probability) ? data.real_probability[0] : data.real_probability;
  const realProb = Number(rawRealProb) || 0;

  // Update box container classes
  resultBox.classList.remove('hidden', 'is-fake', 'is-real');
  resultBox.classList.add(isFake ? 'is-fake' : 'is-real');

  // Update verdict badge
  verdictBadge.className = `verdict-badge ${isFake ? 'fake-badge' : 'real-badge'}`;
  verdictIcon.textContent = isFake ? '⚠️' : '✅';
  verdictLabel.textContent = isFake ? 'FAKE NEWS' : 'REAL NEWS';

  // Update confidence meter
  confidenceValue.textContent = `${confidence}%`;
  confidenceBar.className = `confidence-fill ${isFake ? 'fake-fill' : 'real-fill'}`;
  confidenceBar.style.width = `${confidence}%`;

  // Update breakdown bars
  fakeProbVal.textContent = `${fakeProb}%`;
  fakeProbBar.style.width = `${fakeProb}%`;

  realProbVal.textContent = `${realProb}%`;
  realProbBar.style.width = `${realProb}%`;

  // Metadata
  resultTimestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  analysisMeta.textContent = `Analyzed ${data.text_length_chars || newsTextInput.value.length} characters • ${data.cleaned_word_count || 0} vocabulary tokens matched`;

  // Smooth scroll to result
  resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setLoading(isLoading) {
  if (isLoading) {
    predictBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnLabel.textContent = 'Classifying with R Model...';
  } else {
    predictBtn.disabled = false;
    btnSpinner.style.display = 'none';
    btnLabel.textContent = 'Detect Fake News';
  }
}
