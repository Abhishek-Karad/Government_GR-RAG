const CONFIG = Object.freeze({
  candidate_k: 30,
  top_k: 6,
  max_per_document: 7,
  rerank: true,
  show_evidence: true,
  generate_answer: true,
});

const questionInput = document.querySelector('#question');
const askButton = document.querySelector('#ask-button');
const loading = document.querySelector('#loading');
const errorBox = document.querySelector('#error-box');
const answerSection = document.querySelector('#answer-section');
const evidenceSection = document.querySelector('#evidence-section');

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;',
  }[char]));
}

function formatAnswer(value) {
  const escaped = escapeHtml(value || 'No answer was returned.');
  // Gemini may return lists either one item per line or inline. Insert safe
  // breaks before short numbered points without splitting years or amounts.
  const prepared = escaped
    .replace(/\s+(?=\d{1,2}[.)]\s+)/g, '\n')
    .replace(/\s+(?=[•*]\s+)/g, '\n');
  const lines = prepared.split(/\r?\n/);
  const blocks = [];
  let list = null;

  const closeList = () => {
    if (list) {
      blocks.push(`<${list.type}>${list.items.join('')}</${list.type}>`);
      list = null;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim().replace(/^\*\*(\d{1,2}[.)])\*\*\s+/, '$1 ');
    if (!line) {
      closeList();
      continue;
    }
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      const type = bullet ? 'ul' : 'ol';
      if (!list || list.type !== type) {
        closeList();
        list = { type, items: [] };
      }
      list.items.push(`<li>${formatInline(bullet ? bullet[1] : numbered[1])}</li>`);
      continue;
    }
    closeList();
    blocks.push(`<p>${formatInline(line)}</p>`);
  }
  closeList();
  return blocks.join('');
}

function formatInline(value) {
  return value
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}

function setBusy(isBusy) {
  askButton.disabled = isBusy;
  loading.hidden = !isBusy;
  askButton.querySelector('.button-label').textContent = isBusy ? 'Searching…' : 'Ask MahaGR';
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function pageLabel(item) {
  return item.page_label ? `Page ${item.page_label}` : 'Page unavailable';
}

function renderAnswer(payload) {
  const answer = payload.answer || {};
  document.querySelector('#answer-text').innerHTML = formatAnswer(answer.answer);
  const status = document.querySelector('#answer-status');
  status.textContent = answer.answer_status === 'supported' ? 'Evidence supported' : 'Review evidence';
  status.className = `answer-status ${answer.answer_status === 'supported' ? 'supported' : 'review'}`;

  const limitations = document.querySelector('#limitations');
  const items = answer.limitations || [];
  limitations.hidden = items.length === 0;
  limitations.innerHTML = items.map((item) => `<span>ⓘ ${escapeHtml(item)}</span>`).join('');

  const claims = answer.claims || [];
  const claimsSection = document.querySelector('#claims');
  claimsSection.hidden = claims.length === 0;
  document.querySelector('#claims-list').innerHTML = claims.map((claim) => `
    <div class="claim ${claim.verified ? 'claim-verified' : 'claim-review'}">
      <span class="claim-icon">${claim.verified ? '✓' : '!'}</span>
      <span>${escapeHtml(claim.text)}</span>
      <span class="claim-citations">${(claim.evidence_ids || []).map(escapeHtml).join(' · ')}</span>
    </div>`).join('');
  answerSection.hidden = false;
}

function renderEvidence(payload) {
  const evidence = payload.evidence || [];
  document.querySelector('#evidence-count').textContent = `${evidence.length} passage${evidence.length === 1 ? '' : 's'}`;
  document.querySelector('#evidence-list').innerHTML = evidence.map((item, index) => `
    <article class="evidence-card">
      <div class="evidence-topline">
        <span class="evidence-number">${String(index + 1).padStart(2, '0')}</span>
        <span class="evidence-id">${escapeHtml(item.evidence_id)}</span>
        <span class="evidence-page">${escapeHtml(pageLabel(item))}</span>
      </div>
      <div class="evidence-meta">
        <strong>${escapeHtml(item.circular || 'Government Resolution')}</strong>
        <span>${escapeHtml(item.department || 'Department unavailable')}</span>
        ${item.issue_date ? `<span>${escapeHtml(item.issue_date)}</span>` : ''}
      </div>
      <p class="evidence-text">${escapeHtml(item.text || 'Source text was not returned.')}</p>
      <div class="evidence-score">
        <span>Semantic score ${(Number(item.qdrant_score || 0)).toFixed(3)}</span>
        ${item.reranker_score !== null && item.reranker_score !== undefined ? `<span>Reranker ${(Number(item.reranker_score)).toFixed(3)}</span>` : ''}
      </div>
    </article>`).join('');
  evidenceSection.hidden = false;
}

async function ask() {
  const question = questionInput.value.trim();
  if (!question) {
    showError('Write a question first.');
    questionInput.focus();
    return;
  }
  errorBox.hidden = true;
  answerSection.hidden = true;
  evidenceSection.hidden = true;
  setBusy(true);
  try {
    const response = await fetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, ...CONFIG }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    renderAnswer(payload);
    renderEvidence(payload);
  } catch (error) {
    showError(error.message || 'The search could not be completed.');
  } finally {
    setBusy(false);
  }
}

askButton.addEventListener('click', ask);
questionInput.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') ask();
});
