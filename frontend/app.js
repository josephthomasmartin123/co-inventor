/**
 * Joe-Bot — Frontend
 * Three screens: input → progress → results
 *
 * SSE event types handled:
 *   status                — stage change with message + detail
 *   invention_generated   — new invention from any strategy
 *   search_query          — web search made by any agent (agent, query, results)
 *   proximity_complete    — dedup/cluster summary
 *   review_complete       — one invention evaluated (tier, scores)
 *   matchup_complete      — one Elo debate round
 *   evolution_complete    — evolved invention ready
 *   meta_review_complete  — research overview ready
 *   done                  — pipeline finished
 *   error                 — pipeline failed
 */

// ── State ─────────────────────────────────────────────────────────────────

let currentSessionId = null;
let eventSource = null;
let inventionCount = 0;
let searchCount = 0;
let lastResult = null;   // stored on results load; used by export functions

// ── Strategy definitions ──────────────────────────────────────────────────
// Each entry explains how that strategy works — shown as a popover on the card.

const STRATEGY_INFO = {
  literature_exploration: {
    label: 'Literature exploration',
    description: 'Searched recent patents and scientific papers, then identified recent cross-domain advances — "triggers" — that make new solutions newly possible. Each invention from this strategy is directly derived from one of those triggers.',
  },
  simulated_debate: {
    label: 'Simulated debate',
    description: 'Ran a structured self-play debate between three expert personas: a materials scientist, a process engineer, and a systems thinker. Each proposed an approach, critiqued the others, and refined their position. The invention comes from where their mechanisms genuinely interact — one removing a limitation another identified — rather than from bundling all three together.',
  },
  iterative_assumptions: {
    label: 'Iterative assumptions',
    description: 'Listed the implicit assumptions baked into the conventional approach to this problem — things practitioners take for granted — then inverted or challenged the most fertile ones. The invention follows naturally from questioning what everyone assumes must be true.',
  },
  direct: {
    label: 'Direct',
    description: 'Decomposed the problem down to its root physical or chemical cause, then proposed a mechanism that eliminates that root cause. Avoids treating symptoms; attacks the underlying phenomenon directly.',
  },
  analogical: {
    label: 'Analogical',
    description: 'Identified other technical domains that face the exact same underlying physics or chemistry challenge, then adapted their proven solution to this context. The source domain is named explicitly in the mechanism.',
  },
  enhanced: {
    label: 'Enhanced (evolved)',
    description: 'A refined version of a top-ranked invention from this round. The mechanism was deepened and made more specific, and weaknesses flagged in the evaluation were addressed. It was then reviewed and re-entered the tournament, so it only ranks above its parent by beating it.',
  },
  combined: {
    label: 'Combined (evolved)',
    description: 'A hybrid of two top-ranked inventions drawn from different mechanistic families — combining two variants of one approach yields nothing. It was only produced because the two mechanisms interact to give a technical effect neither parent has alone, and like any other idea it had to win its rank in the tournament.',
  },
};

// ── Boot ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const textarea = document.getElementById('problem-statement');
  const btnStart = document.getElementById('btn-start');
  const charCount = document.getElementById('char-count');

  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length;
    btnStart.disabled = textarea.value.trim().length < 20;
  });

  document.querySelectorAll('.example-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.dataset.problem;
      textarea.dispatchEvent(new Event('input'));
      textarea.focus();
    });
  });

  btnStart.addEventListener('click', () => {
    const p = textarea.value.trim();
    if (p.length >= 20) startSession(p);
  });

  textarea.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !btnStart.disabled) btnStart.click();
  });

  document.getElementById('btn-new').addEventListener('click', resetToInput);
  document.getElementById('btn-export-md').addEventListener('click', exportMarkdown);
  document.getElementById('btn-export-json').addEventListener('click', exportJSON);

  document.getElementById('btn-next-round').addEventListener('click', () => {
    const feedback = document.getElementById('round-feedback').value.trim();
    const problem = document.getElementById('results-problem-echo').textContent;
    if (currentSessionId && problem) startSession(problem, currentSessionId, feedback);
  });
});

// ── Screens ───────────────────────────────────────────────────────────────

function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(`screen-${name}`).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'instant' });
}

function resetToInput() {
  if (eventSource) { eventSource.close(); eventSource = null; }
  currentSessionId = null;
  inventionCount = 0;
  searchCount = 0;
  ['invention-stream', 'search-stream', 'activity-log'].forEach(id => {
    document.getElementById(id).innerHTML = '';
  });
  document.getElementById('invention-count').textContent = '0';
  document.getElementById('search-count').textContent = '0';
  document.getElementById('progress-status-text').textContent = 'Starting pipeline…';
  document.querySelectorAll('.stage-crumb').forEach(c => c.classList.remove('active', 'done'));
  showScreen('input');
}

// ── Session start ─────────────────────────────────────────────────────────

async function startSession(problem, parentSessionId = '', userFeedback = '') {
  showScreen('progress');
  document.getElementById('progress-problem-text').textContent = problem;

  // Reset progress counters
  inventionCount = 0; searchCount = 0;
  ['invention-stream', 'search-stream', 'activity-log'].forEach(id => {
    document.getElementById(id).innerHTML = '';
  });
  document.getElementById('invention-count').textContent = '0';
  document.getElementById('search-count').textContent = '0';
  document.querySelectorAll('.stage-crumb').forEach(c => c.classList.remove('active', 'done'));
  document.getElementById('progress-status-text').textContent = 'Starting pipeline…';

  try {
    const resp = await fetch('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        problem_statement: problem,
        parent_session_id: parentSessionId,
        user_feedback: userFeedback,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      setStatus(err.detail || 'Failed to start session', true); return;
    }
    const { session_id } = await resp.json();
    currentSessionId = session_id;
    connectSSE(session_id);
  } catch (err) {
    setStatus(`Network error: ${err.message}`, true);
  }
}

// ── SSE ───────────────────────────────────────────────────────────────────

const STAGE_ORDER = ['generating', 'proximity', 'reflecting', 'ranking', 'evolving', 'meta_reviewing'];

function connectSSE(sessionId) {
  eventSource = new EventSource(`/api/sessions/${sessionId}/stream`);

  eventSource.addEventListener('status', e => {
    const { data } = JSON.parse(e.data);
    activateStage(data.stage);
    setStatus(data.message || '', false, data.detail);
    addLog(`[${data.stage}] ${data.message || ''}`);
    if (data.detail) addLog(`  ${data.detail}`, 'dim');
  });

  eventSource.addEventListener('invention_generated', e => {
    const { data } = JSON.parse(e.data);
    inventionCount++;
    document.getElementById('invention-count').textContent = inventionCount;
    prependToStream('invention-stream', buildInvItem(data));
    addLog(`+ Invention (${data.strategy}): ${(data.title || '').substring(0, 60)}`);
  });

  eventSource.addEventListener('search_query', e => {
    const { data } = JSON.parse(e.data);
    searchCount++;
    document.getElementById('search-count').textContent = searchCount;
    prependToStream('search-stream', buildSearchItem(data));
    // Log with agent context
    const agentShort = (data.agent || '').split('/').pop();
    addLog(`  🔍 [${agentShort}] ${data.query}  →  ${data.result_count} results`, 'search');
  });

  eventSource.addEventListener('proximity_complete', e => {
    const { data } = JSON.parse(e.data);
    const removed = data.removed_count || 0;
    const msg = removed > 0
      ? `Proximity: ${data.total_in} → ${data.total_out} inventions (${removed} near-duplicate${removed !== 1 ? 's' : ''} removed, ${data.cluster_count} clusters)`
      : `Proximity: ${data.total_out} inventions, ${data.cluster_count} clusters — no duplicates found`;
    addLog(msg);
    if (data.clusters) {
      data.clusters.forEach(c => addLog(`  Cluster ${c.label}: ${c.theme}`, 'dim'));
    }
  });

  eventSource.addEventListener('review_complete', e => {
    const { data } = JSON.parse(e.data);
    if (!data.passed) {
      addLog(`  ✗ Filtered (${data.tier}): ${(data.invention_title || '').substring(0, 45)} — ${data.filter_reason || ''}`, 'dim');
    } else {
      addLog(`  ✓ Reviewed: ${(data.invention_title || '').substring(0, 45)}  [N:${data.novelty_score} Sci:${data.scientific_plausibility_score} Pat:${data.patentability_score} overall:${data.overall_score}]`);
    }
  });

  eventSource.addEventListener('matchup_complete', e => {
    const { data } = JSON.parse(e.data);
    addLog(`  ⚔ Round ${data.round}/${data.total_rounds}: ${(data.winner_title || '').substring(0, 50)}`, 'dim');
  });

  eventSource.addEventListener('evolution_complete', e => {
    const { data } = JSON.parse(e.data);
    inventionCount++;
    document.getElementById('invention-count').textContent = inventionCount;
    prependToStream('invention-stream', buildInvItem(data));
    addLog(`+ Evolved (${data.strategy}): ${(data.title || '').substring(0, 55)}`);
  });

  eventSource.addEventListener('meta_review_complete', e => {
    const { data } = JSON.parse(e.data);
    addLog(`Meta-review: ${(data.overview_preview || '').substring(0, 80)}…`);
  });

  eventSource.addEventListener('done', async e => {
    eventSource.close(); eventSource = null;
    markAllDone();
    setStatus('Complete — loading results…');
    await loadAndShowResults(sessionId);
  });

  eventSource.addEventListener('ping', () => {});

  eventSource.addEventListener('error', e => {
    if (e.data) {
      try { setStatus(JSON.parse(e.data).data?.message || 'Pipeline error', true); } catch {}
    }
    if (eventSource) { eventSource.close(); eventSource = null; }
  });
}

// ── Stage breadcrumb ──────────────────────────────────────────────────────

function activateStage(stageName) {
  const idx = STAGE_ORDER.indexOf(stageName);
  STAGE_ORDER.forEach((s, i) => {
    const el = document.querySelector(`.stage-crumb[data-stage="${s}"]`);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (i < idx) el.classList.add('done');
    else if (i === idx) el.classList.add('active');
  });
}

function markAllDone() {
  STAGE_ORDER.forEach(s => {
    const el = document.querySelector(`.stage-crumb[data-stage="${s}"]`);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  });
}

// ── Progress helpers ──────────────────────────────────────────────────────

function setStatus(msg, isError = false, detail = '') {
  const text = document.getElementById('progress-status-text');
  text.textContent = msg;
  text.className = isError ? 'error-msg' : '';
  const spinner = document.querySelector('#progress-status .spinner');
  if (spinner) spinner.style.display = isError ? 'none' : '';
}

function addLog(msg, type = 'info') {
  const log = document.getElementById('activity-log');
  const line = document.createElement('div');
  line.className = `log-line${type === 'search' ? ' search' : type === 'dim' ? ' dim' : type === 'err' ? ' err' : ''}`;
  const t = new Date().toLocaleTimeString('en-GB', { hour12: false });
  line.textContent = `${t}  ${msg}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

function prependToStream(containerId, html) {
  const el = document.createElement('div');
  el.innerHTML = html;
  const child = el.firstElementChild;
  const container = document.getElementById(containerId);
  container.insertBefore(child, container.firstChild);
}

function buildInvItem(data) {
  const s = data.strategy || 'direct';
  return `<div class="inv-item">
    <div class="strat-pill strat-${esc(s)}">${strategyLabel(s)}</div>
    <div class="inv-title-text">${esc((data.title || '').substring(0, 75))}</div>
  </div>`;
}

function buildSearchItem(data) {
  const agent = (data.agent || '').replace('/', ' › ');
  const n = data.result_count || 0;
  return `<div class="search-item">
    <div class="search-agent">${esc(agent)}</div>
    <div class="search-query">${esc(data.query || '')}</div>
    <div class="search-result-count">${n} result${n !== 1 ? 's' : ''} found</div>
  </div>`;
}

// ── Results ───────────────────────────────────────────────────────────────

async function loadAndShowResults(sessionId) {
  try {
    const resp = await fetch(`/api/sessions/${sessionId}`);
    const result = await resp.json();
    renderResults(result);
    showScreen('results');
  } catch (err) {
    setStatus(`Failed to load results: ${err.message}`, true);
  }
}

function renderResults(result) {
  lastResult = result;
  const { session, inventions, reviews, meta_review } = result;
  document.getElementById('results-problem-echo').textContent = session.problem_statement;

  const roundNum = session.round_number || 1;
  const meta = document.getElementById('results-meta-row');
  meta.innerHTML = `
    ${roundNum > 1 ? `<div class="round-badge">Round ${roundNum}</div>` : ''}
    <div class="meta-stat"><strong>${inventions.length}</strong> inventions ranked</div>
    <div class="meta-stat"><strong>${searchCount}</strong> searches made</div>
    ${session.parent_session_id ? `<div class="meta-stat">Building on <a href="?session=${session.parent_session_id}" style="color:var(--accent)">Round ${roundNum - 1}</a></div>` : ''}
  `;

  renderMetaReview(meta_review || {});
  renderInventions(inventions, reviews);

  // Set up next round button
  document.getElementById('next-round-num').textContent = roundNum + 1;
  document.getElementById('round-feedback').value = '';

  document.getElementById('results-session-id').textContent = `Session ${session.id.substring(0, 8)}`;
}

function renderMetaReview(mr) {
  const block = document.getElementById('meta-review-block');
  const body = document.getElementById('meta-review-body');
  if (!mr.overview && !mr.recommendation) { block.style.display = 'none'; return; }

  block.style.display = '';
  const sections = [];

  if (mr.overview) {
    sections.push(`<div class="meta-section">
      <div class="meta-section-label">Overview</div>
      <div class="meta-section-text">${esc(mr.overview)}</div>
    </div>`);
  }

  const lists = [
    { key: 'strongest_approaches', label: 'Strongest approaches' },
    { key: 'recurring_challenges', label: 'Recurring challenges' },
    { key: 'unexplored_directions', label: 'Unexplored directions' },
  ];
  lists.forEach(({ key, label }) => {
    const items = mr[key];
    if (!items || !items.length) return;
    const lis = items.map(t => `<li>${esc(t)}</li>`).join('');
    sections.push(`<div class="meta-section">
      <div class="meta-section-label">${label}</div>
      <ul class="meta-list">${lis}</ul>
    </div>`);
  });

  if (mr.cross_domain_insight) {
    sections.push(`<div class="meta-section">
      <div class="meta-section-label">Cross-domain insight</div>
      <div class="meta-section-text">${esc(mr.cross_domain_insight)}</div>
    </div>`);
  }

  if (mr.recommendation) {
    sections.push(`<div class="meta-recommendation">
      <div class="meta-section-label">Recommendation</div>
      <div class="meta-section-text">${esc(mr.recommendation)}</div>
    </div>`);
  }

  body.innerHTML = sections.join('');
}

function renderInventions(inventions, reviews) {
  const list = document.getElementById('invention-list');
  list.innerHTML = '';
  inventions.forEach((inv, idx) => {
    list.appendChild(buildInventionCard(inv, reviews[inv.id] || null, idx));
  });
}

function buildInventionCard(inv, review, idx) {
  const card = document.createElement('div');
  card.className = `inv-card${idx === 0 ? ' rank-1' : ''}`;

  const rankLabel = idx === 0 ? '#1' : idx === 1 ? '#2' : idx === 2 ? '#3' : `#${idx + 1}`;
  const s = inv.strategy || 'direct';
  const eloScore = Math.round(inv.elo_score || 1200);
  const info = STRATEGY_INFO[s] || { label: s, description: '' };

  // Trigger block — only for literature_exploration inventions that have a trigger
  const hasTrigger = s === 'literature_exploration' && inv.trigger_advance;
  const triggerHtml = hasTrigger ? `
    <div class="trigger-block">
      <div class="trigger-block-icon">💡</div>
      <div class="trigger-block-body">
        <div class="trigger-block-label">Triggered by</div>
        <div class="trigger-block-advance">${esc(inv.trigger_advance)}</div>
        <div class="trigger-block-meta">
          ${inv.trigger_source_domain ? `<span class="trigger-block-domain">${esc(inv.trigger_source_domain)}</span>` : ''}
          ${inv.trigger_url ? `<span class="trigger-block-url"><a href="${esc(inv.trigger_url)}" target="_blank" rel="noopener">source ↗</a></span>` : ''}
        </div>
      </div>
    </div>` : '';

  const scoreStrip = review ? buildScoreStrip(review) : '';

  const sections = [];
  if (inv.mechanism) {
    sections.push({ title: 'Mechanism', body: `<p>${esc(inv.mechanism)}</p>` });
  }
  if (review) {
    sections.push({ title: 'Evaluation', body: buildEvalBody(review) });
  }
  if (inv.search_evidence && inv.search_evidence.length) {
    sections.push({
      title: `References (${inv.search_evidence.length})`,
      body: `<div>${inv.search_evidence.slice(0, 5)
        .map(u => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>`)
        .join('')}</div>`,
    });
  }

  const sectionsHtml = sections.map(sec => `
    <div class="inv-section">
      <button class="inv-section-toggle" onclick="toggleSection(this)">
        ${esc(sec.title)}
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M2 4L6 8L10 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <div class="inv-section-body">${sec.body}</div>
    </div>`).join('');

  // Unique popover ID per card
  const popoverId = `pop-${inv.id.substring(0, 8)}`;

  card.innerHTML = `
    <div class="inv-card-header">
      <div class="inv-card-top">
        <div class="rank-num${idx === 0 ? ' top' : ''}">${rankLabel}</div>
        <div class="inv-card-meta">
          <div class="strategy-label" style="position:relative">
            <span class="strat-dot dot-${esc(s)}"></span>
            ${info.label}
            <button class="strategy-info-btn" onclick="toggleStrategyPopover('${popoverId}')" title="How this strategy works">?</button>
            <div class="strategy-popover" id="${popoverId}">
              <div class="strategy-popover-name">${esc(info.label)}</div>
              <div class="strategy-popover-desc">${esc(info.description)}</div>
            </div>
          </div>
          <div class="inv-card-title">${esc(inv.title)}</div>
        </div>
        <div class="elo-chip">Elo ${eloScore}</div>
      </div>
      ${inv.summary ? `<p class="inv-summary">${esc(inv.summary)}</p>` : ''}
      ${triggerHtml}
    </div>
    ${scoreStrip}
    ${sectionsHtml ? `<div class="inv-sections">${sectionsHtml}</div>` : ''}
  `;
  return card;
}

function buildScoreStrip(review) {
  const scores = [
    { label: 'Novelty',     val: review.novelty_score },
    { label: 'Plausibility',val: review.scientific_plausibility_score },
    { label: 'Patent.',     val: review.patentability_score },
    { label: 'Feasibility', val: review.feasibility_score },
    { label: 'Fit',         val: review.problem_fit_score },
  ];
  const cells = scores.map(({ label, val }) => {
    const cls = val >= 4 ? 'good' : val >= 3 ? 'mid' : 'poor';
    const pct = (val / 5 * 100).toFixed(0);
    return `<div class="score-cell">
      <div class="score-cell-label">${label}</div>
      <div class="score-cell-val score-${cls}">${val}/5</div>
      <div class="score-bar"><div class="score-bar-fill fill-${cls}" style="width:${pct}%"></div></div>
    </div>`;
  }).join('');
  return `<div class="score-strip">${cells}</div>`;
}

function buildEvalBody(review) {
  const rows = [
    { label: 'Novelty',                val: review.novelty_score,                    rationale: review.novelty_rationale },
    { label: 'Scientific plausibility',val: review.scientific_plausibility_score,    rationale: review.scientific_plausibility_rationale },
    { label: 'Patentability',          val: review.patentability_score,              rationale: review.patentability_rationale },
    { label: 'Feasibility',            val: review.feasibility_score,                rationale: review.feasibility_rationale },
    { label: 'Problem fit',            val: review.problem_fit_score,                rationale: review.problem_fit_rationale },
  ];
  const rowsHtml = rows.map(({ label, val, rationale }) => {
    const cls = val >= 4 ? 'good' : val >= 3 ? 'mid' : 'poor';
    const pct = (val / 5 * 100).toFixed(0);
    return `<div class="eval-row">
      <div class="eval-row-head">
        <span class="eval-row-label">${label}</span>
        <span class="eval-row-score score-${cls}">${val}/5</span>
        <div class="eval-row-bar"><div class="eval-row-bar-fill fill-${cls}" style="width:${pct}%"></div></div>
      </div>
      ${rationale ? `<div class="eval-row-rationale">${esc(rationale)}</div>` : ''}
    </div>`;
  }).join('');

  const prior = (review.prior_art_found || []).slice(0, 4);
  const priorHtml = prior.length ? `
    <div class="prior-art">
      <div class="prior-art-label">Prior art found</div>
      ${prior.map(u => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>`).join('')}
    </div>` : '';

  return `<div class="eval-rows">${rowsHtml}</div>${priorHtml}`;
}

// ── Accordion ─────────────────────────────────────────────────────────────
function toggleSection(btn) { btn.closest('.inv-section').classList.toggle('open'); }

// ── Strategy info popover ─────────────────────────────────────────────────
function toggleStrategyPopover(id) {
  const popover = document.getElementById(id);
  if (!popover) return;
  const isOpen = popover.classList.contains('open');
  // Close all open popovers first
  document.querySelectorAll('.strategy-popover.open').forEach(p => p.classList.remove('open'));
  if (!isOpen) popover.classList.add('open');
}

// Close popovers when clicking outside
document.addEventListener('click', e => {
  if (!e.target.closest('.strategy-label')) {
    document.querySelectorAll('.strategy-popover.open').forEach(p => p.classList.remove('open'));
  }
});

// ── Utilities ─────────────────────────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function strategyLabel(s) {
  return (STRATEGY_INFO[s] && STRATEGY_INFO[s].label) || s;
}

// ── Export ────────────────────────────────────────────────────────────────

function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportFilename(session, ext) {
  const words = (session.problem_statement || 'session')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    .split('-').slice(0, 5).join('-');
  const round = session.round_number || 1;
  return `joe-bot-${words}-round${round}.${ext}`;
}

function exportMarkdown() {
  if (!lastResult) return;
  const { session, inventions, reviews, meta_review: mr } = lastResult;
  const date = new Date().toLocaleDateString('en-GB', { year: 'numeric', month: 'long', day: 'numeric' });
  const round = session.round_number || 1;

  const lines = [];
  lines.push(`# Invention Session — ${session.problem_statement}`);
  lines.push(`*Joe-Bot · ${date} · Round ${round}*`);
  lines.push('');
  lines.push('---');
  lines.push('');

  // Meta-review
  if (mr && (mr.overview || mr.recommendation)) {
    lines.push('## Research Overview');
    lines.push('');
    if (mr.overview) { lines.push(mr.overview); lines.push(''); }

    if (mr.strongest_approaches && mr.strongest_approaches.length) {
      lines.push('### Strongest approaches');
      mr.strongest_approaches.forEach(t => lines.push(`- ${t}`));
      lines.push('');
    }
    if (mr.recurring_challenges && mr.recurring_challenges.length) {
      lines.push('### Recurring challenges');
      mr.recurring_challenges.forEach(t => lines.push(`- ${t}`));
      lines.push('');
    }
    if (mr.unexplored_directions && mr.unexplored_directions.length) {
      lines.push('### Unexplored directions');
      mr.unexplored_directions.forEach(t => lines.push(`- ${t}`));
      lines.push('');
    }
    if (mr.cross_domain_insight) {
      lines.push('### Cross-domain insight');
      lines.push(mr.cross_domain_insight);
      lines.push('');
    }
    if (mr.recommendation) {
      lines.push(`> **Recommendation:** ${mr.recommendation}`);
      lines.push('');
    }
    lines.push('---');
    lines.push('');
  }

  lines.push('## Invention Ideas');
  lines.push('');

  inventions.forEach((inv, idx) => {
    const review = reviews[inv.id] || null;
    const elo = Math.round(inv.elo_score || 1200);
    lines.push(`### #${idx + 1} — ${inv.title}`);
    lines.push(`**Strategy:** ${strategyLabel(inv.strategy || 'direct')} · **Elo score:** ${elo}`);
    lines.push('');
    if (inv.summary) { lines.push(inv.summary); lines.push(''); }

    if (inv.mechanism) {
      lines.push('**Mechanism**');
      lines.push('');
      lines.push(inv.mechanism);
      lines.push('');
    }

    if (review) {
      lines.push('**Evaluation**');
      lines.push('');
      lines.push('| Dimension | Score | Notes |');
      lines.push('|---|---|---|');
      lines.push(`| Novelty | ${review.novelty_score}/5 | ${(review.novelty_rationale || '').replace(/\|/g, '\\|')} |`);
      lines.push(`| Scientific plausibility | ${review.scientific_plausibility_score}/5 | ${(review.scientific_plausibility_rationale || '').replace(/\|/g, '\\|')} |`);
      lines.push(`| Patentability | ${review.patentability_score}/5 | ${(review.patentability_rationale || '').replace(/\|/g, '\\|')} |`);
      lines.push(`| Feasibility | ${review.feasibility_score}/5 | ${(review.feasibility_rationale || '').replace(/\|/g, '\\|')} |`);
      lines.push(`| Problem fit | ${review.problem_fit_score}/5 | ${(review.problem_fit_rationale || '').replace(/\|/g, '\\|')} |`);
      lines.push('');
      const prior = (review.prior_art_found || []).slice(0, 4);
      if (prior.length) {
        lines.push('**Prior art found**');
        prior.forEach(u => lines.push(`- ${u}`));
        lines.push('');
      }
    }

    lines.push('---');
    lines.push('');
  });

  lines.push(`*Session ${session.id} · Generated by Joe-Bot using the Co-Scientist architecture (Google DeepMind, 2025)*`);

  downloadFile(lines.join('\n'), exportFilename(session, 'md'), 'text/markdown;charset=utf-8');
}

function exportJSON() {
  if (!lastResult) return;
  const content = JSON.stringify(lastResult, null, 2);
  downloadFile(content, exportFilename(lastResult.session, 'json'), 'application/json;charset=utf-8');
}
