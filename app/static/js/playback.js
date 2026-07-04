let pbData = null;
let pbStep = 0;
let pbInterval = null;
let pbRunId = null;

async function openPlayback(runId) {
  pbRunId = runId;
  document.getElementById('pb-run-id').textContent = runId;
  document.getElementById('pb-outcome').textContent = '';
  document.getElementById('pb-step-label').textContent = 'Loading…';
  document.getElementById('pb-maze-grid').innerHTML = '';
  pbStop();

  const modal = new bootstrap.Modal(document.getElementById('playback-modal'));
  modal.show();

  await pbFetch();
}

async function pbFetch(start) {
  document.getElementById('pb-outcome').textContent = '';
  document.getElementById('pb-step-label').textContent = 'Loading…';
  document.getElementById('pb-play-btn').disabled = true;
  document.getElementById('pb-next-btn').disabled = true;

  try {
    const url = start
      ? `/api/runs/${pbRunId}/play?start=${start[0]},${start[1]}`
      : `/api/runs/${pbRunId}/play`;
    pbData = await apiFetch(url);
    pbStep = 0;
    document.getElementById('pb-play-btn').disabled = false;
    document.getElementById('pb-next-btn').disabled = false;
    pbRender();
    if (pbData.steps.length <= 1) {
      document.getElementById('pb-play-btn').disabled = true;
      document.getElementById('pb-next-btn').disabled = true;
      const outcomeEl = document.getElementById('pb-outcome');
      outcomeEl.textContent = 'No moves recorded for this run';
      outcomeEl.style.color = '#6c757d';
    }
  } catch (e) {
    document.getElementById('pb-step-label').textContent = 'Error: ' + e.message;
  }
}

function pbRender() {
  if (!pbData) return;
  const { grid, start, target, steps, outcome, bfs_path } = pbData;
  if (pbStep >= steps.length) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const agentPos = steps[pbStep];

  const bfsSet = new Set((bfs_path || []).map(p => `${p[0]},${p[1]}`));

  const maxGridPx = Math.min(window.innerWidth * 0.7, 600);
  const cellSize = Math.min(36, Math.floor(maxGridPx / Math.max(rows, cols)));

  const container = document.getElementById('pb-maze-grid');
  container.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
  container.style.display = 'inline-grid';
  container.style.gap = '2px';
  container.innerHTML = '';

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement('div');
      cell.style.cssText = `width:${cellSize}px;height:${cellSize}px;border-radius:4px;`;

      const isAgent = agentPos[0] === r && agentPos[1] === c;
      const isTarget = target[0] === r && target[1] === c;
      const isStart = start[0] === r && start[1] === c;
      const isWall = grid[r][c] === 0;
      const onBfs = bfsSet.has(`${r},${c}`);

      if (isAgent) {
        cell.style.background = '#20c997';
        cell.title = 'Agent';
      } else if (isTarget) {
        cell.style.background = '#ffc107';
        cell.title = 'Treasure';
      } else if (isStart) {
        cell.style.background = '#0d6efd';
        cell.title = 'Start';
      } else if (isWall) {
        cell.style.background = '#212529';
      } else if (onBfs) {
        cell.style.background = '#cfe2ff';
        cell.style.border = '1px solid #9ec5fe';
        cell.title = 'BFS optimal path';
      } else {
        cell.style.background = '#ffffff';
        cell.style.border = '1px solid #dee2e6';
      }

      if (!isWall && !isTarget) {
        cell.style.cursor = 'pointer';
        cell.addEventListener('click', () => {
          pbStop();
          pbFetch([r, c]);
        });
      }

      container.appendChild(cell);
    }
  }

  document.getElementById('pb-step-label').textContent = `Step ${pbStep} / ${steps.length - 1}`;

  const isLast = pbStep === steps.length - 1;
  if (isLast) {
    pbStop();
    const outcomeEl = document.getElementById('pb-outcome');
    outcomeEl.textContent = outcome === 'win' ? '🏆 Agent reached the treasure!' : '💀 Agent failed';
    outcomeEl.style.color = outcome === 'win' ? '#198754' : '#dc3545';
  }
}

function pbAdvance(delta) {
  if (!pbData) return;
  pbStep = Math.max(0, Math.min(pbData.steps.length - 1, pbStep + delta));
  pbRender();
}

function pbTogglePlay() {
  if (pbInterval) {
    pbStop();
  } else {
    pbStart();
  }
}

function pbRestart() {
  pbStop();
  pbStep = 0;
  document.getElementById('pb-outcome').textContent = '';
  pbRender();
}

function pbStart() {
  if (!pbData || pbStep >= pbData.steps.length - 1) pbStep = 0;
  const btn = document.getElementById('pb-play-btn');
  btn.textContent = '⏸ Pause';
  pbInterval = setInterval(() => {
    pbStep++;
    pbRender();
    if (pbStep >= pbData.steps.length - 1) pbStop();
  }, 300);
}

function pbStop() {
  if (pbInterval) {
    clearInterval(pbInterval);
    pbInterval = null;
  }
  const btn = document.getElementById('pb-play-btn');
  if (btn) btn.textContent = '▶ Play';
}

document.addEventListener('DOMContentLoaded', () => {
  const modalEl = document.getElementById('playback-modal');
  if (modalEl) modalEl.addEventListener('hidden.bs.modal', pbStop);
});
