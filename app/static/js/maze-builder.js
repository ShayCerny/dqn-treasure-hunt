let gridState = [];
let startCell = [0, 0];
let targetCell = null; // set on first grid init
let currentMode = 'wall';

function initGrid(defaultMaze) {
  gridState = defaultMaze.map(row => [...row]);
  const rows = gridState.length;
  const cols = gridState[0].length;
  if (!targetCell) targetCell = [rows - 1, cols - 1];
  renderGrid();
}

function renderGrid() {
  const container = document.getElementById('maze-grid');
  container.innerHTML = '';
  const rows = gridState.length;
  const cols = gridState[0].length;
  container.style.gridTemplateColumns = `repeat(${cols}, 40px)`;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = document.createElement('div');
      cell.className = 'maze-cell ' + (gridState[r][c] === 1 ? 'free' : 'wall');
      cell.dataset.r = r;
      cell.dataset.c = c;

      if (r === startCell[0] && c === startCell[1]) cell.classList.add('start');
      if (r === targetCell[0] && c === targetCell[1]) cell.classList.add('target');

      cell.addEventListener('click', () => handleCellClick(r, c));
      container.appendChild(cell);
    }
  }
}

function handleCellClick(r, c) {
  const rows = gridState.length;
  const cols = gridState[0].length;
  const isStart = r === startCell[0] && c === startCell[1];
  const isTarget = r === targetCell[0] && c === targetCell[1];

  if (currentMode === 'wall') {
    if (isStart || isTarget) return; // never wall out start/target
    gridState[r][c] = gridState[r][c] === 1 ? 0 : 1;
  } else if (currentMode === 'start') {
    if (isTarget) return; // can't place start on target
    if (gridState[r][c] !== 1) return; // can't place start on a wall
    startCell = [r, c];
  } else if (currentMode === 'treasure') {
    if (isStart) return; // can't place target on start
    if (gridState[r][c] !== 1) return; // can't place target on a wall
    targetCell = [r, c];
  }
  renderGrid();
}

function applySize() {
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const rows = clamp(parseInt(document.getElementById('maze-rows').value) || 8, 4, 32);
  const cols = clamp(parseInt(document.getElementById('maze-cols').value) || 8, 4, 32);
  document.getElementById('maze-rows').value = rows;
  document.getElementById('maze-cols').value = cols;
  startCell = [0, 0];
  targetCell = [rows - 1, cols - 1];
  initGrid(Array.from({ length: rows }, () => Array(cols).fill(1)));
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
}

async function saveMaze() {
  const name = document.getElementById('maze-name').value.trim() || 'My Maze';
  const statusEl = document.getElementById('save-status');
  try {
    const result = await apiFetch('/api/mazes/', {
      method: 'POST',
      body: JSON.stringify({ name, grid: gridState, start: startCell, target: targetCell }),
    });
    statusEl.textContent = `Saved! Maze ID: ${result.id}`;
    statusEl.className = 'text-success';
    loadSavedMazes();
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    statusEl.className = 'text-danger';
  }
}

async function loadSavedMazes() {
  const list = document.getElementById('saved-mazes-list');
  if (!list) return;
  try {
    const mazes = await apiFetch('/api/mazes/');
    list.innerHTML = mazes.length === 0
      ? '<li class="list-group-item text-muted">No saved mazes yet</li>'
      : mazes.map(m => `
          <li class="list-group-item d-flex justify-content-between align-items-center">
            <span>${m.name} <small class="text-muted">(${m.rows}×${m.cols})</small></span>
            <div>
              <button class="btn btn-sm btn-outline-primary me-1" onclick="loadMaze('${m.id}')">Load</button>
              <button class="btn btn-sm btn-outline-danger" onclick="deleteMaze('${m.id}')">Delete</button>
            </div>
          </li>`).join('');
  } catch (e) {
    list.innerHTML = `<li class="list-group-item text-danger">${e.message}</li>`;
  }
}

async function loadMaze(id) {
  try {
    const data = await apiFetch(`/api/mazes/${id}`);
    gridState = data.grid;
    const rows = gridState.length;
    const cols = gridState[0].length;
    startCell = data.start || [0, 0];
    targetCell = data.target || [rows - 1, cols - 1];
    renderGrid();
    document.getElementById('maze-name').value = data.name;
  } catch (e) {
    alert('Could not load maze: ' + e.message);
  }
}

async function deleteMaze(id) {
  if (!confirm('Delete this maze?')) return;
  try {
    await apiFetch(`/api/mazes/${id}`, { method: 'DELETE' });
    loadSavedMazes();
  } catch (e) {
    alert('Could not delete maze: ' + e.message);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadSavedMazes();
  setMode('wall');
});
