let lossChart, winChart, epsilonChart;
let currentRunId = null;

const CHART_WINDOW = 200;
const HISTORY_MAX_POINTS = 500;
const EPOCH_LOG_PAGE_SIZE = 50;

let epochLogData = [];
let epochLogCurrentPage = 0;

function initCharts() {
	const chartOpts = (label, color, extraY = {}) => ({
		type: "line",
		data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.3, pointRadius: 0 }] },
		options: {
			animation: false,
			maintainAspectRatio: false,
			scales: {
				x: {
					display: true,
					ticks: { maxTicksLimit: 8, color: "#6c757d", font: { size: 10 } },
					grid: { display: false },
				},
				y: { ...extraY },
			},
			plugins: { legend: { display: false } },
		},
	});

	lossChart = new Chart(document.getElementById("loss-chart"), chartOpts("Loss", "#dc3545"));
	winChart = new Chart(
		document.getElementById("win-chart"),
		chartOpts("Win Rate", "#198754", {
			min: 0,
			max: 1,
			ticks: { callback: (v) => (v * 100).toFixed(0) + "%" },
		}),
	);
	epsilonChart = new Chart(document.getElementById("epsilon-chart"), chartOpts("Epsilon", "#0d6efd"));
}

function resetCharts() {
	[lossChart, winChart, epsilonChart].forEach((c) => {
		c.data.labels = [];
		c.data.datasets[0].data = [];
		c.update("none");
	});
	document.getElementById("progress-bar").style.width = "0%";
	document.getElementById("progress-bar").textContent = "0%";
	document.getElementById("training-status").textContent = "";
	document.getElementById("epoch-counter").textContent = "Epoch: —";
}

function _slideWindow(chart, label, value) {
	chart.data.labels.push(label);
	chart.data.datasets[0].data.push(value);
	if (chart.data.labels.length > CHART_WINDOW) {
		chart.data.labels.shift();
		chart.data.datasets[0].data.shift();
	}
}

function loadFullHistory(epochs) {
	const step = Math.max(1, Math.ceil(epochs.length / HISTORY_MAX_POINTS));
	const sampled = epochs.filter((_, i) => i % step === 0 || i === epochs.length - 1);

	lossChart.data.labels = sampled.map((e) => String(e.epoch));
	winChart.data.labels = sampled.map((e) => String(e.epoch));
	epsilonChart.data.labels = sampled.map((e) => String(e.epoch));

	lossChart.data.datasets[0].data = sampled.map((e) => e.loss);
	winChart.data.datasets[0].data = sampled.map((e) => e.win_rate);
	epsilonChart.data.datasets[0].data = sampled.map((e) => e.epsilon);

	lossChart.update("none");
	winChart.update("none");
	epsilonChart.update("none");
}

function injectRunHighlights(epochs) {
	const el = document.getElementById("run-highlights");
	if (!el || epochs.length === 0) return;

	const lowestLoss = epochs.reduce((a, b) => (b.loss < a.loss ? b : a));
	const first90 = epochs.find((e) => e.win_rate >= 0.9);
	const first100 = epochs.find((e) => e.win_rate >= 1.0);
	const tail = epochs.slice(-10);
	const row = (label, value) => `<p class="mb-2"><strong>${label}:</strong> ${value}</p>`;

	let html =
		'<hr class="my-2"><p class="small text-muted fw-semibold mb-2 text-uppercase" style="letter-spacing:.05em">Highlights</p>';
	html += row(
		"Lowest Loss",
		`${lowestLoss.loss.toFixed(4)} <span class="text-muted small">(epoch ${lowestLoss.epoch})</span>`,
	);
	if (first90)
		html += row(
			"Exploitation Began",
			`epoch ${first90.epoch} <span class="text-muted small">(≥90% win rate)</span>`,
		);
	if (first100) html += row("First 100% Win Rate", `epoch ${first100.epoch}`);

	el.innerHTML = html;
}

function renderEpochLog(page) {
	const tbody = document.getElementById("epoch-log-body");
	const info = document.getElementById("epoch-log-page-info");
	const range = document.getElementById("epoch-log-range");
	const prevBtn = document.getElementById("epoch-log-prev");
	const nextBtn = document.getElementById("epoch-log-next");
	if (!tbody) return;

	const totalPages = Math.ceil(epochLogData.length / EPOCH_LOG_PAGE_SIZE);
	epochLogCurrentPage = Math.max(0, Math.min(page, totalPages - 1));

	const start = epochLogCurrentPage * EPOCH_LOG_PAGE_SIZE;
	const end = Math.min(start + EPOCH_LOG_PAGE_SIZE, epochLogData.length);
	const visible = epochLogData.slice(start, end);

	tbody.innerHTML = visible
		.map(
			(e) => `
    <tr>
      <td>${e.epoch}</td>
      <td>${e.loss.toFixed(4)}</td>
      <td>${(e.win_rate * 100).toFixed(1)}%</td>
      <td>${e.epsilon.toFixed(4)}</td>
      <td>${e.n_episodes}</td>
    </tr>
  `,
		)
		.join("");

	info.textContent = `Page ${epochLogCurrentPage + 1} of ${totalPages} — ${epochLogData.length} epochs total`;
	range.textContent = `Epochs ${visible[0].epoch}–${visible[visible.length - 1].epoch}`;
	prevBtn.disabled = epochLogCurrentPage === 0;
	nextBtn.disabled = epochLogCurrentPage >= totalPages - 1;
}

function epochLogPage(delta) {
	renderEpochLog(epochLogCurrentPage + delta);
}

function pushEpoch(data) {
	const label = String(data.epoch);
	_slideWindow(lossChart, label, data.loss);
	_slideWindow(winChart, label, data.win_rate);
	_slideWindow(epsilonChart, label, data.epsilon);

	// Only re-render every 10 epochs to keep the UI responsive
	if (data.epoch % 10 === 0) {
		lossChart.update("none");
		winChart.update("none");
		epsilonChart.update("none");
	}

	const pct = data.progress + "%";
	document.getElementById("progress-bar").style.width = pct;
	document.getElementById("progress-bar").textContent = pct;
	document.getElementById("epoch-counter").textContent = `Epoch: ${data.epoch}`;
}

function setBtn(mode) {
	const btn = document.getElementById("start-btn");
	if (!btn) return;
	// In watch mode, hide the button once training completes/errors.
	if (typeof WATCH_RUN_ID !== "undefined") {
		if (mode === "start" || mode === "playback") btn.style.display = "none";
		return;
	}
	btn.disabled = false;
	btn.classList.remove("btn-success", "btn-danger", "btn-info");
	if (mode === "start") {
		btn.textContent = "Start Training";
		btn.classList.add("btn-success");
		btn.onclick = startTraining;
	} else if (mode === "cancel") {
		btn.textContent = "Cancel Training";
		btn.classList.add("btn-danger");
		btn.onclick = cancelTraining;
	} else if (mode === "playback") {
		btn.textContent = "Watch Playback";
		btn.classList.add("btn-info");
		btn.onclick = () => openPlayback(currentRunId);
	}
}

async function loadRunHistory(runId) {
	const statusEl = document.getElementById("training-status");
	try {
		const epochs = await apiFetch(`/api/runs/${runId}/epochs`);
		const isCompleted = typeof WATCH_RUN_STATUS !== "undefined" && WATCH_RUN_STATUS !== "running";

		if (isCompleted && epochs.length > 0) {
			loadFullHistory(epochs);
			injectRunHighlights(epochs);
			epochLogData = epochs;
			const logSection = document.getElementById("epoch-log-section");
			if (logSection) {
				logSection.style.display = "";
				renderEpochLog(0);
			}
		} else {
			epochs.forEach((e) => {
				_slideWindow(lossChart, String(e.epoch), e.loss);
				_slideWindow(winChart, String(e.epoch), e.win_rate);
				_slideWindow(epsilonChart, String(e.epoch), e.epsilon);
			});
			lossChart.update("none");
			winChart.update("none");
			epsilonChart.update("none");
		}

		if (epochs.length > 0) {
			const last = epochs[epochs.length - 1];
			const pct = last.progress + "%";
			document.getElementById("progress-bar").style.width = pct;
			document.getElementById("progress-bar").textContent = pct;
			document.getElementById("epoch-counter").textContent = `Epoch: ${last.epoch}`;
		}
		if (WATCH_RUN_STATUS !== "running" && statusEl) {
			statusEl.textContent = `Run ${runId} — ${WATCH_RUN_STATUS}`;
		}
	} catch (e) {
		if (statusEl) statusEl.textContent = "Error loading history: " + e.message;
	}
}

async function startTraining() {
	const mazeId = document.getElementById("maze-select").value;
	const statusEl = document.getElementById("training-status");

	if (!mazeId) {
		alert("Please select a maze first.");
		return;
	}

	const config = {
		n_epoch: parseInt(document.getElementById("n_epoch").value),
		batch_size: parseInt(document.getElementById("batch_size").value),
		max_memory: parseInt(document.getElementById("max_memory").value),
		learning_rate: parseFloat(document.getElementById("learning_rate").value),
		hidden_layer_size: parseInt(document.getElementById("hidden_layer_size").value),
		epsilon: parseFloat(document.getElementById("epsilon").value),
		epsilon_min: parseFloat(document.getElementById("epsilon_min").value),
		epsilon_decay: parseFloat(document.getElementById("epsilon_decay").value),
		patience: parseInt(document.getElementById("patience").value),
		loss_min_delta: parseFloat(document.getElementById("loss_min_delta").value),
		alpha: parseFloat(document.getElementById("alpha").value),
		beta_start: parseFloat(document.getElementById("beta_start").value),
		target_update_freq: parseInt(document.getElementById("target_update_freq").value),
		discount: parseFloat(document.getElementById("discount").value),
	};

	// Collapse the config panel now that training is locked in
	const configPanel = document.getElementById("config-panel");
	if (configPanel) {
		bootstrap.Collapse.getOrCreateInstance(configPanel).hide();
	}

	setBtn("cancel");
	resetCharts();
	statusEl.textContent = "Starting training...";

	try {
		const result = await apiFetch("/api/runs/", {
			method: "POST",
			body: JSON.stringify({ maze_id: mazeId, config }),
		});
		currentRunId = result.run_id;
		statusEl.textContent = `Training run ${currentRunId} started`;
	} catch (e) {
		statusEl.textContent = "Error: " + e.message;
		setBtn("start");
		// Re-expand config panel on failure so user can adjust settings
		if (configPanel) {
			bootstrap.Collapse.getOrCreateInstance(configPanel).show();
		}
	}
}

async function cancelTraining() {
	if (!currentRunId) return;
	const btn = document.getElementById("start-btn");
	btn.disabled = true;
	btn.textContent = "Cancelling…";
	try {
		await apiFetch(`/api/runs/${currentRunId}/cancel`, { method: "POST" });
	} catch (e) {
		document.getElementById("training-status").textContent = "Cancel failed: " + e.message;
		setBtn("cancel");
	}
}

document.addEventListener("DOMContentLoaded", () => {
	initCharts();

	// Watch mode: connect to an existing run instead of starting a new one.
	if (typeof WATCH_RUN_ID !== "undefined") {
		currentRunId = WATCH_RUN_ID;
		loadRunHistory(WATCH_RUN_ID);
	}

	const socket = io();

	socket.on("epoch_update", (data) => {
		if (data.run_id !== currentRunId) return;
		pushEpoch(data);
	});

	socket.on("training_complete", (data) => {
		if (data.run_id !== currentRunId) return;
		lossChart.update("none");
		winChart.update("none");
		epsilonChart.update("none");

		const statusEl = document.getElementById("training-status");
		const winPct = (data.final_win_rate * 100).toFixed(1);

		const stopMessages = {
			early_stop: "⚡ Stopped early — 100% win rate sustained for 10 consecutive epochs",
			cancelled: "⛔ Training cancelled",
			complete: "✅ Training complete",
		};
		const stopLine = stopMessages[data.stop_reason] || "✅ Training complete";
		const checkLine =
			data.completion_check != null
				? `Completion check: <strong>${data.completion_check ? "PASSED" : "FAILED"}</strong>`
				: "";

		statusEl.innerHTML = `
      ${stopLine} — ${data.epochs_run} epochs<br>
      Final win rate: ${winPct}%<br>
      ${checkLine}
    `;

		setBtn("playback");
		const bar = document.getElementById("progress-bar");
		bar.style.width = "100%";
		bar.textContent = "100%";
	});

	socket.on("training_error", (data) => {
		if (data.run_id !== currentRunId) return;
		document.getElementById("training-status").textContent = "Error: " + data.error;
		setBtn("start");
	});
});
