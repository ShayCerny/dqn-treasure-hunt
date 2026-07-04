# DQN Treasure Hunt

A web application for training and visualizing a Deep Q-Network (DQN) agent solving custom maze environments in real time.

**[Live Demo](#)** — build a maze, train an agent, and watch it learn in your browser.

> The hosted demo runs on a free-tier container with an ephemeral filesystem: mazes, trained models, and run history reset if the instance restarts or sleeps from inactivity. Run it locally (below) for persistent storage.

<!-- Add a screenshot or GIF of the app here, e.g.: -->
<!-- ![Training dashboard](docs/screenshot-train.png) -->

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
python run.py
```

Then open your browser to: **http://localhost:5000**

---

## How to Use

1. **Build a Maze** — Go to **Maze Builder**, draw walls, set a start (blue) and treasure (yellow) cell, then save.
2. **Train the Agent** — Go to **Train**, select your maze, adjust hyperparameters if desired, and click **Start Training**. Live charts update in real time.
3. **Watch a Replay** — From **Run History** or **Recent Runs** on the home page, click **Watch** to see the trained agent navigate the maze.
4. **Review Results** — **Run History** shows all runs with final win rate and a completion check (did the agent win from every starting cell?).

---

## How It Works

The agent uses a **Deep Q-Network (DQN)** — a neural network that learns to estimate the value of each action (move left/right/up/down) in each state (position in the maze). Training follows the standard DQN algorithm:

- **Epsilon-greedy exploration**: the agent starts by acting randomly and gradually shifts to exploiting its learned Q-values.
- **Prioritized Experience Replay (PER)**: past transitions are stored in a sum-tree and sampled proportionally to their TD error, so surprising transitions are replayed more often. Importance-sampling weights correct the resulting sampling bias.
- **Target network**: a second, slower-updating copy of the network is used to compute stable Q-targets, preventing feedback loops.
- **Double DQN**: the main network selects the best next action and the target network evaluates it, decoupling action selection from evaluation to reduce Q-value overestimation.

Training ends when the maximum number of epochs is reached, when the agent achieves 100% win rate for `patience` consecutive epochs (early stop), or when manually cancelled.

---

## Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| **Epochs** | 1000 | Total training cycles. |
| **Batch Size** | 32 | Experiences sampled per network update. |
| **Replay Memory** | 512 | Max past experiences stored in the buffer. |
| **Learning Rate** | 0.001 | How fast network weights update (Adam optimizer). |
| **Hidden Layer Size** | 64 | Neurons per hidden layer in the Q-network. |
| **Epsilon Start** | 1.0 | Initial exploration rate (1.0 = fully random). |
| **Epsilon Min** | 0.05 | Minimum exploration rate floor. |
| **Epsilon Decay** | 0.995 | Multiplier applied to epsilon each epoch. |
| **Patience** | 10 | Consecutive 100% win-rate epochs before early stop. |
| **Early Stop Loss Delta** | 0.1 | Min relative loss improvement over patience window before early stop. |
| **PER Priority Exponent (α)** | 0.6 | Prioritization strength; 0 = uniform, 1 = fully prioritized. |
| **IS Correction Start (β)** | 0.4 | Importance-sampling correction start, annealed to 1.0 over training. |
| **Target Update Freq** | 50 | Epochs between target network weight syncs. |
| **Discount Factor** | 0.95 | How much future rewards are valued vs. immediate. |

Full explanations and tuning advice are available on the **Help** page inside the app.

---

## Algorithm Improvements

Two enhancements over standard DQN are included:

**Double DQN** addresses the overestimation bias in vanilla DQN, where the target network both selects and evaluates the best next action, inflating Q-values. DDQN decouples these roles: the main network selects, the target network evaluates. The result is more accurate value estimates and a more stable learned policy.

**Prioritized Experience Replay (PER)** replaces uniform random batch sampling with sampling proportional to each transition's TD error — the difference between the predicted and target Q-value. Transitions the network currently finds surprising are replayed more often, leading to faster convergence on difficult examples. A sum-tree enables O(log n) sampling and priority updates. Importance-sampling weights correct the non-uniform sampling bias, annealed from `beta_start` toward 1.0 over training.

---

## Project Structure

```
app/
├── __init__.py          # Flask app factory and SocketIO setup
├── config.py            # App config and TrainingConfig dataclass
├── api/
│   ├── mazes.py         # Maze CRUD endpoints
│   └── runs.py          # Training orchestration, playback, run history
├── ml/
│   ├── agent.py         # DQNAgent — model, training loop, evaluation
│   ├── game_experience.py  # Replay memory and batch sampling
│   └── treasure_maze.py    # Maze environment (states, rewards, actions)
├── routes/
│   ├── main.py          # Home and Help page routes
│   ├── mazes.py         # Maze Builder page route
│   └── training.py      # Train and Run History page routes
├── static/
│   ├── css/styles.css   # Custom styles
│   └── js/
│       ├── api-client.js    # Thin fetch wrapper
│       ├── dashboard.js     # Training page charts and Socket.IO
│       ├── maze-builder.js  # Interactive maze grid editor
│       └── playback.js      # Replay modal controls
└── templates/
    ├── base.html        # Shared layout, navbar, replay modal
    ├── index.html       # Home page with recent runs
    ├── train.html       # Training page with config and live charts
    ├── runs.html        # Full run history table
    ├── maze_builder.html   # Maze editor page
    └── help.html        # Configuration guide and DQN explanation
instance/                # SQLite database (mazes, run history)
models/                  # Trained model weight files (.weights.h5)
run.py                   # Entry point
requirements.txt         # Python dependencies
```

---

## Technology Stack

- **Backend**: Python, Flask, Flask-SocketIO (threading mode)
- **ML**: TensorFlow / Keras (DQN with target network)
- **Frontend**: Bootstrap 5, Chart.js, Socket.IO client
- **Storage**: SQLite (mazes, run history), HDF5 weight files (models)

---

## Deployment

The app is containerized (see `Dockerfile`) and configured via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECRET_KEY` | `dev-secret-key` | Flask session signing key. **Set a real value in any public deployment.** |
| `DATABASE_URL` | local SQLite file | Swap in a persistent database for non-ephemeral storage. |
| `PORT` | `5000` | Port the server listens on. |
| `FLASK_HOST` | `0.0.0.0` | Bind address. |
| `FLASK_DEBUG` | `true` | Disable in any public deployment. |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated list to restrict Socket.IO CORS in production. |

Build and run locally with Docker:

```bash
docker build -t dqn-treasure-hunt .
docker run -p 5000:5000 -e PORT=5000 -e FLASK_DEBUG=false -e SECRET_KEY=change-me dqn-treasure-hunt
```

The hosted demo linked above runs this same image on [Hugging Face Spaces](https://huggingface.co/spaces) (free CPU tier, Docker SDK).
