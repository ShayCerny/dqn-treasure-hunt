import os
from dataclasses import dataclass, asdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class DevelopmentConfig:
    # WARNING: SECRET_KEY falls back to a well-known dev value if the env var
    # is unset. Always set SECRET_KEY in the environment before any deployment
    # where sessions must be secure.
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f"sqlite:///{os.path.join(_PROJECT_ROOT, 'instance', 'treasure_hunt.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')


# Default maze used as the starting template in the maze builder UI.
# Stored here so it can be imported without pulling in any route module.
DEFAULT_MAZE = [
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1],
]


@dataclass
class TrainingConfig:
    n_epoch: int = 1000
    batch_size: int = 32
    max_memory: int = 512
    learning_rate: float = 0.001
    hidden_layer_size: int = 64
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    patience: int = 10        # early-stop window length (epochs)
    loss_min_delta: float = 0.1
    target_update_freq: int = 50  # epochs between target-network syncs
    discount: float = 0.95
    alpha: float = 0.6        # PER priority exponent: 0 = uniform, 1 = full priority
    beta_start: float = 0.4   # IS-weight initial value; annealed toward 1.0 during training

    def __post_init__(self):
        if not 1 <= self.n_epoch <= 50000:
            raise ValueError("n_epoch must be between 1 and 50000")
        if not 0.0 < self.epsilon <= 1.0:
            raise ValueError("epsilon must be in (0, 1]")
        if not 0.0 < self.epsilon_min < self.epsilon:
            raise ValueError("epsilon_min must be less than epsilon")
        if not 0.0 < self.epsilon_decay < 1.0:
            raise ValueError("epsilon_decay must be in (0, 1)")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if not 0.0 < self.loss_min_delta < 1.0:
            raise ValueError("loss_min_delta must be in (0, 1)")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.beta_start <= 1.0:
            raise ValueError("beta_start must be in [0, 1]")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 4 <= self.hidden_layer_size <= 1024:
            raise ValueError("hidden_layer_size must be between 4 and 1024")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_memory < self.batch_size:
            raise ValueError("max_memory must be >= batch_size")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.target_update_freq < 1:
            raise ValueError("target_update_freq must be at least 1")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
