"""
training/trainer.py — variational parameter optimisation.
Uses the parameter shift rule for analytic gradients.
Always trains on the local simulator — fast, free, reproducible.
"""
from __future__ import annotations
import json, logging, math, os, random, time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from config.settings import settings
from core.circuit_factory import build_variational_bottleneck
from core.executor import QuantumExecutor
from core.job_store import JobStore
from core.quantum_backend import BackendManager, BackendMode
from training.loss import expectation_value

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    n_qubits: int              = 2
    learning_rate: float       = settings.learning_rate
    max_iterations: int        = settings.max_training_iterations
    shots: int                 = settings.default_shots
    convergence_threshold: float = 1e-4
    checkpoint_dir: str        = settings.checkpoint_dir
    save_every: int            = 10


@dataclass
class TrainingHistory:
    losses: List[float]           = field(default_factory=list)
    best_loss: float              = float("inf")
    best_thetas: List[float]      = field(default_factory=list)
    iterations_run: int           = 0
    converged: bool               = False
    elapsed_seconds: float        = 0.0

    def to_dict(self): return asdict(self)


class VariationalTrainer:

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg = config or TrainingConfig()
        mgr      = BackendManager()
        store    = JobStore()
        self._ex = QuantumExecutor(mgr, store)

    def train(self, input_data: List[float], initial_thetas: Optional[List[float]] = None) -> TrainingHistory:
        cfg = self.cfg
        n   = cfg.n_qubits
        if len(input_data) != n:
            raise ValueError(f"input_data length {len(input_data)} != n_qubits {n}")

        thetas  = initial_thetas or [random.uniform(0, 2*math.pi) for _ in range(n)]
        history = TrainingHistory(best_thetas=thetas[:])
        start   = time.time()
        logger.info("Training: %dq, max %d iters, lr=%.4f", n, cfg.max_iterations, cfg.learning_rate)

        for i in range(cfg.max_iterations):
            loss = self._loss(input_data, thetas)
            history.losses.append(loss)
            if loss < history.best_loss:
                history.best_loss   = loss
                history.best_thetas = thetas[:]
            if i % 10 == 0:
                logger.info("iter %3d | loss=%.6f", i, loss)
            if i % cfg.save_every == 0:
                self._checkpoint(thetas, loss, i)
            # Convergence
            if len(history.losses) >= 5 and max(history.losses[-5:]) - min(history.losses[-5:]) < cfg.convergence_threshold:
                logger.info("Converged at iter %d", i)
                history.converged = True
                break
            # Parameter shift gradient descent
            grads  = self._gradients(input_data, thetas)
            thetas = [t - cfg.learning_rate * g for t, g in zip(thetas, grads)]

        history.iterations_run   = len(history.losses)
        history.elapsed_seconds  = round(time.time() - start, 2)
        self._checkpoint(history.best_thetas, history.best_loss, "final")
        logger.info("Training done: %d iters, best loss=%.6f, %.1fs", history.iterations_run, history.best_loss, history.elapsed_seconds)
        return history

    def _gradients(self, input_data, thetas):
        shift = math.pi / 2
        grads = []
        for i in range(len(thetas)):
            tp = thetas[:]; tp[i] += shift
            tm = thetas[:]; tm[i] -= shift
            grads.append((self._loss(input_data, tp) - self._loss(input_data, tm)) / 2.0)
        return grads

    def _loss(self, input_data, thetas) -> float:
        job = self._ex.run(
            build_variational_bottleneck(self.cfg.n_qubits, input_data, thetas),
            mode=BackendMode.SIMULATOR, shots=self.cfg.shots,
        )
        if not job.result:
            return float("inf")
        return 1.0 - expectation_value(job.result.get("counts", {}))

    def _checkpoint(self, thetas, loss, step) -> None:
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_{step}.json")
        with open(path, "w") as f:
            json.dump({"thetas": thetas, "loss": loss, "step": step,
                       "n_qubits": self.cfg.n_qubits,
                       "saved_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)

    @staticmethod
    def load_best(checkpoint_dir: str = settings.checkpoint_dir) -> Optional[dict]:
        if not os.path.isdir(checkpoint_dir):
            return None
        ckpts = []
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(checkpoint_dir, f)) as fh:
                        ckpts.append(json.load(fh))
                except Exception:
                    pass
        return min(ckpts, key=lambda c: c.get("loss", float("inf"))) if ckpts else None
