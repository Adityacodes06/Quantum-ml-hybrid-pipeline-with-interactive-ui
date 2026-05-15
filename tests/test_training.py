"""Tests for training/trainer.py — variational parameter optimisation."""
import pytest
import math
from training.trainer import VariationalTrainer, TrainingConfig, TrainingHistory


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.n_qubits == 2
        assert cfg.learning_rate > 0
        assert cfg.max_iterations > 0

    def test_custom(self):
        cfg = TrainingConfig(n_qubits=4, learning_rate=0.05, max_iterations=10)
        assert cfg.n_qubits == 4
        assert cfg.learning_rate == 0.05


class TestTrainingHistory:
    def test_to_dict(self):
        h = TrainingHistory(losses=[0.5, 0.3], best_loss=0.3, iterations_run=2)
        d = h.to_dict()
        assert d["losses"] == [0.5, 0.3]
        assert d["best_loss"] == 0.3


class TestVariationalTrainer:
    def test_input_validation(self):
        cfg = TrainingConfig(n_qubits=2, max_iterations=1)
        trainer = VariationalTrainer(config=cfg)
        with pytest.raises(ValueError, match="input_data length"):
            trainer.train([1.0, 2.0, 3.0])  # 3 inputs for 2-qubit config

    def test_short_training(self):
        """Run 2 iterations to verify the training loop works end-to-end."""
        cfg = TrainingConfig(n_qubits=2, max_iterations=2, shots=64)
        trainer = VariationalTrainer(config=cfg)
        history = trainer.train([0.5, 1.0], [0.1, 0.2])

        assert history.iterations_run <= 2
        assert len(history.losses) > 0
        assert history.best_loss < float("inf")
        assert len(history.best_thetas) == 2
        assert history.elapsed_seconds >= 0

    def test_convergence_detection(self):
        """If we set a very high threshold, training should converge quickly."""
        cfg = TrainingConfig(n_qubits=2, max_iterations=100, shots=64,
                             convergence_threshold=100.0)  # trivially easy
        trainer = VariationalTrainer(config=cfg)
        history = trainer.train([0.5, 1.0])

        assert history.converged is True
        assert history.iterations_run < 100  # should stop early

    def test_custom_initial_thetas(self):
        cfg = TrainingConfig(n_qubits=2, max_iterations=1, shots=64)
        trainer = VariationalTrainer(config=cfg)
        history = trainer.train([0.5, 1.0], initial_thetas=[math.pi, math.pi / 2])
        assert len(history.losses) == 1
