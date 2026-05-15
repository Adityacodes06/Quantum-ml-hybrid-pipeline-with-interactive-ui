"""Tests for training/loss.py — quantum loss functions."""
import pytest
from training.loss import (
    expectation_value, cross_entropy_loss, fidelity_loss,
    tv_distance, kl_divergence,
)


class TestExpectationValue:
    def test_all_zeros(self):
        """All |0⟩ outcomes → EV = +1 (Pauli-Z eigenvalue)."""
        counts = {"00": 100}
        assert expectation_value(counts, qubit=0) == pytest.approx(1.0)
        assert expectation_value(counts, qubit=1) == pytest.approx(1.0)

    def test_all_ones(self):
        """All |1⟩ outcomes → EV = -1."""
        counts = {"11": 100}
        assert expectation_value(counts, qubit=0) == pytest.approx(-1.0)
        assert expectation_value(counts, qubit=1) == pytest.approx(-1.0)

    def test_bell_state(self):
        """50/50 Bell state: EV on qubit 0 = 0."""
        counts = {"00": 500, "11": 500}
        assert expectation_value(counts, qubit=0) == pytest.approx(0.0)

    def test_biased(self):
        """75% |0⟩, 25% |1⟩ → EV = 0.5."""
        counts = {"0": 75, "1": 25}
        assert expectation_value(counts, qubit=0) == pytest.approx(0.5)

    def test_empty_counts(self):
        assert expectation_value({}) == 0.0

    def test_qubit_index_beyond_string(self):
        """Qubit index beyond bitstring length should default to 0."""
        counts = {"0": 100}
        result = expectation_value(counts, qubit=5)
        assert isinstance(result, float)


class TestCrossEntropyLoss:
    def test_perfect_match(self):
        """If target state has 100% probability, loss = -ln(1) = 0."""
        counts = {"00": 1000}
        assert cross_entropy_loss(counts, "00") == pytest.approx(0.0)

    def test_no_match(self):
        """If target state has 0 probability, loss is large (clamped)."""
        counts = {"00": 1000}
        loss = cross_entropy_loss(counts, "11")
        assert loss > 10  # -ln(1e-10) ≈ 23

    def test_empty_counts(self):
        assert cross_entropy_loss({}, "00") == float("inf")


class TestFidelityLoss:
    def test_perfect(self):
        counts = {"00": 1000}
        assert fidelity_loss(counts, "00") == pytest.approx(0.0)

    def test_zero_fidelity(self):
        counts = {"00": 1000}
        assert fidelity_loss(counts, "11") == pytest.approx(1.0)

    def test_partial(self):
        counts = {"00": 700, "11": 300}
        assert fidelity_loss(counts, "00") == pytest.approx(0.3)


class TestTVDistance:
    def test_identical(self):
        p = {"00": 500, "11": 500}
        assert tv_distance(p, p) == pytest.approx(0.0)

    def test_completely_different(self):
        p = {"00": 1000}
        q = {"11": 1000}
        assert tv_distance(p, q) == pytest.approx(1.0)

    def test_empty(self):
        assert tv_distance({}, {"00": 100}) == 1.0


class TestKLDivergence:
    def test_identical(self):
        p = {"00": 500, "11": 500}
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_empty(self):
        assert kl_divergence({}, {"00": 100}) == float("inf")

    def test_non_negative(self):
        p = {"00": 700, "11": 300}
        q = {"00": 500, "11": 500}
        assert kl_divergence(p, q) >= 0
