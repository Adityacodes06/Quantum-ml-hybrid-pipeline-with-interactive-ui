"""tests/test_circuits.py — circuit factory unit tests. No backend required."""
import math
import pytest
from qiskit import QuantumCircuit
from core.circuit_factory import (
    build_variational_bottleneck, build_bell_state, build_ghz_state,
    build_amplitude_encoding, circuit_info,
)

class TestVariational:
    def test_basic(self):
        qc = build_variational_bottleneck(2, [1.0, 0.5], [0.8, 1.3])
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

    def test_n_qubits(self):
        for n in range(2, 7):
            qc = build_variational_bottleneck(n, [0.1]*n, [0.2]*n)
            assert qc.num_qubits == n

    def test_input_mismatch_raises(self):
        with pytest.raises(ValueError, match="input_data length"):
            build_variational_bottleneck(2, [1.0], [0.8, 1.3])

    def test_theta_mismatch_raises(self):
        with pytest.raises(ValueError, match="thetas length"):
            build_variational_bottleneck(2, [1.0, 0.5], [0.8])

class TestBell:
    def test_shape(self):
        qc = build_bell_state()
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

class TestGHZ:
    def test_n_qubits(self):
        for n in range(2, 6):
            qc = build_ghz_state(n)
            assert qc.num_qubits == n
            assert qc.num_clbits == n

class TestAmplitude:
    def test_power_of_two(self):
        qc = build_amplitude_encoding([0.5, 0.5, 0.5, 0.5])
        assert qc.num_qubits == 2

    def test_pads_to_power_of_two(self):
        qc = build_amplitude_encoding([1.0, 0.0, 0.0])
        assert qc.num_qubits == 2

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="near-zero norm"):
            build_amplitude_encoding([0.0, 0.0, 0.0, 0.0])

class TestCircuitInfo:
    def test_keys(self):
        info = circuit_info(build_bell_state())
        for k in ("name","num_qubits","depth","num_gates","gate_counts"):
            assert k in info
        assert info["num_qubits"] == 2
