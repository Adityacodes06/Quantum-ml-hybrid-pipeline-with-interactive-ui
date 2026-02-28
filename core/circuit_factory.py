"""
core/circuit_factory.py — all quantum circuit constructors.
No backend imports. Every function returns a plain QuantumCircuit.
"""
from __future__ import annotations
import math
from typing import List
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_variational_bottleneck(
    n_qubits: int,
    input_data: List[float],
    thetas: List[float],
) -> QuantumCircuit:
    """
    Encode → Entangle (CX chain) → Variational → Measure.
    Standard bottleneck for hybrid QML inference.
    """
    if len(input_data) != n_qubits:
        raise ValueError(f"input_data length {len(input_data)} != n_qubits {n_qubits}")
    if len(thetas) != n_qubits:
        raise ValueError(f"thetas length {len(thetas)} != n_qubits {n_qubits}")

    qc = QuantumCircuit(n_qubits, name="variational_bottleneck")
    for i, x in enumerate(input_data):
        qc.ry(float(x), i)
    qc.barrier(label="encode")
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.barrier(label="entangle")
    for i, t in enumerate(thetas):
        qc.ry(float(t), i)
    qc.barrier(label="variational")
    qc.measure_all()
    return qc


def build_parametric_circuit(n_qubits: int, reps: int = 1) -> QuantumCircuit:
    """Symbolic-parameter circuit for training. Bind with circuit.assign_parameters()."""
    inputs = ParameterVector("x", n_qubits)
    thetas = ParameterVector("θ", n_qubits * reps)
    phis   = ParameterVector("φ", n_qubits * reps)
    qc = QuantumCircuit(n_qubits, name=f"parametric_{n_qubits}q_{reps}rep")
    for i in range(n_qubits):
        qc.ry(inputs[i], i)
    qc.barrier()
    for rep in range(reps):
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(n_qubits):
            qc.rz(thetas[rep * n_qubits + i], i)
            qc.rx(phis[rep * n_qubits + i], i)
        qc.barrier()
    qc.measure_all()
    return qc


def build_bell_state() -> QuantumCircuit:
    """2-qubit Bell state. ~50% |00⟩, ~50% |11⟩ — basic entanglement benchmark."""
    qc = QuantumCircuit(2, name="bell_state")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def build_ghz_state(n_qubits: int = 3) -> QuantumCircuit:
    """n-qubit GHZ. Tests multi-qubit coherence on real hardware."""
    qc = QuantumCircuit(n_qubits, name=f"ghz_{n_qubits}q")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def build_amplitude_encoding(data: List[float]) -> QuantumCircuit:
    """Encode a vector as quantum amplitudes. Auto-pads to next power of 2."""
    n = len(data)
    n_qubits = math.ceil(math.log2(max(n, 2)))
    target = 2 ** n_qubits
    padded = list(data) + [0.0] * (target - n)
    norm = math.sqrt(sum(v ** 2 for v in padded))
    if norm < 1e-10:
        raise ValueError("Data vector has near-zero norm — cannot encode")
    normalised = [v / norm for v in padded]
    qc = QuantumCircuit(n_qubits, name="amplitude_encoding")
    qc.initialize(normalised, list(range(n_qubits)))
    qc.measure_all()
    return qc


def circuit_info(qc: QuantumCircuit) -> dict:
    return {
        "name":        qc.name,
        "num_qubits":  qc.num_qubits,
        "depth":       qc.depth(),
        "num_gates":   sum(qc.count_ops().values()),
        "gate_counts": dict(qc.count_ops()),
    }
