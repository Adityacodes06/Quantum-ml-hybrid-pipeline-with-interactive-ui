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
    encode_gate: str = "ry",
    entangle_type: str = "linear",
    var_gate: str = "ry"
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
        getattr(qc, encode_gate)(float(x), i)
    qc.barrier(label="encode")
    
    if entangle_type == "linear":
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    elif entangle_type == "circular":
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    elif entangle_type == "full":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.cx(i, j)
                
    qc.barrier(label="entangle")
    for i, t in enumerate(thetas):
        getattr(qc, var_gate)(float(t), i)
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
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    QFT, GroverOperator, RealAmplitudes, EfficientSU2, QAOAAnsatz,
    QuantumVolume, TwoLocal, GraphState
)
from qiskit.circuit.random import random_circuit

def build_w_state(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"w_state_{n_qubits}q")
    if n_qubits < 2:
        return qc
    # Simplified W-state via controlled rotations (just an approximation for demo)
    qc.x(0)
    for i in range(1, n_qubits):
        theta = 2 * math.asin(1 / math.sqrt(n_qubits - i + 1))
        qc.cry(theta, i-1, i)
        qc.cx(i, i-1)
    qc.measure_all()
    return qc

def build_qft(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"qft_{n_qubits}q")
    qc.append(QFT(n_qubits), range(n_qubits))
    qc.measure_all()
    return qc

def build_iqft(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"iqft_{n_qubits}q")
    qc.append(QFT(n_qubits, inverse=True), range(n_qubits))
    qc.measure_all()
    return qc

def build_grover(n_qubits: int) -> QuantumCircuit:
    # simple oracle that flips sign of |11..1>
    oracle = QuantumCircuit(n_qubits)
    oracle.h(n_qubits-1)
    oracle.mcx(list(range(n_qubits-1)), n_qubits-1)
    oracle.h(n_qubits-1)
    grover_op = GroverOperator(oracle)
    
    qc = QuantumCircuit(n_qubits, name=f"grover_{n_qubits}q")
    qc.h(range(n_qubits))
    iterations = math.floor(math.pi/4 * math.sqrt(2**n_qubits))
    for _ in range(max(1, iterations)):
        qc.append(grover_op, range(n_qubits))
    qc.measure_all()
    return qc

def build_bernstein_vazirani(n_qubits: int) -> QuantumCircuit:
    # hidden string = 1010...
    qc = QuantumCircuit(n_qubits, n_qubits-1, name=f"bv_{n_qubits}q")
    if n_qubits < 2: return qc
    qc.x(n_qubits-1)
    qc.h(range(n_qubits))
    for i in range(n_qubits-1):
        if i % 2 == 0:
            qc.cx(i, n_qubits-1)
    qc.h(range(n_qubits-1))
    qc.measure(range(n_qubits-1), range(n_qubits-1))
    return qc

def build_dj_algo(n_qubits: int) -> QuantumCircuit:
    # balanced oracle (CNOT from 0 to target)
    qc = QuantumCircuit(n_qubits, n_qubits-1, name=f"dj_{n_qubits}q")
    if n_qubits < 2: return qc
    qc.x(n_qubits-1)
    qc.h(range(n_qubits))
    qc.cx(0, n_qubits-1)
    qc.h(range(n_qubits-1))
    qc.measure(range(n_qubits-1), range(n_qubits-1))
    return qc

def build_qaoa_maxcut(n_qubits: int) -> QuantumCircuit:
    # ring graph
    from qiskit.quantum_info import Pauli, SparsePauliOp
    paulis = [("ZZ", [i, (i+1)%n_qubits]) for i in range(n_qubits)]
    op = SparsePauliOp.from_sparse_list(paulis, num_qubits=n_qubits)
    qaoa = QAOAAnsatz(op, reps=1)
    qc = QuantumCircuit(n_qubits, name=f"qaoa_{n_qubits}q")
    qc.append(qaoa.bind_parameters([0.5, 0.5]), range(n_qubits))
    qc.measure_all()
    return qc

def build_random_circuit(n_qubits: int) -> QuantumCircuit:
    qc = random_circuit(n_qubits, depth=n_qubits, measure=True)
    qc.name = f"random_{n_qubits}q"
    return qc

def build_hwea(n_qubits: int) -> QuantumCircuit:
    ansatz = EfficientSU2(n_qubits, reps=2, entanglement='linear')
    qc = QuantumCircuit(n_qubits, name=f"hwea_{n_qubits}q")
    # bind with random parameters for quick simulation
    params = [0.1 * i for i in range(ansatz.num_parameters)]
    qc.append(ansatz.bind_parameters(params), range(n_qubits))
    qc.measure_all()
    return qc

def build_quantum_volume(n_qubits: int) -> QuantumCircuit:
    qc = QuantumVolume(n_qubits, depth=n_qubits)
    qc.measure_all()
    qc.name = f"qv_{n_qubits}q"
    return qc

def build_cluster_state(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name=f"cluster_{n_qubits}q")
    qc.h(range(n_qubits))
    for i in range(n_qubits-1):
        qc.cz(i, i+1)
    qc.measure_all()
    return qc

def build_graph_state(n_qubits: int) -> QuantumCircuit:
    import networkx as nx
    G = nx.cycle_graph(n_qubits)
    qc = GraphState(G)
    qc.measure_all()
    qc.name = f"graph_state_{n_qubits}q"
    return qc

def build_teleportation(n_qubits: int) -> QuantumCircuit:
    # uses exactly 3 qubits, if user asks for n_qubits, we just use 3
    qc = QuantumCircuit(3, 2, name="teleportation")
    qc.x(0) # State to teleport
    qc.h(1); qc.cx(1, 2) # Bell pair
    qc.cx(0, 1); qc.h(0) # Alice operation
    qc.measure([0, 1], [0, 1])
    # Bob conditionally applies
    qc.cx(1, 2)
    qc.cz(0, 2)
    # We can measure the result to verify it's 1
    # Adding a classical bit for the result
    return qc

def build_superdense_coding(n_qubits: int) -> QuantumCircuit:
    # uses 2 qubits
    qc = QuantumCircuit(2, 2, name="superdense_coding")
    qc.h(0); qc.cx(0, 1) # Bell pair
    # Encode '11'
    qc.z(0); qc.x(0)
    # Decode
    qc.cx(0, 1); qc.h(0)
    qc.measure([0,1], [0,1])
    return qc

def build_swap_test(n_qubits: int) -> QuantumCircuit:
    # Requires an odd number of qubits: 1 control, 2 states
    n = max(3, n_qubits)
    if n % 2 == 0: n += 1
    half = (n-1)//2
    qc = QuantumCircuit(n, 1, name=f"swap_test_{n}q")
    qc.h(0)
    for i in range(half):
        qc.cswap(0, 1+i, 1+half+i)
    qc.h(0)
    qc.measure(0, 0)
    return qc

def build_simon_algo(n_qubits: int) -> QuantumCircuit:
    # simple simon algo where hidden string is all 0s
    n = max(1, n_qubits//2)
    qc = QuantumCircuit(2*n, n, name=f"simon_{n_qubits}q")
    qc.h(range(n))
    for i in range(n):
        qc.cx(i, n+i)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def build_phase_estimation(n_qubits: int) -> QuantumCircuit:
    n = max(2, n_qubits)
    qc = QuantumCircuit(n, n-1, name=f"qpe_{n}q")
    qc.x(n-1) # eigenstate
    qc.h(range(n-1))
    for i in range(n-1):
        qc.cp(math.pi/(2**(i)), i, n-1)
    qc.append(QFT(n-1, inverse=True), range(n-1))
    qc.measure(range(n-1), range(n-1))
    return qc

def build_vqe_ansatz(n_qubits: int) -> QuantumCircuit:
    ansatz = TwoLocal(n_qubits, 'ry', 'cz', 'linear', reps=1)
    qc = QuantumCircuit(n_qubits, name=f"vqe_{n_qubits}q")
    params = [0.1 * i for i in range(ansatz.num_parameters)]
    qc.append(ansatz.bind_parameters(params), range(n_qubits))
    qc.measure_all()
    return qc

def build_entanglement_swapping(n_qubits: int) -> QuantumCircuit:
    # 4 qubits needed
    qc = QuantumCircuit(4, 2, name="ent_swapping")
    qc.h(0); qc.cx(0,1)
    qc.h(2); qc.cx(2,3)
    # Bell measurement on 1 and 2
    qc.cx(1,2); qc.h(1)
    qc.measure([1,2], [0,1])
    return qc

def build_shor_dummy(n_qubits: int) -> QuantumCircuit:
    # just an empty shell for a period finding demo
    qc = QuantumCircuit(n_qubits, name=f"shor_dummy_{n_qubits}q")
    qc.h(range(n_qubits//2))
    qc.measure_all()
    return qc

