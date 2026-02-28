"""training/loss.py — quantum loss functions operating on count dicts."""
from __future__ import annotations
import math
from typing import Dict


def expectation_value(counts: Dict[str, int], qubit: int = 0) -> float:
    """Pauli-Z expectation on the given qubit. Returns float in [-1, +1]."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ev = 0.0
    for bits, c in counts.items():
        # Qiskit bitstring: rightmost char = qubit 0
        idx = -(qubit + 1)
        val = int(bits[idx]) if abs(idx) <= len(bits) else 0
        ev += c * (1 - 2 * val)
    return ev / total


def cross_entropy_loss(counts: Dict[str, int], target_state: str) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("inf")
    return -math.log(max(counts.get(target_state, 0) / total, 1e-10))


def fidelity_loss(counts: Dict[str, int], target_state: str) -> float:
    total = sum(counts.values())
    return 1.0 - (counts.get(target_state, 0) / total if total else 0.0)


def tv_distance(p: Dict[str, int], q: Dict[str, int]) -> float:
    """Total variation distance ∈ [0,1] — useful for sim vs hardware comparison."""
    tp, tq = sum(p.values()), sum(q.values())
    if tp == 0 or tq == 0:
        return 1.0
    states = set(p) | set(q)
    return 0.5 * sum(abs(p.get(s,0)/tp - q.get(s,0)/tq) for s in states)


def kl_divergence(p: Dict[str, int], q: Dict[str, int]) -> float:
    """KL divergence D(P||Q). Useful for comparing distributions."""
    tp, tq = sum(p.values()), sum(q.values())
    if tp == 0 or tq == 0:
        return float("inf")
    kl = 0.0
    for s in set(p) | set(q):
        pv = p.get(s,0)/tp
        qv = max(q.get(s,0)/tq, 1e-10)
        if pv > 0:
            kl += pv * math.log(pv / qv)
    return kl
