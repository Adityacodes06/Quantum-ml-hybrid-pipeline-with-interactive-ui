"""api/routes/circuits.py — POST /run and POST /run/circuit"""
from __future__ import annotations
import logging, math, random
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_executor
from api.schemas import CircuitTypeRequest, JobResponse, RunCircuitRequest
from core.circuit_factory import build_bell_state, build_ghz_state, build_variational_bottleneck
from core.executor import QuantumExecutor
from core.quantum_backend import BackendMode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/run", tags=["circuits"])


@router.post("", response_model=JobResponse, summary="Run variational bottleneck circuit")
async def run_variational(req: RunCircuitRequest, executor: QuantumExecutor = Depends(get_executor)):
    try:
        circuit = build_variational_bottleneck(len(req.input_data), req.input_data, req.thetas)
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        job = executor.run(
            circuit=circuit, mode=BackendMode(req.backend_mode),
            shots=req.shots, optimization_level=req.optimization_level,
            preferred_device=req.preferred_device, async_mode=req.async_mode,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return JobResponse(**job.to_dict())


@router.post("/circuit", response_model=JobResponse, summary="Run a named built-in circuit")
async def run_named(req: CircuitTypeRequest, executor: QuantumExecutor = Depends(get_executor)):
    if req.circuit_type == "bell":
        circuit = build_bell_state()
    elif req.circuit_type == "ghz":
        circuit = build_ghz_state(req.n_qubits)
    elif req.circuit_type == "variational":
        n = req.n_qubits
        circuit = build_variational_bottleneck(
            n, [random.uniform(0, math.pi) for _ in range(n)],
               [random.uniform(0, math.pi) for _ in range(n)],
        )
    else:
        raise HTTPException(400, f"Unknown circuit_type: {req.circuit_type!r}")
    try:
        job = executor.run(
            circuit=circuit, mode=BackendMode(req.backend_mode),
            shots=req.shots, preferred_device=req.preferred_device, async_mode=req.async_mode,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return JobResponse(**job.to_dict())
