"""api/routes/circuits.py — POST /run, POST /run/circuit, POST /run/train, GET /run/circuit/draw"""
from __future__ import annotations
import logging, math, random
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from api.dependencies import get_executor
from api.schemas import CircuitTypeRequest, JobResponse, RunCircuitRequest, TrainRequest, TrainResponse, QasmRunRequest
from core.circuit_factory import (
    build_bell_state, build_ghz_state, build_variational_bottleneck,
    build_amplitude_encoding,
)
from core.executor import QuantumExecutor
from core.quantum_backend import BackendMode
from training.trainer import VariationalTrainer, TrainingConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/run", tags=["circuits"])


@router.post("", response_model=JobResponse, summary="Run variational bottleneck circuit")
async def run_variational(req: RunCircuitRequest, executor: QuantumExecutor = Depends(get_executor)):
    try:
        circuit = build_variational_bottleneck(len(req.input_data), req.input_data, req.thetas)
    except ValueError as e:
        raise HTTPException(422, str(e))
    try:
        job = await run_in_threadpool(
            executor.run,
            circuit=circuit, mode=BackendMode(req.backend_mode),
            shots=req.shots, optimization_level=req.optimization_level,
            preferred_device=req.preferred_device, async_mode=req.async_mode,
            seed=req.seed, noise_rate=req.noise_rate,
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
        job = await run_in_threadpool(
            executor.run,
            circuit=circuit, mode=BackendMode(req.backend_mode),
            shots=req.shots, preferred_device=req.preferred_device, async_mode=req.async_mode,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return JobResponse(**job.to_dict())


@router.post("/train", response_model=TrainResponse, summary="Run server-side variational training")
async def run_training(req: TrainRequest, executor: QuantumExecutor = Depends(get_executor)):
    """
    Server-side parameter-shift training. Runs entirely on the local simulator.
    Returns the full training history including loss curve and best parameters.
    """
    config = TrainingConfig(
        n_qubits=len(req.input_data),
        learning_rate=req.learning_rate,
        max_iterations=req.max_iterations,
        shots=req.shots,
        convergence_threshold=req.convergence_threshold,
    )
    # The Trainer doesn't currently take optimizer config natively in the BaseModel
    # We will pass it directly to trainer or update TrainingConfig
    setattr(config, "optimizer", req.optimizer)
    setattr(config, "seed", req.seed)
    
    trainer = VariationalTrainer(config=config, executor=executor)
    try:
        history = await run_in_threadpool(trainer.train, req.input_data, req.thetas)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")
    return TrainResponse(
        losses=history.losses,
        best_loss=history.best_loss,
        best_thetas=history.best_thetas,
        iterations_run=history.iterations_run,
        converged=history.converged,
        elapsed_seconds=history.elapsed_seconds,
    )


@router.get("/circuit/draw", summary="Get ASCII circuit diagram")
async def draw_circuit(
    circuit_type: str = "bell",
    n_qubits: int = 2,
):
    """Returns an ASCII art representation of the requested circuit."""
    if circuit_type == "bell":
        qc = build_bell_state()
    elif circuit_type == "ghz":
        qc = build_ghz_state(n_qubits)
    elif circuit_type == "variational":
        qc = build_variational_bottleneck(
            n_qubits,
            [random.uniform(0, math.pi) for _ in range(n_qubits)],
            [random.uniform(0, math.pi) for _ in range(n_qubits)],
        )
    elif circuit_type == "amplitude":
        qc = build_amplitude_encoding([1.0 / (i + 1) for i in range(n_qubits)])
    else:
        raise HTTPException(400, f"Unknown circuit_type: {circuit_type!r}")

    try:
        diagram = qc.draw(output="text").__str__()
    except Exception:
        diagram = str(qc)

    return PlainTextResponse(content=diagram, media_type="text/plain")


@router.post("/circuit/qasm", summary="Get OpenQASM 2.0 representation")
async def export_qasm(
    req: RunCircuitRequest,
):
    """
    Returns the OpenQASM string for the variational circuit defined by
    the provided input_data and thetas.
    """
    try:
        circuit = build_variational_bottleneck(len(req.input_data), req.input_data, req.thetas)
        qasm_str = circuit.qasm()
        return PlainTextResponse(
            content=qasm_str,
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="variational_circuit.qasm"'}
        )
    except Exception as e:
        raise HTTPException(500, f"QASM generation failed: {e}")


@router.post("/qasm", response_model=JobResponse, summary="Execute raw OpenQASM 2.0")
async def run_qasm_route(req: QasmRunRequest, executor: QuantumExecutor = Depends(get_executor)):
    from qiskit import QuantumCircuit
    try:
        circuit = QuantumCircuit.from_qasm_str(req.qasm_str)
    except Exception as e:
        raise HTTPException(400, f"Invalid QASM string: {e}")
    try:
        job = await run_in_threadpool(
            executor.run,
            circuit=circuit, mode=BackendMode(req.backend_mode),
            shots=req.shots, optimization_level=req.optimization_level,
            seed=req.seed, noise_rate=req.noise_rate,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return JobResponse(**job.to_dict())
