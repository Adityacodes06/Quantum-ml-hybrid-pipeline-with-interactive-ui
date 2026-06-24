"""api/routes/circuits.py — POST /run, POST /run/circuit, POST /run/train, GET /run/circuit/draw"""
from __future__ import annotations
import logging, math, random
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from api.dependencies import get_executor
from api.schemas import CircuitTypeRequest, JobResponse, RunCircuitRequest, TrainRequest, TrainResponse, QasmRunRequest, DensityMatrixRequest, DensityMatrixResponse
from core.circuit_factory import (
    build_bell_state, build_ghz_state, build_variational_bottleneck,
    build_amplitude_encoding, build_w_state, build_qft, build_iqft,
    build_grover, build_bernstein_vazirani, build_dj_algo, build_qaoa_maxcut,
    build_random_circuit, build_hwea, build_quantum_volume, build_cluster_state,
    build_graph_state, build_teleportation, build_superdense_coding,
    build_swap_test, build_simon_algo, build_phase_estimation, build_vqe_ansatz,
    build_entanglement_swapping, build_shor_dummy
)
from core.executor import QuantumExecutor
from core.quantum_backend import BackendMode
from training.trainer import VariationalTrainer, TrainingConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/run", tags=["circuits"])


@router.post("", response_model=JobResponse, summary="Run variational bottleneck circuit")
async def run_variational(req: RunCircuitRequest, executor: QuantumExecutor = Depends(get_executor)):
    try:
        circuit = build_variational_bottleneck(
            len(req.input_data), req.input_data, req.thetas,
            encode_gate=req.encode_gate, entangle_type=req.entangle_type, var_gate=req.var_gate
        )
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
    elif req.circuit_type == "w_state": circuit = build_w_state(req.n_qubits)
    elif req.circuit_type == "qft": circuit = build_qft(req.n_qubits)
    elif req.circuit_type == "iqft": circuit = build_iqft(req.n_qubits)
    elif req.circuit_type == "grover": circuit = build_grover(req.n_qubits)
    elif req.circuit_type == "bv": circuit = build_bernstein_vazirani(req.n_qubits)
    elif req.circuit_type == "dj": circuit = build_dj_algo(req.n_qubits)
    elif req.circuit_type == "qaoa": circuit = build_qaoa_maxcut(req.n_qubits)
    elif req.circuit_type == "random": circuit = build_random_circuit(req.n_qubits)
    elif req.circuit_type == "hwea": circuit = build_hwea(req.n_qubits)
    elif req.circuit_type == "qv": circuit = build_quantum_volume(req.n_qubits)
    elif req.circuit_type == "cluster": circuit = build_cluster_state(req.n_qubits)
    elif req.circuit_type == "graph": circuit = build_graph_state(req.n_qubits)
    elif req.circuit_type == "teleportation": circuit = build_teleportation(req.n_qubits)
    elif req.circuit_type == "superdense": circuit = build_superdense_coding(req.n_qubits)
    elif req.circuit_type == "swap_test": circuit = build_swap_test(req.n_qubits)
    elif req.circuit_type == "simon": circuit = build_simon_algo(req.n_qubits)
    elif req.circuit_type == "qpe": circuit = build_phase_estimation(req.n_qubits)
    elif req.circuit_type == "vqe": circuit = build_vqe_ansatz(req.n_qubits)
    elif req.circuit_type == "ent_swap": circuit = build_entanglement_swapping(req.n_qubits)
    elif req.circuit_type == "shor_dummy": circuit = build_shor_dummy(req.n_qubits)
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
    setattr(config, "encode_gate", req.encode_gate)
    setattr(config, "entangle_type", req.entangle_type)
    setattr(config, "var_gate", req.var_gate)
    
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
        circuit = build_variational_bottleneck(
            len(req.input_data), req.input_data, req.thetas,
            encode_gate=req.encode_gate, entangle_type=req.entangle_type, var_gate=req.var_gate
        )
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


@router.post("/statevector", response_model=DensityMatrixResponse, summary="Compute density matrix from circuit state")
async def compute_density_matrix(req: DensityMatrixRequest):
    """
    Builds the circuit WITHOUT measurements, evolves |0⟩^n through it,
    and returns the full statevector, density matrix ρ = |ψ⟩⟨ψ|,
    measurement probabilities, purity Tr(ρ²), and von Neumann entropy.
    """
    import numpy as np
    from qiskit.quantum_info import Statevector, DensityMatrix, entropy

    # ── Build circuit (without measurement gates) ──────────────────────
    if req.circuit_type:
        # Named circuit — strip measurements
        ct = req.circuit_type
        nq = req.n_qubits or len(req.input_data)
        if ct == "bell": qc = build_bell_state()
        elif ct == "ghz": qc = build_ghz_state(nq)
        elif ct == "variational":
            qc = build_variational_bottleneck(
                nq, [random.uniform(0, math.pi) for _ in range(nq)],
                     [random.uniform(0, math.pi) for _ in range(nq)])
        elif ct == "w_state": qc = build_w_state(nq)
        elif ct == "qft": qc = build_qft(nq)
        elif ct == "iqft": qc = build_iqft(nq)
        elif ct == "grover": qc = build_grover(nq)
        elif ct == "bv": qc = build_bernstein_vazirani(nq)
        elif ct == "dj": qc = build_dj_algo(nq)
        elif ct == "qaoa": qc = build_qaoa_maxcut(nq)
        elif ct == "random": qc = build_random_circuit(nq)
        elif ct == "hwea": qc = build_hwea(nq)
        elif ct == "qv": qc = build_quantum_volume(nq)
        elif ct == "cluster": qc = build_cluster_state(nq)
        elif ct == "graph": qc = build_graph_state(nq)
        elif ct == "teleportation": qc = build_teleportation(nq)
        elif ct == "superdense": qc = build_superdense_coding(nq)
        elif ct == "swap_test": qc = build_swap_test(nq)
        elif ct == "simon": qc = build_simon_algo(nq)
        elif ct == "qpe": qc = build_phase_estimation(nq)
        elif ct == "vqe": qc = build_vqe_ansatz(nq)
        elif ct == "ent_swap": qc = build_entanglement_swapping(nq)
        elif ct == "shor_dummy": qc = build_shor_dummy(nq)
        else:
            raise HTTPException(400, f"Unknown circuit_type: {ct!r}")
    else:
        # Variational bottleneck from user inputs
        try:
            qc = build_variational_bottleneck(
                len(req.input_data), req.input_data, req.thetas,
                encode_gate=req.encode_gate, entangle_type=req.entangle_type,
                var_gate=req.var_gate
            )
        except ValueError as e:
            raise HTTPException(422, str(e))

    # ── Strip measurement gates to get unitary-only circuit ────────────
    from qiskit import QuantumCircuit as QC
    qc_no_meas = QC(qc.num_qubits)
    for instruction in qc.data:
        if instruction.operation.name != "measure" and instruction.operation.name != "barrier":
            qc_no_meas.append(instruction.operation, instruction.qubits, instruction.clbits)

    # ── Compute statevector and density matrix ─────────────────────────
    try:
        sv = Statevector.from_instruction(qc_no_meas)
        dm = DensityMatrix(sv)
        n_qubits = qc_no_meas.num_qubits
        dim = 2 ** n_qubits

        # Statevector as formatted strings
        sv_arr = sv.data
        sv_strs = []
        for amp in sv_arr:
            r, im = amp.real, amp.imag
            if abs(im) < 1e-10:
                sv_strs.append(f"{r:.6f}")
            elif abs(r) < 1e-10:
                sv_strs.append(f"{im:+.6f}i")
            else:
                sv_strs.append(f"{r:.6f}{im:+.6f}i")

        # Density matrix as formatted strings
        dm_arr = dm.data
        dm_strs = []
        for row in dm_arr:
            row_strs = []
            for val in row:
                r, im = val.real, val.imag
                if abs(im) < 1e-10:
                    row_strs.append(f"{r:.6f}")
                elif abs(r) < 1e-10:
                    row_strs.append(f"{im:+.6f}i")
                else:
                    row_strs.append(f"{r:.6f}{im:+.6f}i")
            dm_strs.append(row_strs)

        # Measurement probabilities
        probs = sv.probabilities_dict()
        # Ensure all basis states present
        prob_dict = {}
        for i in range(dim):
            bs = format(i, f"0{n_qubits}b")
            prob_dict[bs] = float(probs.get(bs, 0.0))

        # Purity: Tr(ρ²)
        purity = float(np.real(np.trace(dm_arr @ dm_arr)))

        # Von Neumann entropy
        try:
            vn_entropy = float(entropy(dm, base=2))
        except Exception:
            vn_entropy = 0.0

        return DensityMatrixResponse(
            num_qubits=n_qubits,
            dimension=dim,
            statevector=sv_strs,
            density_matrix=dm_strs,
            probabilities=prob_dict,
            purity=purity,
            von_neumann_entropy=vn_entropy,
        )
    except Exception as e:
        logger.error("Density matrix computation failed: %s", e)
        raise HTTPException(500, f"Density matrix computation failed: {e}")

