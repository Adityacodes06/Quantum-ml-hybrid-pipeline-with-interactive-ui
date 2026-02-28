"""
core/executor.py — the execution engine.
Glues BackendManager + circuit + transpilation + Sampler + JobStore.
This is the only file that calls qiskit_ibm_runtime.Sampler.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional
from qiskit import QuantumCircuit

from config.settings import settings
from core.circuit_factory import circuit_info
from core.job_store import JobStatus, JobStore, QuantumJob
from core.quantum_backend import BackendManager, BackendMode

logger = logging.getLogger(__name__)


class QuantumExecutor:

    def __init__(self, backend_manager: BackendManager, job_store: JobStore):
        self._mgr   = backend_manager
        self._store = job_store

    def run(
        self,
        circuit: QuantumCircuit,
        mode: BackendMode              = BackendMode.SIMULATOR,
        shots: int                     = settings.default_shots,
        optimization_level: int        = settings.default_optimization_level,
        preferred_device: Optional[str] = None,
        async_mode: bool               = False,
    ) -> QuantumJob:
        backend, backend_name = self._mgr.get(mode, preferred_device)
        info = circuit_info(circuit)

        job = QuantumJob.create(
            backend_mode=mode.value,
            backend_name=backend_name,
            num_qubits=info["num_qubits"],
            circuit_depth=info["depth"],
            circuit_name=info["name"],
            shots=shots,
            optimization_level=optimization_level,
        )
        self._store.save(job)

        transpiled = self._transpile(circuit, backend, optimization_level)

        if mode == BackendMode.REAL and async_mode:
            self._submit_async(job, backend, transpiled, shots)
        else:
            self._run_sync(job, backend, transpiled, shots)

        return self._store.get(job.job_id)

    def fetch_result(self, job_id: str) -> Optional[QuantumJob]:
        """Poll an async IBM job. Updates store in-place."""
        job = self._store.get(job_id)
        if job is None or job.is_terminal() or job.ibm_job_id is None:
            return job
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            ibm_job = QiskitRuntimeService().job(job.ibm_job_id)
            sname   = ibm_job.status().name
            if sname == "DONE":
                counts = self._extract_counts(ibm_job.result(), job.num_qubits, job.shots)
                self._store.update(job_id, status=JobStatus.COMPLETED,
                                   completed_at=_now(), result={"counts": counts, "shots": job.shots})
            elif sname in ("ERROR", "CANCELLED"):
                self._store.update(job_id, status=JobStatus.FAILED, error=sname)
            else:
                pos = None
                try: pos = ibm_job.queue_position()
                except Exception: pass
                self._store.update(job_id,
                    status=JobStatus.RUNNING if sname == "RUNNING" else JobStatus.QUEUED,
                    queue_position=pos)
        except Exception as exc:
            logger.error("Poll failed for IBM job %s: %s", job.ibm_job_id, exc)
        return self._store.get(job_id)

    # ── internals ──────────────────────────────────────────────────────────────

    def _transpile(self, circuit, backend, level: int) -> QuantumCircuit:
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            return generate_preset_pass_manager(backend=backend, optimization_level=level).run(circuit)
        except Exception as e:
            logger.debug("Transpilation skipped: %s", e)
            return circuit

    def _run_sync(self, job: QuantumJob, backend, circuit, shots: int) -> None:
        self._store.update(job.job_id, status=JobStatus.RUNNING)
        try:
            result = self._execute(backend, circuit, shots, job.num_qubits)
            self._store.update(job.job_id, status=JobStatus.COMPLETED,
                               completed_at=_now(), result=result)
            logger.info("Job %s completed on %s", job.job_id, job.backend_name)
        except Exception as exc:
            self._store.update(job.job_id, status=JobStatus.FAILED, error=str(exc))
            logger.error("Job %s failed: %s", job.job_id, exc)

    def _submit_async(self, job: QuantumJob, backend, circuit, shots: int) -> None:
        try:
            from qiskit_ibm_runtime import Sampler
            ibm_job = Sampler(backend).run(circuit, shots=shots)
            pos = None
            try: pos = ibm_job.queue_position()
            except Exception: pass
            self._store.update(job.job_id, status=JobStatus.QUEUED,
                               ibm_job_id=ibm_job.job_id(), queue_position=pos)
            logger.info("Job %s queued on IBM as %s", job.job_id, ibm_job.job_id())
        except Exception as exc:
            self._store.update(job.job_id, status=JobStatus.FAILED, error=str(exc))

    def _execute(self, backend, circuit, shots: int, n_qubits: int) -> dict:
        # Primary: IBM Runtime Sampler (works for both Aer and IBM backends)
        try:
            from qiskit_ibm_runtime import Sampler
            result = Sampler(backend).run(circuit, shots=shots).result()
            return {"counts": self._extract_counts(result, n_qubits, shots), "shots": shots}
        except Exception as e1:
            logger.debug("Sampler path failed (%s), trying direct Aer", e1)
        # Fallback: direct Aer execute
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile
            sim = AerSimulator()
            counts = sim.run(transpile(circuit, sim), shots=shots).result().get_counts()
            return {"counts": counts, "shots": shots}
        except Exception as e2:
            raise RuntimeError(f"Execution failed.\nSampler: {e1}\nAer: {e2}")

    @staticmethod
    def _extract_counts(result, n_qubits: int, shots: int) -> dict:
        try:
            return {
                format(s, f"0{n_qubits}b"): max(1, int(p * shots))
                for s, p in result.quasi_dists[0].items() if p > 0
            }
        except Exception:
            return dict(result)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
