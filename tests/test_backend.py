"""tests/test_backend.py — backend + executor tests using local Aer only."""
import pytest
from core.circuit_factory import build_bell_state, build_variational_bottleneck
from core.executor import QuantumExecutor
from core.job_store import JobStatus, JobStore, QuantumJob
from core.quantum_backend import BackendManager, BackendMode


@pytest.fixture
def manager(): return BackendManager()

@pytest.fixture
def store():    return JobStore()

@pytest.fixture
def executor(manager, store): return QuantumExecutor(manager, store)


class TestBackendManager:
    def test_simulator_returns(self, manager):
        backend, name = manager.get(BackendMode.SIMULATOR)
        assert backend is not None
        assert "aer" in name.lower()

    def test_noisy_returns(self, manager):
        backend, name = manager.get(BackendMode.NOISY_SIMULATOR)
        assert backend is not None
        assert "noisy" in name.lower()

    def test_real_without_token_raises(self, manager):
        with pytest.raises(RuntimeError, match="IBM Quantum token"):
            manager.get(BackendMode.REAL)

    def test_ibm_not_available_without_token(self, manager):
        assert not manager.ibm_available()

    def test_list_devices_empty_without_token(self, manager):
        assert manager.list_devices() == []


class TestJobStore:
    def _job(self):
        return QuantumJob.create(
            backend_mode="simulator", backend_name="aer_simulator",
            num_qubits=2, circuit_depth=5, circuit_name="test",
            shots=1024, optimization_level=1,
        )

    def test_save_get(self, store):
        job = self._job()
        store.save(job)
        assert store.get(job.job_id) is not None

    def test_update(self, store):
        job = self._job(); store.save(job)
        updated = store.update(job.job_id, status=JobStatus.COMPLETED)
        assert updated.status == JobStatus.COMPLETED

    def test_delete(self, store):
        job = self._job(); store.save(job)
        assert store.delete(job.job_id)
        assert store.get(job.job_id) is None

    def test_count(self, store):
        store.clear()
        for _ in range(3):
            store.save(self._job())
        assert store.count() == 3

    def test_list_by_status(self, store):
        store.clear()
        job = self._job(); store.save(job)
        store.update(job.job_id, status=JobStatus.COMPLETED)
        completed = store.list_by_status(JobStatus.COMPLETED)
        assert any(j.job_id == job.job_id for j in completed)


class TestExecutor:
    def test_bell_completes(self, executor):
        job = executor.run(build_bell_state(), mode=BackendMode.SIMULATOR, shots=512)
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert "counts" in job.result

    def test_bell_distribution(self, executor):
        job = executor.run(build_bell_state(), mode=BackendMode.SIMULATOR, shots=2048)
        counts = job.result["counts"]
        total  = sum(counts.values())
        for state in ["00", "11"]:
            prob = counts.get(state, 0) / total
            assert 0.3 < prob < 0.7, f"|{state}⟩ prob {prob:.3f} out of range"

    def test_variational_completes(self, executor):
        qc  = build_variational_bottleneck(3, [1.0,0.5,0.7], [0.8,1.3,0.4])
        job = executor.run(qc, mode=BackendMode.SIMULATOR, shots=256)
        assert job.status == JobStatus.COMPLETED

    def test_noisy_completes(self, executor):
        job = executor.run(build_bell_state(), mode=BackendMode.NOISY_SIMULATOR, shots=512)
        assert job.status == JobStatus.COMPLETED

    def test_job_stored(self, executor, store):
        job = executor.run(build_bell_state(), mode=BackendMode.SIMULATOR, shots=256)
        assert store.get(job.job_id) is not None
