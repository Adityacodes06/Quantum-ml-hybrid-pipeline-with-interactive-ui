"""Tests for core/job_store.py — thread-safe job store."""
import pytest
import threading
from core.job_store import JobStore, QuantumJob, JobStatus


@pytest.fixture
def store():
    return JobStore(max_jobs=10)


@pytest.fixture
def sample_job():
    return QuantumJob.create(
        backend_mode="simulator",
        backend_name="aer_simulator",
        num_qubits=2,
        circuit_depth=5,
        circuit_name="test_circuit",
        shots=1024,
        optimization_level=1,
    )


class TestQuantumJob:
    def test_create_generates_id(self, sample_job):
        assert len(sample_job.job_id) == 12  # 12 hex chars
        assert sample_job.status == JobStatus.PENDING

    def test_to_dict(self, sample_job):
        d = sample_job.to_dict()
        assert d["status"] == "pending"
        assert d["job_id"] == sample_job.job_id

    def test_is_terminal(self, sample_job):
        assert not sample_job.is_terminal()
        sample_job.status = JobStatus.COMPLETED
        assert sample_job.is_terminal()
        sample_job.status = JobStatus.FAILED
        assert sample_job.is_terminal()
        sample_job.status = JobStatus.RUNNING
        assert not sample_job.is_terminal()


class TestJobStore:
    def test_save_and_get(self, store, sample_job):
        store.save(sample_job)
        retrieved = store.get(sample_job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == sample_job.job_id

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_update(self, store, sample_job):
        store.save(sample_job)
        store.update(sample_job.job_id, status=JobStatus.COMPLETED)
        job = store.get(sample_job.job_id)
        assert job.status == JobStatus.COMPLETED

    def test_update_nonexistent(self, store):
        assert store.update("nonexistent", status=JobStatus.FAILED) is None

    def test_list_all(self, store):
        for i in range(3):
            job = QuantumJob.create(
                backend_mode="simulator", backend_name="aer",
                num_qubits=2, circuit_depth=1, circuit_name=f"test_{i}",
                shots=100, optimization_level=0,
            )
            store.save(job)
        assert len(store.list_all()) == 3

    def test_list_by_status(self, store, sample_job):
        store.save(sample_job)
        assert len(store.list_by_status(JobStatus.PENDING)) == 1
        assert len(store.list_by_status(JobStatus.COMPLETED)) == 0

    def test_delete(self, store, sample_job):
        store.save(sample_job)
        assert store.delete(sample_job.job_id) is True
        assert store.get(sample_job.job_id) is None
        assert store.delete(sample_job.job_id) is False

    def test_clear(self, store, sample_job):
        store.save(sample_job)
        store.clear()
        assert store.count() == 0

    def test_count(self, store, sample_job):
        assert store.count() == 0
        store.save(sample_job)
        assert store.count() == 1

    def test_eviction(self):
        """Eviction should remove oldest terminal jobs when at max capacity."""
        store = JobStore(max_jobs=3)
        for i in range(3):
            job = QuantumJob.create(
                backend_mode="simulator", backend_name="aer",
                num_qubits=2, circuit_depth=1, circuit_name=f"test_{i}",
                shots=100, optimization_level=0,
            )
            job.status = JobStatus.COMPLETED  # mark as terminal
            store.save(job)

        # Store is full, saving one more should evict
        new_job = QuantumJob.create(
            backend_mode="simulator", backend_name="aer",
            num_qubits=2, circuit_depth=1, circuit_name="new",
            shots=100, optimization_level=0,
        )
        store.save(new_job)
        assert store.count() <= 3

    def test_thread_safety(self, store):
        """Concurrent saves should not corrupt the store."""
        errors = []

        def save_jobs(start):
            try:
                for i in range(20):
                    job = QuantumJob.create(
                        backend_mode="simulator", backend_name="aer",
                        num_qubits=2, circuit_depth=1, circuit_name=f"t{start}_{i}",
                        shots=100, optimization_level=0,
                    )
                    store.save(job)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=save_jobs, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 5 threads × 20 jobs = 100, but max_jobs=10 so eviction should have kicked in
        assert store.count() <= 10
