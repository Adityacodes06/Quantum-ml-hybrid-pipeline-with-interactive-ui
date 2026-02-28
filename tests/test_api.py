"""tests/test_api.py — FastAPI endpoint integration tests."""
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_executor, get_job_store
from core.executor import QuantumExecutor
from core.job_store import JobStore
from core.quantum_backend import BackendManager


@pytest.fixture
def client():
    store    = JobStore()
    executor = QuantumExecutor(BackendManager(), store)
    app.dependency_overrides[get_executor]  = lambda: executor
    app.dependency_overrides[get_job_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestRoot:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert "available_modes" in d
        assert "simulator" in d["available_modes"]


class TestRun:
    def test_simulator(self, client):
        r = client.post("/run", json={
            "input_data": [1.0, 0.5], "thetas": [0.8, 1.3],
            "backend_mode": "simulator", "shots": 256,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "completed"
        assert d["result"] is not None
        assert "counts" in d["result"]

    def test_noisy_simulator(self, client):
        r = client.post("/run", json={
            "input_data": [1.0, 0.5], "thetas": [0.8, 1.3],
            "backend_mode": "noisy_simulator", "shots": 256,
        })
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_three_qubits(self, client):
        r = client.post("/run", json={
            "input_data": [1.0,0.5,0.7], "thetas": [0.8,1.3,0.4],
            "backend_mode": "simulator",
        })
        assert r.status_code == 200
        assert r.json()["num_qubits"] == 3

    def test_length_mismatch_422(self, client):
        r = client.post("/run", json={
            "input_data": [1.0,0.5], "thetas": [0.8],
            "backend_mode": "simulator",
        })
        assert r.status_code == 422

    def test_bad_mode_422(self, client):
        r = client.post("/run", json={
            "input_data": [1.0], "thetas": [0.8],
            "backend_mode": "quantum_mainframe",
        })
        assert r.status_code == 422

    def test_shots_over_max_422(self, client):
        r = client.post("/run", json={
            "input_data": [1.0], "thetas": [0.8],
            "backend_mode": "simulator", "shots": 999_999,
        })
        assert r.status_code == 422


class TestNamedCircuit:
    def test_bell(self, client):
        r = client.post("/run/circuit", json={"circuit_type":"bell","backend_mode":"simulator","shots":256})
        assert r.status_code == 200
        assert r.json()["num_qubits"] == 2

    def test_ghz(self, client):
        r = client.post("/run/circuit", json={"circuit_type":"ghz","n_qubits":4,"backend_mode":"simulator"})
        assert r.status_code == 200
        assert r.json()["num_qubits"] == 4

    def test_variational(self, client):
        r = client.post("/run/circuit", json={"circuit_type":"variational","n_qubits":3,"backend_mode":"simulator"})
        assert r.status_code == 200
        assert r.json()["num_qubits"] == 3


class TestJobs:
    def _run(self, client):
        r = client.post("/run", json={
            "input_data": [1.0,0.5], "thetas": [0.8,1.3], "backend_mode": "simulator",
        })
        return r.json()["job_id"]

    def test_list(self, client):
        self._run(client)
        r = client.get("/jobs")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_get_job(self, client):
        jid = self._run(client)
        r = client.get(f"/job/{jid}")
        assert r.status_code == 200
        assert r.json()["job_id"] == jid

    def test_missing_404(self, client):
        assert client.get("/job/doesnotexist").status_code == 404

    def test_delete(self, client):
        jid = self._run(client)
        assert client.delete(f"/job/{jid}").status_code == 200
        assert client.get(f"/job/{jid}").status_code == 404

    def test_clear_all(self, client):
        self._run(client); self._run(client)
        r = client.delete("/jobs")
        assert r.status_code == 200
        assert r.json()["cleared"] >= 2
