# Quantum ML — Hybrid Quantum-Classical Pipeline

> **No IBM account needed.** `simulator` and `noisy_simulator` run entirely locally.

---

## Quick Start (3 commands)

```bash
pip install -r requirements.txt
python run.py
# Open frontend/index.html in your browser
```

API docs: http://localhost:8000/docs

---

## Project Structure

```
quantum-ml/
├── core/
│   ├── quantum_backend.py   # BackendManager (Aer / noisy Aer / IBM Real)
│   ├── circuit_factory.py   # Circuit builders (variational, bell, ghz, etc.)
│   ├── executor.py          # Execution engine (transpile → run → store)
│   └── job_store.py         # Thread-safe job store (+ optional Redis)
├── api/
│   ├── main.py              # FastAPI app (lifespan, CORS, error handler)
│   ├── schemas.py           # Pydantic request/response models
│   ├── dependencies.py      # Singleton service injection
│   └── routes/
│       ├── circuits.py      # POST /run, POST /run/circuit
│       ├── jobs.py          # GET|DELETE /job(s)
│       └── devices.py       # GET /devices, GET /health
├── training/
│   ├── trainer.py           # Parameter shift gradient descent
│   └── loss.py              # Loss functions (EV, cross-entropy, TVD, KL)
├── frontend/
│   └── index.html           # Full interactive UI — open directly in browser
├── config/
│   └── settings.py          # All config via pydantic-settings + .env
├── tests/                   # pytest suite (no IBM token needed)
├── scripts/
│   ├── save_ibm_credentials.py
│   └── benchmark.py
├── run.py                   # ← START THE SERVER WITH THIS
└── requirements.txt
```

---

## Backend Modes

| Mode | Token needed | Speed | Noise |
|------|:-----------:|-------|-------|
| `simulator` | No | Instant | None |
| `noisy_simulator` | No | Instant | Depolarising + T1/T2 + readout |
| `real` | Yes | Minutes | Real IBM hardware |

### Local noise model (no token)
- Single-qubit gates: 0.1% depolarising
- CX gates: 1.0% depolarising
- Thermal relaxation: T1=50µs / T2=70µs
- Readout: 2% |0⟩→|1⟩, 5% |1⟩→|0⟩

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/run` | Run variational bottleneck circuit |
| `POST` | `/run/circuit` | Run bell / ghz / variational |
| `GET`  | `/job/{id}` | Get job (polls IBM if async) |
| `GET`  | `/jobs` | List all jobs |
| `DELETE` | `/job/{id}` | Delete job |
| `DELETE` | `/jobs` | Clear all jobs |
| `GET`  | `/devices` | List IBM backends (empty if no token) |
| `GET`  | `/health` | System status + available modes |

---

## UI Features (frontend/index.html)

- **▶ Run Circuit** — executes your qubit config against the selected backend
- **⚡ Quick Circuits** — one-click Bell, GHZ-3Q, GHZ-5Q, Variational, Benchmark All
- **🧠 Train** — live parameter shift gradient loop with real-time loss curve
- **📋 All Jobs** — full job history with status badges
- **Sidebar** — mode selector, shots/opt-level sliders, add/remove qubits
- **Top bar** — API URL input, live connection indicator

---

## Add IBM Real Hardware (Optional)

```bash
# 1. Get token at https://quantum.ibm.com
# 2. Add to .env:
IBM_QUANTUM_TOKEN=your_token_here

# 3. Save credentials once:
python scripts/save_ibm_credentials.py

# 4. Restart server — 'real' mode is now available
python run.py
```

---

## Tests

```bash
pytest tests/ -v
# All tests use local Aer — no IBM token required
```

---

## Optional: Redis Persistence

```bash
# Add to .env:
REDIS_URL=redis://localhost:6379

# Start Redis:
docker run -p 6379:6379 redis
```
