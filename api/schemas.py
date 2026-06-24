"""api/schemas.py — all Pydantic request/response models in one place."""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from config.settings import settings


class RunCircuitRequest(BaseModel):
    input_data: List[float] = Field(..., min_length=1, max_length=settings.max_qubits,
        description="Classical input features, one per qubit (values in [0, 2π])",
        examples=[[1.2, 0.5, 0.9]])
    thetas: List[float] = Field(..., min_length=1, max_length=settings.max_qubits,
        description="Variational parameters, one per qubit (values in [0, 2π])",
        examples=[[0.8, 1.3, 0.4]])
    backend_mode: Literal["simulator","noisy_simulator","real"] = "simulator"
    shots: int = Field(settings.default_shots, ge=1, le=settings.max_shots)
    optimization_level: int = Field(settings.default_optimization_level, ge=0, le=3)
    preferred_device: Optional[str] = None
    async_mode: bool = False
    seed: Optional[int] = Field(None, description="Random seed for simulator reproducibility")
    noise_rate: float = Field(0.01, ge=0.0, le=1.0, description="Depolarizing error rate for noisy_simulator")
    encode_gate: Literal["rx", "ry", "rz"] = "ry"
    entangle_type: Literal["linear", "circular", "full", "none"] = "linear"
    var_gate: Literal["rx", "ry", "rz"] = "ry"

    @field_validator("thetas")
    @classmethod
    def lengths_match(cls, v, info):
        inp = info.data.get("input_data", [])
        if inp and len(v) != len(inp):
            raise ValueError(f"thetas length ({len(v)}) must equal input_data length ({len(inp)})")
        return v


class CircuitTypeRequest(BaseModel):
    circuit_type: str = Field("bell", description="bell, ghz, variational, qft, w_state, etc.")
    n_qubits: int = Field(2, ge=1, le=settings.max_qubits)
    backend_mode: Literal["simulator","noisy_simulator","real"] = "simulator"
    shots: int = Field(settings.default_shots, ge=1, le=settings.max_shots)
    async_mode: bool = False
    preferred_device: Optional[str] = None


class TrainRequest(BaseModel):
    """Request model for server-side variational training."""
    input_data: List[float] = Field(..., min_length=1, max_length=settings.max_qubits,
        description="Classical input features, one per qubit")
    thetas: Optional[List[float]] = Field(None,
        description="Initial theta parameters. Random if omitted.")
    learning_rate: float = Field(settings.learning_rate, gt=0, le=1.0)
    max_iterations: int = Field(50, ge=1, le=settings.max_training_iterations)
    shots: int = Field(256, ge=1, le=settings.max_shots)
    convergence_threshold: float = Field(1e-4, gt=0)
    seed: Optional[int] = Field(None, description="Random seed for simulator reproducibility")
    optimizer: Literal["gd", "cobyla"] = Field("gd", description="Optimizer algorithm to use")
    encode_gate: Literal["rx", "ry", "rz"] = "ry"
    entangle_type: Literal["linear", "circular", "full", "none"] = "linear"
    var_gate: Literal["rx", "ry", "rz"] = "ry"

    @field_validator("thetas")
    @classmethod
    def thetas_match_input(cls, v, info):
        if v is not None:
            inp = info.data.get("input_data", [])
            if inp and len(v) != len(inp):
                raise ValueError(f"thetas length ({len(v)}) must equal input_data length ({len(inp)})")
        return v


class QasmRunRequest(BaseModel):
    """Request model for running raw OpenQASM."""
    qasm_str: str = Field(..., description="OpenQASM 2.0 string")
    backend_mode: Literal["simulator","noisy_simulator","real"] = "simulator"
    shots: int = Field(settings.default_shots, ge=1, le=settings.max_shots)
    optimization_level: int = Field(settings.default_optimization_level, ge=0, le=3)
    seed: Optional[int] = None
    noise_rate: float = 0.01


class TrainResponse(BaseModel):
    """Response model for training results."""
    losses: List[float]
    best_loss: float
    best_thetas: List[float]
    iterations_run: int
    converged: bool
    elapsed_seconds: float


class JobResponse(BaseModel):
    job_id: str
    backend_mode: str
    backend_name: str
    num_qubits: int
    circuit_depth: int
    circuit_name: str
    shots: int
    optimization_level: int
    status: str
    created_at: str
    completed_at: Optional[str]    = None
    result: Optional[Dict[str,Any]] = None
    error: Optional[str]           = None
    ibm_job_id: Optional[str]      = None
    queue_position: Optional[int]  = None
    model_config = {"from_attributes": True}


class DeviceInfo(BaseModel):
    name: str
    num_qubits: int
    pending_jobs: int
    status: str
    simulator: bool


class HealthResponse(BaseModel):
    status: Literal["ok","degraded"]
    api_version: str = "1.0.0"
    ibm_token_configured: bool
    ibm_reachable: bool
    job_count: int
    available_modes: List[str] = ["simulator","noisy_simulator"]


class DensityMatrixRequest(BaseModel):
    """Request model for generating density matrix from a circuit."""
    input_data: List[float] = Field(..., min_length=1, max_length=settings.max_qubits,
        description="Classical input features, one per qubit")
    thetas: List[float] = Field(..., min_length=1, max_length=settings.max_qubits,
        description="Variational parameters, one per qubit")
    encode_gate: Literal["rx", "ry", "rz"] = "ry"
    entangle_type: Literal["linear", "circular", "full", "none"] = "linear"
    var_gate: Literal["rx", "ry", "rz"] = "ry"
    circuit_type: Optional[str] = Field(None, description="Named circuit type (overrides input_data/thetas if set)")
    n_qubits: Optional[int] = Field(None, ge=1, le=settings.max_qubits)

    @field_validator("thetas")
    @classmethod
    def lengths_match(cls, v, info):
        inp = info.data.get("input_data", [])
        if inp and len(v) != len(inp):
            raise ValueError(f"thetas length ({len(v)}) must equal input_data length ({len(inp)})")
        return v


class DensityMatrixResponse(BaseModel):
    """Response model for density matrix computation."""
    num_qubits: int
    dimension: int
    statevector: List[str] = Field(description="Statevector amplitudes as complex strings")
    density_matrix: List[List[str]] = Field(description="Density matrix entries as complex strings")
    probabilities: Dict[str, float] = Field(description="Measurement probabilities per basis state")
    purity: float = Field(description="Tr(ρ²) — 1.0 for pure states")
    von_neumann_entropy: float = Field(description="Von Neumann entropy S(ρ)")


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
