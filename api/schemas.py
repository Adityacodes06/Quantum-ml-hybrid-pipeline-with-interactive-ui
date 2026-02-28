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

    @field_validator("thetas")
    @classmethod
    def lengths_match(cls, v, info):
        inp = info.data.get("input_data", [])
        if inp and len(v) != len(inp):
            raise ValueError(f"thetas length ({len(v)}) must equal input_data length ({len(inp)})")
        return v


class CircuitTypeRequest(BaseModel):
    circuit_type: Literal["bell","ghz","variational"] = "bell"
    n_qubits: int = Field(2, ge=2, le=settings.max_qubits)
    backend_mode: Literal["simulator","noisy_simulator","real"] = "simulator"
    shots: int = Field(settings.default_shots, ge=1, le=settings.max_shots)
    async_mode: bool = False
    preferred_device: Optional[str] = None


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


class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
