"""api/routes/devices.py — device listing and health check"""
from __future__ import annotations
import logging
from fastapi import APIRouter, Depends
from api.dependencies import get_backend_manager, get_job_store
from api.schemas import DeviceInfo, HealthResponse
from core.job_store import JobStore
from core.quantum_backend import BackendManager
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/devices", response_model=list[DeviceInfo], summary="List IBM Quantum backends")
async def list_devices(manager: BackendManager = Depends(get_backend_manager)):
    """Returns IBM backends with queue depth. Empty list if no token configured."""
    return [DeviceInfo(**d) for d in manager.list_devices() if "error" not in d]


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health(
    manager: BackendManager = Depends(get_backend_manager),
    store: JobStore = Depends(get_job_store),
):
    ibm_ok = bool(settings.ibm_quantum_token)
    return HealthResponse(
        status="ok",
        ibm_token_configured=ibm_ok,
        ibm_reachable=manager.ibm_available() if ibm_ok else False,
        job_count=store.count(),
        available_modes=(
            ["simulator","noisy_simulator","real"] if ibm_ok
            else ["simulator","noisy_simulator"]
        ),
    )
