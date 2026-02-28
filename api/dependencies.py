"""api/dependencies.py — singleton services via FastAPI dependency injection."""
from functools import lru_cache
from config.settings import settings
from core.executor import QuantumExecutor
from core.job_store import JobStore
from core.quantum_backend import BackendManager


@lru_cache(maxsize=1)
def get_backend_manager() -> BackendManager:
    return BackendManager(
        ibm_token=settings.ibm_quantum_token,
        channel=settings.ibm_channel,
        ibm_instance=settings.ibm_instance,
    )


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    redis_url = getattr(settings, "redis_url", None)
    return JobStore(redis_url=redis_url)


@lru_cache(maxsize=1)
def get_executor() -> QuantumExecutor:
    return QuantumExecutor(get_backend_manager(), get_job_store())
