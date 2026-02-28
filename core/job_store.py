"""
core/job_store.py — thread-safe in-memory job store with optional Redis backend.
"""
from __future__ import annotations
import json, logging, threading, uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class QuantumJob:
    job_id: str
    backend_mode: str
    backend_name: str
    num_qubits: int
    circuit_depth: int
    circuit_name: str
    shots: int
    optimization_level: int
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict]      = None
    error: Optional[str]        = None
    ibm_job_id: Optional[str]   = None
    queue_position: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)

    @classmethod
    def create(cls, **kwargs) -> "QuantumJob":
        return cls(
            job_id=uuid.uuid4().hex[:8],
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )


class JobStore:
    """Thread-safe in-memory store. Optional Redis for persistence across restarts."""

    def __init__(self, redis_url: Optional[str] = None, max_jobs: int = 500):
        self._jobs: Dict[str, QuantumJob] = {}
        self._lock = threading.Lock()
        self._max  = max_jobs
        self._redis = None
        if redis_url:
            self._connect_redis(redis_url)

    def save(self, job: QuantumJob) -> None:
        with self._lock:
            self._evict()
            self._jobs[job.job_id] = job
            self._rsave(job)

    def get(self, job_id: str) -> Optional[QuantumJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[QuantumJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in kwargs.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            self._rsave(job)
            return job

    def list_all(self) -> List[QuantumJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def list_by_status(self, status: JobStatus) -> List[QuantumJob]:
        with self._lock:
            return [j for j in self._jobs.values() if j.status == status]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            existed = job_id in self._jobs
            self._jobs.pop(job_id, None)
            if self._redis:
                try: self._redis.delete(f"qjob:{job_id}")
                except Exception: pass
            return existed

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    # ── internals ──────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        if len(self._jobs) < self._max:
            return
        terminal = sorted(
            [j for j in self._jobs.values() if j.is_terminal()],
            key=lambda j: j.created_at
        )
        for j in terminal[:max(1, len(terminal)//4)]:
            del self._jobs[j.job_id]

    def _rsave(self, job: QuantumJob) -> None:
        if self._redis:
            try:
                self._redis.set(f"qjob:{job.job_id}", json.dumps(job.to_dict()), ex=86400)
            except Exception as e:
                logger.warning("Redis write failed: %s", e)

    def _connect_redis(self, url: str) -> None:
        try:
            import redis
            self._redis = redis.from_url(url, decode_responses=True)
            self._redis.ping()
            logger.info("Job store connected to Redis")
            for key in self._redis.keys("qjob:*"):
                raw = self._redis.get(key)
                if raw:
                    d = json.loads(raw)
                    self._jobs[d["job_id"]] = QuantumJob(**{**d,"status":JobStatus(d["status"])})
        except Exception as e:
            logger.warning("Redis unavailable (%s) — using in-memory only", e)
            self._redis = None
