"""api/routes/jobs.py — job management endpoints"""
from __future__ import annotations
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from api.dependencies import get_executor, get_job_store
from api.schemas import JobResponse
from core.executor import QuantumExecutor
from core.job_store import JobStatus, JobStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/job", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse, summary="Get job status and result")
async def get_job(job_id: str, executor: QuantumExecutor = Depends(get_executor)):
    """Fetch job. For async IBM jobs, triggers a live status poll."""
    job = executor.fetch_result(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return JobResponse(**job.to_dict())


@router.get("s", response_model=List[JobResponse], summary="List all jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter: pending|queued|running|completed|failed"),
    store: JobStore = Depends(get_job_store),
):
    if status:
        try:
            jobs = store.list_by_status(JobStatus(status))
        except ValueError:
            raise HTTPException(400, f"Invalid status '{status}'")
    else:
        jobs = store.list_all()
    return [JobResponse(**j.to_dict()) for j in jobs]


@router.delete("/{job_id}", summary="Delete a job")
async def delete_job(job_id: str, store: JobStore = Depends(get_job_store)):
    if not store.delete(job_id):
        raise HTTPException(404, f"Job '{job_id}' not found")
    return {"deleted": job_id}


@router.delete("s", summary="Clear all jobs")
async def clear_jobs(store: JobStore = Depends(get_job_store)):
    n = store.count()
    store.clear()
    return {"cleared": n}
