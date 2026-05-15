"""
config/settings.py — single source of truth for all configuration.
Loads from .env automatically. IBM token is fully optional.
"""
from functools import lru_cache
from typing import Literal, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # IBM Quantum — optional, only needed for backend_mode='real'
    ibm_quantum_token: Optional[str] = None
    ibm_channel: Literal["ibm_quantum", "ibm_cloud"] = "ibm_quantum"
    ibm_instance: Optional[str] = None

    # Execution defaults
    default_shots: int = 1024
    default_optimization_level: int = 1
    max_shots: int = 20_000
    max_qubits: int = 10
    poll_interval_seconds: int = 5
    max_poll_attempts: int = 60

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    cors_origins: list[str] = ["*"]

    # Rate limiting
    rate_limit: str = "30/minute"

    # Training
    learning_rate: float = 0.01
    max_training_iterations: int = 100
    checkpoint_dir: str = "training/checkpoints"

    # Redis (optional)
    redis_url: Optional[str] = None

    # Logging
    log_level: str = "INFO"

    # ── validators ──────────────────────────────────────────────────────────

    @field_validator("api_port")
    @classmethod
    def port_in_range(cls, v):
        if not (1024 <= v <= 65535):
            raise ValueError(f"api_port must be 1024-65535, got {v}")
        return v

    @field_validator("default_shots")
    @classmethod
    def shots_positive(cls, v):
        if v < 1:
            raise ValueError(f"default_shots must be >= 1, got {v}")
        return v

    @field_validator("max_qubits")
    @classmethod
    def qubits_reasonable(cls, v):
        if not (1 <= v <= 50):
            raise ValueError(f"max_qubits must be 1-50, got {v}")
        return v

    @field_validator("learning_rate")
    @classmethod
    def lr_positive(cls, v):
        if v <= 0:
            raise ValueError(f"learning_rate must be > 0, got {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
