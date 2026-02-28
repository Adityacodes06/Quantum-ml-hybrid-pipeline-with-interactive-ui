"""
core/quantum_backend.py — unified backend manager.
Handles Aer (clean), Aer (noisy local model), and real IBM Quantum hardware.
No FastAPI imports. IBM token is never required for simulator modes.
"""
from __future__ import annotations
import logging
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class BackendMode(str, Enum):
    SIMULATOR       = "simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    REAL            = "real"


class BackendManager:
    """
    Returns (backend_object, backend_name) for the requested mode.
    - Lazy IBM imports: qiskit-ibm-runtime only loaded when actually needed
    - IBM service cached after first init
    - Clear error messages on every failure path
    """

    def __init__(
        self,
        ibm_token: Optional[str] = None,
        channel: str = "ibm_quantum",
        ibm_instance: Optional[str] = None,
    ):
        self._ibm_token   = ibm_token
        self._channel     = channel
        self._ibm_instance = ibm_instance
        self._service     = None

    def get(self, mode: BackendMode, preferred_device: Optional[str] = None) -> Tuple[object, str]:
        if mode == BackendMode.SIMULATOR:
            return self._aer_clean()
        if mode == BackendMode.NOISY_SIMULATOR:
            return self._aer_noisy()
        if mode == BackendMode.REAL:
            return self._ibm_real(preferred_device)
        raise ValueError(f"Unknown BackendMode: {mode!r}")

    def list_devices(self) -> list[dict]:
        """Returns IBM backend metadata. Empty list (not exception) when unavailable."""
        try:
            service  = self._get_service()
            backends = service.backends(operational=True)
            return [
                {
                    "name":         b.name,
                    "num_qubits":   b.num_qubits,
                    "pending_jobs": b.status().pending_jobs,
                    "status":       b.status().status_msg,
                    "simulator":    b.configuration().simulator,
                }
                for b in backends
            ]
        except Exception as exc:
            logger.warning("Could not list IBM devices: %s", exc)
            return []

    def ibm_available(self) -> bool:
        try:
            self._get_service()
            return True
        except Exception:
            return False

    # ── backends ──────────────────────────────────────────────────────────────

    def _aer_clean(self) -> Tuple[object, str]:
        try:
            from qiskit_aer import AerSimulator
        except ImportError:
            raise RuntimeError("qiskit-aer not installed. Run: pip install qiskit-aer")
        return AerSimulator(), "aer_simulator"

    def _aer_noisy(self) -> Tuple[object, str]:
        """
        Local noise model — no IBM token required.
        Depolarising + T1/T2 thermal relaxation + readout errors
        tuned to match typical IBM superconducting qubit hardware.
        """
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import (
                NoiseModel, depolarizing_error,
                thermal_relaxation_error, ReadoutError,
            )

            nm = NoiseModel()

            # Single-qubit gate depolarising ~0.1%
            e1 = depolarizing_error(0.001, 1)
            nm.add_all_qubit_quantum_error(
                e1, ["u1","u2","u3","ry","rx","rz","h","x","y","z","s","t"]
            )

            # Two-qubit gate depolarising ~1%
            e2 = depolarizing_error(0.01, 2)
            nm.add_all_qubit_quantum_error(e2, ["cx","cz","swap"])

            # T1=50µs, T2=70µs thermal relaxation
            t1, t2 = 50_000, 70_000          # ns
            r1 = thermal_relaxation_error(t1, t2, 50)       # 1q gate ~50ns
            r2 = thermal_relaxation_error(t1, t2, 300).expand(
                 thermal_relaxation_error(t1, t2, 300))      # CX gate ~300ns
            nm.add_all_qubit_quantum_error(r1, ["u1","u2","u3","ry","rx","rz"])
            nm.add_all_qubit_quantum_error(r2, ["cx"])

            # Readout: 2% |0>→|1>, 5% |1>→|0>
            nm.add_all_qubit_readout_error(ReadoutError([[0.98, 0.02],[0.05, 0.95]]))

            logger.info("Noisy simulator ready (depolarising + T1/T2 + readout)")
            return AerSimulator(noise_model=nm), "noisy_sim[local_hardware_model]"

        except ImportError:
            raise RuntimeError("qiskit-aer not installed. Run: pip install qiskit-aer")
        except Exception as exc:
            logger.warning("Noisy sim build failed (%s) — falling back to clean Aer", exc)
            return self._aer_clean()

    def _ibm_real(self, preferred_device: Optional[str]) -> Tuple[object, str]:
        if not self._ibm_token:
            raise RuntimeError(
                "backend_mode='real' requires an IBM Quantum token.\n"
                "  1. Sign up free at https://quantum.ibm.com\n"
                "  2. Add IBM_QUANTUM_TOKEN=your_token to .env\n"
                "  3. Restart the API\n"
                "Use 'simulator' or 'noisy_simulator' in the meantime — both work locally."
            )
        service = self._get_service()
        if preferred_device:
            try:
                b = service.backend(preferred_device)
                return b, preferred_device
            except Exception:
                logger.warning("Device %r unavailable, using least_busy", preferred_device)
        b = service.least_busy(simulator=False, operational=True)
        logger.info("Using IBM backend: %s", b.name)
        return b, b.name

    def _get_service(self):
        if self._service is not None:
            return self._service
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            raise RuntimeError("qiskit-ibm-runtime not installed. Run: pip install qiskit-ibm-runtime")
        if self._ibm_token:
            QiskitRuntimeService.save_account(
                channel=self._channel,
                token=self._ibm_token,
                instance=self._ibm_instance,
                overwrite=True,
            )
        self._service = QiskitRuntimeService(channel=self._channel)
        return self._service
