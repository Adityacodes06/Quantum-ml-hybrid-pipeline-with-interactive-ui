"""
scripts/save_ibm_credentials.py
Run once to save your IBM Quantum token to disk.

Usage:
    python scripts/save_ibm_credentials.py --token YOUR_TOKEN
    python scripts/save_ibm_credentials.py          # reads IBM_QUANTUM_TOKEN from .env
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", default=None)
    p.add_argument("--channel", default="ibm_quantum", choices=["ibm_quantum","ibm_cloud"])
    args = p.parse_args()

    token = args.token
    if not token:
        from config.settings import settings
        token = settings.ibm_quantum_token
    if not token:
        print("No token provided.\n  python scripts/save_ibm_credentials.py --token YOUR_TOKEN")
        sys.exit(1)

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel=args.channel, token=token, overwrite=True)
        print("✅  Credentials saved.")
        service  = QiskitRuntimeService(channel=args.channel)
        backends = service.backends(operational=True, simulator=False)
        print(f"✅  Connected — {len(backends)} real backend(s):")
        for b in backends[:5]:
            print(f"     • {b.name:<22} {b.num_qubits}q  queue: {b.status().pending_jobs}")
    except ImportError:
        print("qiskit-ibm-runtime not installed: pip install qiskit-ibm-runtime")
        sys.exit(1)
    except Exception as e:
        print(f"Failed: {e}"); sys.exit(1)

if __name__ == "__main__":
    main()
