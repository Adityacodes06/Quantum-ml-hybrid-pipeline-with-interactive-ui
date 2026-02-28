"""
scripts/benchmark.py — compare simulator vs noisy vs real (optional).

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --include-real
    python scripts/benchmark.py --shots 2048
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run(include_real=False, shots=1024):
    from core.circuit_factory import build_bell_state, build_variational_bottleneck
    from core.executor import QuantumExecutor
    from core.job_store import JobStore
    from core.quantum_backend import BackendManager, BackendMode
    from training.loss import tv_distance
    from config.settings import settings

    mgr      = BackendManager(ibm_token=settings.ibm_quantum_token)
    executor = QuantumExecutor(mgr, JobStore())
    circuits = {
        "Bell State":       build_bell_state(),
        "Variational 2Q":   build_variational_bottleneck(2, [1.2,0.5], [0.8,1.3]),
    }
    modes = [BackendMode.SIMULATOR, BackendMode.NOISY_SIMULATOR]
    if include_real: modes.append(BackendMode.REAL)

    print("\n" + "="*68)
    print("  QUANTUM BACKEND BENCHMARK")
    print("="*68)

    for name, circuit in circuits.items():
        print(f"\n▶  {name}  ({circuit.num_qubits}q, depth {circuit.depth()})")
        print("-"*68)
        results = {}
        for m in modes:
            label = m.value.upper().replace("_"," ")
            print(f"  {label:<22}", end=" ", flush=True)
            t0  = time.time()
            job = executor.run(circuit, mode=m, shots=shots)
            elapsed = time.time()-t0
            if job.result:
                counts = job.result["counts"]
                total  = sum(counts.values())
                top    = sorted(counts.items(), key=lambda x:-x[1])[:4]
                dist   = "  ".join(f"|{s}⟩:{c/total:.2f}" for s,c in top)
                print(f"✓  {elapsed:.2f}s   {dist}")
                results[m] = counts
            else:
                print(f"✗  {job.error}")

        if BackendMode.SIMULATOR in results and BackendMode.NOISY_SIMULATOR in results:
            tvd = tv_distance(results[BackendMode.SIMULATOR], results[BackendMode.NOISY_SIMULATOR])
            ok  = "✓ within tolerance" if tvd < 0.15 else "⚠ high noise"
            print(f"\n  TVD (sim ↔ noisy): {tvd:.4f}  {ok}")

        if include_real and BackendMode.REAL in results and BackendMode.SIMULATOR in results:
            tvd = tv_distance(results[BackendMode.SIMULATOR], results[BackendMode.REAL])
            ok  = "✓ within tolerance" if tvd < 0.2 else "⚠ high noise"
            print(f"  TVD (sim ↔ real):  {tvd:.4f}  {ok}")

    print("\n" + "="*68 + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--include-real", action="store_true")
    p.add_argument("--shots", type=int, default=1024)
    args = p.parse_args()
    if args.include_real:
        from config.settings import settings
        if not settings.ibm_quantum_token:
            print("--include-real requires IBM_QUANTUM_TOKEN in .env"); sys.exit(1)
    run(args.include_real, args.shots)

if __name__ == "__main__":
    main()
