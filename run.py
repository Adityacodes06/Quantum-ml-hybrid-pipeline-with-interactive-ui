"""
run.py — the recommended way to start the API.

Avoids the RuntimeWarning caused by `python -m api.main` on Windows.

Usage:
    python run.py                # development (auto-reload on)
    python run.py --no-reload    # production
    python run.py --port 8080    # custom port
"""
import argparse
import uvicorn
from config.settings import settings


def main():
    p = argparse.ArgumentParser(description="Quantum ML API server")
    p.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    p.add_argument("--port",  type=int, default=settings.api_port)
    p.add_argument("--host",  type=str, default=settings.api_host)
    args = p.parse_args()

    reload = not args.no_reload
    print(f"\n  ⚛  Quantum ML API")
    print(f"     http://localhost:{args.port}")
    print(f"     Docs   → http://localhost:{args.port}/docs")
    print(f"     Health → http://localhost:{args.port}/health")
    print(f"     UI     → open frontend/index.html in your browser")
    print(f"     Reload: {reload}\n")

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=reload)


if __name__ == "__main__":
    main()
