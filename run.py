"""Development server entry point.

Usage:
    python run.py            # default: host=0.0.0.0, port=8899, reload=True
    python run.py --no-reload
    python run.py --port 8899
"""

import argparse

from dotenv import load_dotenv

load_dotenv()  # inject ALL .env vars (including LANGCHAIN_*) into os.environ before uvicorn forks

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AIDD Agent backend")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8899, help="Bind port (default: 8899)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )
