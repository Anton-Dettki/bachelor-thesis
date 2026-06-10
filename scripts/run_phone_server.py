#!/usr/bin/env python3
"""Launch a FastAPI phone server for one subject."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.event_log import DEFAULT_EVENT_LOG_DIR  # noqa: E402
from fpm.loader import SUBJECT_IDS  # noqa: E402
from fpm.phone import Phone  # noqa: E402
from fpm.prefix import DEFAULT_PREFIX_DIR  # noqa: E402
from fpm.server import create_phone_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FPM phone API for one subject (Phase D)."
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=SUBJECT_IDS,
        required=True,
        help="Subject id (1-7)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Listen port (default: 8000 + subject id, e.g. 8001)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address",
    )
    parser.add_argument(
        "--event-log-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help="Directory containing generated event logs",
    )
    parser.add_argument(
        "--prefix-dir",
        type=Path,
        default=DEFAULT_PREFIX_DIR,
        help="Directory containing prefix datasets for /predict/params",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = args.port if args.port is not None else 8000 + args.subject

    phone = Phone(args.subject, event_log_dir=args.event_log_dir)
    app = create_phone_app(phone, prefix_dir=args.prefix_dir)

    import uvicorn

    print(f"Serving {phone.subject_label} on http://{args.host}:{port}")
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
