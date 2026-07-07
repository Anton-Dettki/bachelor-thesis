"""Generate docker-compose.yml and optional dev overlay with live reload."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.dataset import participant_ids  # noqa: E402

DATA_DIR = ROOT / "data"
COMPOSE_PATH = ROOT / "docker-compose.yml"
DEV_COMPOSE_PATH = ROOT / "docker-compose.dev.yml"

RELOAD_COMMANDS = {
    "server": (
        "uvicorn fpm.server:app --host 0.0.0.0 --port 8000 "
        "--reload --reload-dir /app/fpm --reload-dir /app/shared --reload-dir /app/CASAS2"
    ),
    "client": (
        "uvicorn fpm.client:app --host 0.0.0.0 --port 8000 "
        "--reload --reload-dir /app/fpm --reload-dir /app/shared --reload-dir /app/CASAS2"
    ),
}


def service_name(participant: str) -> str:
    return f"client-{participant}"


def render(participants: list[str]) -> str:
    clients = ",".join(
        f"http://{service_name(participant)}:8000" for participant in participants
    )
    lines = [
        "services:",
        "  server:",
        "    build: .",
        '    command: uvicorn fpm.server:app --host 0.0.0.0 --port 8000',
        "    ports:",
        '      - "8080:8000"',
        "    volumes:",
        "      - grouped-outputs:/app/fpm/outputs",
        "    environment:",
        "      GROUPED_OUTPUT_DIR: /app/fpm/outputs/grouped",
        f'      CLIENTS: "{clients}"',
        "      DATA_DIR: data",
        "    depends_on:",
    ]
    for participant in participants:
        lines.append(f"      - {service_name(participant)}")

    for participant in participants:
        name = service_name(participant)
        lines.extend(
            [
                f"  {name}:",
                "    build: .",
                '    command: uvicorn fpm.client:app --host 0.0.0.0 --port 8000',
                "    environment:",
                f"      PARTICIPANT: {participant}",
                "      DATA_DIR: data",
            ]
        )

    lines.extend(
        [
            "",
            "volumes:",
            "  grouped-outputs:",
        ]
    )
    return "\n".join(lines) + "\n"


def render_dev_overlay(participants: list[str]) -> str:
    lines = [
        "# Dev overlay: mount source code and auto-reload uvicorn on changes.",
        "# Usage: docker compose -f docker-compose.yml -f docker-compose.dev.yml up",
        "# Rebuild only when requirements.txt or the Dockerfile changes.",
        "services:",
        "  server:",
        f"    command: {RELOAD_COMMANDS['server']}",
        "    volumes:",
        "      - grouped-outputs:/app/fpm/outputs",
        "      - ./fpm:/app/fpm",
        "      - ./shared:/app/shared",
        "      - ./CASAS2:/app/CASAS2",
        "      - ./data:/app/data",
    ]

    for participant in participants:
        name = service_name(participant)
        lines.extend(
            [
                f"  {name}:",
                f"    command: {RELOAD_COMMANDS['client']}",
                "    volumes:",
                "      - ./fpm:/app/fpm",
                "      - ./shared:/app/shared",
                "      - ./CASAS2:/app/CASAS2",
                "      - ./data:/app/data",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    participants = participant_ids(DATA_DIR)
    if not participants:
        raise SystemExit(f"No participants found in {DATA_DIR}")

    COMPOSE_PATH.write_text(render(participants), encoding="utf-8")
    DEV_COMPOSE_PATH.write_text(render_dev_overlay(participants), encoding="utf-8")
    print(f"Wrote {COMPOSE_PATH} with {len(participants)} clients")
    print(f"Wrote {DEV_COMPOSE_PATH} for live-reload development")


if __name__ == "__main__":
    main()
