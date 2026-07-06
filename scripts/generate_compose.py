"""Generate docker-compose.yml with one federated client per participant."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpm.dataset import participant_ids  # noqa: E402

DATA_DIR = ROOT / "data"
COMPOSE_PATH = ROOT / "docker-compose.yml"


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
        "      - ./fpm/outputs:/app/fpm/outputs",
        "    environment:",
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

    return "\n".join(lines) + "\n"


def main() -> None:
    participants = participant_ids(DATA_DIR)
    if not participants:
        raise SystemExit(f"No participants found in {DATA_DIR}")
    COMPOSE_PATH.write_text(render(participants), encoding="utf-8")
    print(f"Wrote {COMPOSE_PATH} with {len(participants)} clients")


if __name__ == "__main__":
    main()
