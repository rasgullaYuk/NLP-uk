"""
Runs application initialization/migration routines for deployment.
"""

from __future__ import annotations

from scripts.init_environment import initialize_environment


def run_migrations() -> None:
    initialize_environment()


if __name__ == "__main__":
    run_migrations()
