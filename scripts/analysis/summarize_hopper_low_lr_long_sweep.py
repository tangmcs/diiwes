#!/usr/bin/env python3
"""Strict validator for the low-rate, 2,000-update Hopper Hessian sweep.

The implementation reuses the coordinate-level validation and paired summary
logic from ``summarize_hopper_hessian_no_trust`` while replacing only the
locked grid. Prefix-horizon analysis is performed after this full-run audit.
"""

from __future__ import annotations

from typing import Sequence

if __package__:
    from . import summarize_hopper_hessian_no_trust as base
else:
    import summarize_hopper_hessian_no_trust as base


INITIAL_LEARNING_RATES = (0.1, 0.25, 0.5, 1.0, 2.0)
EXPECTED_ITERATIONS = 2000


def configure_locked_grid() -> None:
    base.LR_SCHEDULES = ("inverse_sqrt",)
    base.INITIAL_LEARNING_RATES = INITIAL_LEARNING_RATES
    base.ALPHA0S = INITIAL_LEARNING_RATES
    base.EXPECTED_ITERATIONS = EXPECTED_ITERATIONS


def main(argv: Sequence[str] | None = None) -> int:
    configure_locked_grid()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
