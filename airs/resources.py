from __future__ import annotations


TRACE_NAMES = ("steady", "sudden_drop", "recovery", "cyclic")


def resource_availability(trace: str, batch_index: int, total_batches: int) -> float:
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    progress = min(1.0, max(0.0, batch_index / max(1, total_batches - 1)))
    if trace == "steady":
        return 1.0
    if trace == "sudden_drop":
        return 1.0 if progress < 0.5 else 0.2
    if trace == "recovery":
        if progress < 1.0 / 3.0:
            return 0.2
        if progress < 2.0 / 3.0:
            return 0.5
        return 1.0
    if trace == "cyclic":
        phase = int(progress * 12.0) % 4
        return (1.0, 0.5, 0.2, 0.5)[phase]
    raise ValueError(f"unsupported resource trace: {trace}")
