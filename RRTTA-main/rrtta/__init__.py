"""RRTTA: reliability-routed test-time adaptation."""

from .method import RRTTA, route_by_reliability, setup_rrtta

__all__ = ["RRTTA", "route_by_reliability", "setup_rrtta"]
