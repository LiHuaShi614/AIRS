#!/usr/bin/env python
from __future__ import annotations

import argparse

from rrtta.method import setup_rrtta
from rrtta.runner import add_common_args, add_rrtta_args, apply_rrtta_args, config_from_args, evaluate_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RRTTA on CIFAR10-C or CIFAR100-C.")
    add_common_args(parser, output_name="rrtta")
    add_rrtta_args(parser)
    args = parser.parse_args()
    cfg = apply_rrtta_args(config_from_args(args), args)
    evaluate_method(cfg, setup_rrtta)


if __name__ == "__main__":
    main()
