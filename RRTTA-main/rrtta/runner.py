from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from robustbench.data import load_cifar10c
try:
    from robustbench.data import load_cifar100c
except ImportError:
    load_cifar100c = None
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT
DEFAULT_DATA_DIR = os.environ.get("RRTTA_DATA_DIR", str(REPO_ROOT / "data"))
DEFAULT_CKPT_DIR = os.environ.get("RRTTA_CKPT_DIR", str(REPO_ROOT / "checkpoints"))
DEFAULT_OUTPUT_ROOT = os.environ.get("RRTTA_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
DEFAULT_ARCH = ""
DEFAULT_ARCHS = {
    "cifar10": "Standard",
    "cifar100": "Hendrycks2020AugMix_ResNeXt",
}
DEFAULT_BATCH_SIZE = 200
DEFAULT_NUM_EX = 10000
DEFAULT_SEVERITY = 5
DEFAULT_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


@dataclass
class OptimConfig:
    method: str = "Adam"
    steps: int = 1
    lr: float = 1e-3
    beta: float = 0.9
    momentum: float = 0.9
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = True


@dataclass
class ReliabilityConfig:
    sce_alpha: float = 0.70
    sce_beta: float = 0.30
    reliability_high_threshold: float = 0.40
    reliability_low_threshold: float = 0.10
    max_update_fraction: float = 1.00


@dataclass
class RRTTAConfig:
    teacher_arch: str = ""
    teacher_momentum: float = 0.999
    adapt_batch_size: int = 16
    refine_label_margin: float = 0.02
    student_disagreement_weight: float = 0.25
    prediction_blend: float = 0.0
    augment_times: int = 4
    flip_p: float = 0.5
    brightness: float = 0.20
    contrast: float = 0.20
    noise_std: float = 0.005
    high_loss_weight: float = 1.0
    medium_loss_weight: float = 0.75
    low_loss_weight: float = 0.0
    low_conf_threshold: float = 0.50
    proto_momentum: float = 0.95
    proto_align_weight: float = 0.02
    proto_warmup_batches: int = 5
    proto_min_count: int = 2
    proto_high_only: bool = True


@dataclass
class RunConfig:
    method: str = "RRTTA"
    dataset: str = "cifar10"
    num_classes: int = 10
    arch: str = DEFAULT_ARCHS["cifar10"]
    ckpt_dir: str = DEFAULT_CKPT_DIR
    data_dir: str = DEFAULT_DATA_DIR
    output_root: str = DEFAULT_OUTPUT_ROOT
    run_id: str = ""
    batch_size: int = DEFAULT_BATCH_SIZE
    num_ex: int = DEFAULT_NUM_EX
    severity: int = DEFAULT_SEVERITY
    corruptions: Sequence[str] = field(default_factory=lambda: list(DEFAULT_CORRUPTIONS))
    seed: int = 1
    episodic: bool = False
    image_size: int = 32
    optim: OptimConfig = field(default_factory=OptimConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    rrtta: RRTTAConfig = field(default_factory=RRTTAConfig)


def default_arch(dataset: str) -> str:
    try:
        return DEFAULT_ARCHS[str(dataset).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {dataset}") from exc


def dataset_num_classes(dataset: str) -> int:
    dataset = str(dataset).lower()
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    raise ValueError(f"unsupported dataset: {dataset}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(run_dir: Path, method: str, run_id: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"rrtta_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(filename)s:%(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(run_dir / f"{method.lower()}_{run_id}.txt")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_corruption(
    dataset: str,
    num_ex: int,
    severity: int,
    data_dir: str,
    corruption: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dataset == "cifar10":
        loader = load_cifar10c
    elif load_cifar100c is None:
        raise RuntimeError("CIFAR100-C requires a newer RobustBench data loader.")
    else:
        loader = load_cifar100c
    return loader(num_ex, severity, data_dir, False, [corruption])


def load_corruption_dataset(cfg: RunConfig, corruption: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return load_corruption(cfg.dataset, cfg.num_ex, cfg.severity, cfg.data_dir, corruption)


def add_common_args(
    parser: argparse.ArgumentParser,
    output_name: str = "rrtta",
    default_num_ex: int = DEFAULT_NUM_EX,
) -> None:
    parser.add_argument("--dataset", choices=sorted(DEFAULT_ARCHS), default="cifar10")
    parser.add_argument("--arch", default=DEFAULT_ARCH, help="RobustBench model name")
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=str(Path(DEFAULT_OUTPUT_ROOT) / output_name))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-ex", type=int, default=default_num_ex)
    parser.add_argument("--severity", type=int, default=DEFAULT_SEVERITY)
    parser.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--episodic", action="store_true")


def add_rrtta_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--teacher-arch", default="")
    parser.add_argument("--teacher-momentum", type=float, default=0.999)
    parser.add_argument("--adapt-batch-size", type=int, default=16)
    parser.add_argument("--refine-label-margin", type=float, default=0.02)
    parser.add_argument("--student-disagreement-weight", type=float, default=0.25)
    parser.add_argument("--prediction-blend", type=float, default=0.0)
    parser.add_argument("--sce-alpha", type=float, default=0.70)
    parser.add_argument("--sce-beta", type=float, default=0.30)
    parser.add_argument("--reliability-high-threshold", type=float, default=0.40)
    parser.add_argument("--reliability-low-threshold", type=float, default=0.10)
    parser.add_argument("--max-update-fraction", type=float, default=1.00)
    parser.add_argument("--augment-times", type=int, default=4)
    parser.add_argument("--flip-p", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.20)
    parser.add_argument("--contrast", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--high-loss-weight", type=float, default=1.0)
    parser.add_argument("--medium-loss-weight", type=float, default=0.75)
    parser.add_argument("--low-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-conf-threshold", type=float, default=0.50)
    parser.add_argument("--proto-momentum", type=float, default=0.95)
    parser.add_argument("--proto-align-weight", type=float, default=0.02)
    parser.add_argument("--proto-warmup-batches", type=int, default=5)
    parser.add_argument("--proto-min-count", type=int, default=2)
    parser.add_argument("--proto-all-routes", action="store_true")


def config_from_args(args, method: str = "RRTTA") -> RunConfig:
    dataset = getattr(args, "dataset", "cifar10")
    cfg = RunConfig(method=method)
    cfg.dataset = dataset
    cfg.num_classes = dataset_num_classes(dataset)
    cfg.arch = getattr(args, "arch", "") or default_arch(dataset)
    cfg.ckpt_dir = args.ckpt_dir
    cfg.data_dir = args.data_dir
    cfg.output_root = args.output_root
    cfg.run_id = args.run_id
    cfg.batch_size = args.batch_size
    cfg.num_ex = args.num_ex
    cfg.severity = args.severity
    cfg.corruptions = args.corruptions
    cfg.seed = args.seed
    cfg.optim.lr = args.lr
    cfg.optim.steps = args.steps
    cfg.episodic = args.episodic
    return cfg


def apply_rrtta_args(cfg: RunConfig, args) -> RunConfig:
    cfg.image_size = args.image_size
    cfg.rrtta.teacher_arch = args.teacher_arch or cfg.arch
    cfg.rrtta.teacher_momentum = args.teacher_momentum
    cfg.rrtta.adapt_batch_size = args.adapt_batch_size
    cfg.rrtta.refine_label_margin = args.refine_label_margin
    cfg.rrtta.student_disagreement_weight = args.student_disagreement_weight
    cfg.rrtta.prediction_blend = args.prediction_blend
    cfg.reliability.sce_alpha = args.sce_alpha
    cfg.reliability.sce_beta = args.sce_beta
    cfg.reliability.reliability_high_threshold = args.reliability_high_threshold
    cfg.reliability.reliability_low_threshold = args.reliability_low_threshold
    cfg.reliability.max_update_fraction = args.max_update_fraction
    cfg.rrtta.augment_times = args.augment_times
    cfg.rrtta.flip_p = args.flip_p
    cfg.rrtta.brightness = args.brightness
    cfg.rrtta.contrast = args.contrast
    cfg.rrtta.noise_std = args.noise_std
    cfg.rrtta.high_loss_weight = args.high_loss_weight
    cfg.rrtta.medium_loss_weight = args.medium_loss_weight
    cfg.rrtta.low_loss_weight = args.low_loss_weight
    cfg.rrtta.low_conf_threshold = args.low_conf_threshold
    cfg.rrtta.proto_momentum = args.proto_momentum
    cfg.rrtta.proto_align_weight = args.proto_align_weight
    cfg.rrtta.proto_warmup_batches = args.proto_warmup_batches
    cfg.rrtta.proto_min_count = args.proto_min_count
    cfg.rrtta.proto_high_only = not args.proto_all_routes
    return cfg


def evaluate_method(
    cfg: RunConfig,
    setup_method: Callable,
    logger_name: Optional[str] = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CIFAR-C test-time adaptation.")
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    run_id = cfg.run_id or datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / cfg.dataset / run_id
    logger = setup_logger(run_dir, logger_name or cfg.method, run_id)
    logger.info("torch=%s cuda=%s cudnn=%s", torch.__version__, torch.version.cuda, torch.backends.cudnn.version())
    logger.info("method=%s dataset=%s arch=%s", cfg.method, cfg.dataset, cfg.arch)
    logger.info("batch_size=%s num_ex=%s severity=%s", cfg.batch_size, cfg.num_ex, cfg.severity)
    logger.info("data_dir=%s ckpt_dir=%s output_dir=%s", cfg.data_dir, cfg.ckpt_dir, run_dir)

    base_model = load_model(cfg.arch, cfg.ckpt_dir, cfg.dataset, ThreatModel.corruptions).cuda()
    model = setup_method(base_model, cfg, logger)
    model.reset()

    rows: List[Dict[str, object]] = []
    errors: List[float] = []
    total_start = time.time()
    for index, corruption in enumerate(cfg.corruptions, start=1):
        logger.info("corruption %s/%s %s start", index, len(cfg.corruptions), corruption)
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        x_test, y_test = load_corruption_dataset(cfg, corruption)
        x_test, y_test = x_test.cuda(), y_test.cuda()
        correct = 0
        total = 0
        for start in range(0, x_test.shape[0], cfg.batch_size):
            end = min(start + cfg.batch_size, x_test.shape[0])
            with torch.no_grad():
                logits = model(x_test[start:end])
            labels = y_test[start:end]
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        accuracy = correct / max(1, total)
        error = 1.0 - accuracy
        elapsed = time.time() - start_time
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        errors.append(error)
        rows.append(
            {
                "method": cfg.method,
                "dataset": cfg.dataset,
                "severity": cfg.severity,
                "corruption": corruption,
                "num_examples": total,
                "accuracy_percent": accuracy * 100.0,
                "error_percent": error * 100.0,
                "time_seconds": elapsed,
                "peak_gpu_mb": peak_mb,
            }
        )
        logger.info("error [%s]: %.2f%% | time: %.1fs | peak_gpu: %.0fMB", corruption, error * 100.0, elapsed, peak_mb)
        del x_test, y_test

    mean_error = sum(errors) / len(errors) if errors else 0.0
    slug = cfg.method.lower()
    write_csv(run_dir / f"{slug}_{run_id}_per_corruption.csv", rows)
    write_csv(
        run_dir / f"{slug}_{run_id}_summary.csv",
        [
            {
                "method": cfg.method,
                "dataset": cfg.dataset,
                "run_id": run_id,
                "mean_error_percent": mean_error * 100.0,
                "mean_accuracy_percent": (1.0 - mean_error) * 100.0,
                "num_corruptions": len(rows),
                "total_time_seconds": time.time() - total_start,
                "arch": cfg.arch,
                "batch_size": cfg.batch_size,
                "num_ex": cfg.num_ex,
                "severity": cfg.severity,
            }
        ],
    )
    logger.info("mean error: %.2f%%", mean_error * 100.0)
    return run_dir
