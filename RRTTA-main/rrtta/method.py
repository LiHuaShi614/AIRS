from __future__ import annotations

from copy import deepcopy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

from .optim import setup_optimizer


def _is_adapter_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))


def configure_model(model: nn.Module) -> nn.Module:
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.requires_grad_(True)
    return model


def collect_params(model: nn.Module):
    params = []
    names = []
    for module_name, module in model.named_modules():
        if not _is_adapter_module(module):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param_name not in {"weight", "bias"} or not param.requires_grad:
                continue
            params.append(param)
            names.append(f"{module_name}.{param_name}" if module_name else param_name)
    if not params:
        raise RuntimeError("rrtta found no BN/Norm parameters to adapt")
    return params, names


class FeatureLogitExtractor:
    """Extract logits and the tensor before the final linear classifier."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.classifier_name, self.classifier = self._find_classifier(model)
        self.feature: Optional[torch.Tensor] = None
        self._hook = self.classifier.register_forward_pre_hook(self._capture_feature)

    @staticmethod
    def _find_classifier(model: nn.Module) -> Tuple[str, nn.Linear]:
        candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        if not candidates:
            raise RuntimeError("rrtta requires a model with a Linear classifier head")
        return candidates[-1]

    def _capture_feature(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        self.feature = inputs[0]

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.feature = None
        logits = self.model(x)
        if self.feature is None:
            raise RuntimeError("rrtta failed to capture classifier input features")
        feature = self.feature
        if feature.dim() > 2:
            feature = feature.flatten(1)
        return logits, feature


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * logits.log_softmax(dim=1)).sum(dim=1)


def route_by_reliability(logits: torch.Tensor, num_classes: int, high_threshold: float, low_threshold: float):
    probs = logits.softmax(dim=1)
    top2 = probs.topk(k=2, dim=1)
    confidence = top2.values[:, 0]
    labels = top2.indices[:, 0]
    margin = confidence - top2.values[:, 1]
    entropy = softmax_entropy(logits)
    entropy_score = (1.0 - entropy / max(1e-6, math.log(num_classes))).clamp(0.0, 1.0)
    reliability = (confidence * margin.clamp_min(0.0) * entropy_score).clamp(0.0, 1.0)
    high_mask = reliability >= high_threshold
    medium_mask = (reliability >= low_threshold) & (~high_mask)
    low_mask = ~(high_mask | medium_mask)
    return {
        "probs": probs,
        "labels": labels,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "entropy_score": entropy_score,
        "reliability": reliability,
        "high_mask": high_mask,
        "medium_mask": medium_mask,
        "low_mask": low_mask,
    }


def augment_tensor_batch(
    x: torch.Tensor,
    flip_p: float,
    brightness: float,
    contrast: float,
    noise_std: float,
) -> torch.Tensor:
    y = x.clone()
    if flip_p > 0:
        flip_mask = torch.rand(y.shape[0], device=y.device) < flip_p
        if flip_mask.any():
            y[flip_mask] = torch.flip(y[flip_mask], dims=[3])
    if brightness > 0:
        lo = max(0.0, 1.0 - brightness)
        hi = 1.0 + brightness
        factor = torch.empty(y.shape[0], 1, 1, 1, device=y.device, dtype=y.dtype).uniform_(lo, hi)
        y = y * factor
    if contrast > 0:
        lo = max(0.0, 1.0 - contrast)
        hi = 1.0 + contrast
        factor = torch.empty(y.shape[0], 1, 1, 1, device=y.device, dtype=y.dtype).uniform_(lo, hi)
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * factor + mean
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    return y.clamp(0.0, 1.0)


class RRTTA(nn.Module):
    """Reliability-routed test-time adaptation."""

    def __init__(self, teacher, student, optimizer, cfg):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_extractor = FeatureLogitExtractor(teacher)
        self.student_extractor = FeatureLogitExtractor(student)
        self.optimizer = optimizer
        self.steps = cfg.optim.steps
        self.episodic = cfg.episodic
        self.num_classes = cfg.num_classes
        self.high_threshold = cfg.reliability.reliability_high_threshold
        self.low_threshold = cfg.reliability.reliability_low_threshold
        self.max_update_fraction = cfg.reliability.max_update_fraction
        self.sce_alpha = cfg.reliability.sce_alpha
        self.sce_beta = cfg.reliability.sce_beta
        self.augment_times = max(0, int(cfg.rrtta.augment_times))
        self.flip_p = cfg.rrtta.flip_p
        self.brightness = cfg.rrtta.brightness
        self.contrast = cfg.rrtta.contrast
        self.noise_std = cfg.rrtta.noise_std
        self.high_loss_weight = cfg.rrtta.high_loss_weight
        self.medium_loss_weight = cfg.rrtta.medium_loss_weight
        self.low_loss_weight = cfg.rrtta.low_loss_weight
        self.low_conf_threshold = cfg.rrtta.low_conf_threshold
        self.teacher_momentum = cfg.rrtta.teacher_momentum
        self.adapt_batch_size = max(1, int(cfg.rrtta.adapt_batch_size))
        self.refine_label_margin = max(0.0, float(cfg.rrtta.refine_label_margin))
        self.student_disagreement_weight = max(0.0, min(1.0, float(cfg.rrtta.student_disagreement_weight)))
        self.prediction_blend = max(0.0, min(1.0, float(cfg.rrtta.prediction_blend)))
        self.proto_momentum = max(0.0, min(1.0, float(cfg.rrtta.proto_momentum)))
        self.proto_align_weight = max(0.0, float(cfg.rrtta.proto_align_weight))
        self.proto_warmup_batches = max(0, int(cfg.rrtta.proto_warmup_batches))
        self.proto_min_count = max(1, int(cfg.rrtta.proto_min_count))
        self.proto_high_only = bool(cfg.rrtta.proto_high_only)
        self.batch_index = 0
        self.proto_dim = 0
        self.register_buffer("proto_bank", torch.empty(0))
        self.register_buffer("proto_count", torch.zeros(self.num_classes))
        self.teacher_state = deepcopy(teacher.state_dict())
        self.student_state = deepcopy(student.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        self.last_adaptation_stats = {}
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    def reset(self):
        self.teacher.load_state_dict(self.teacher_state, strict=True)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self.student.load_state_dict(self.student_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.batch_index = 0
        self.proto_dim = 0
        self.proto_bank = self.proto_bank.new_empty(0)
        self.proto_count.zero_()
        self.last_adaptation_stats = {}

    def _ensure_proto_bank(self, feature_dim: int, device: torch.device, dtype: torch.dtype):
        if self.proto_bank.numel() == 0:
            self.proto_dim = feature_dim
            self.proto_bank = torch.zeros(self.num_classes, feature_dim, device=device, dtype=dtype)
            self.proto_count = torch.zeros(self.num_classes, device=device, dtype=dtype)

    @torch.no_grad()
    def update_teacher(self):
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(self.teacher_momentum).add_(
                student_param.data,
                alpha=1.0 - self.teacher_momentum,
            )

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.no_grad()
    def make_teacher_targets(self, x: torch.Tensor):
        teacher_outputs, teacher_features = self.teacher_extractor(x)
        if self.proto_align_weight > 0.0:
            self._ensure_proto_bank(teacher_features.shape[1], teacher_features.device, teacher_features.dtype)
        teacher_probs = teacher_outputs.softmax(dim=1)
        route = route_by_reliability(teacher_outputs, self.num_classes, self.high_threshold, self.low_threshold)
        labels = route["labels"].detach().clone()
        refine_mask = route["medium_mask"] | route["low_mask"]
        keep_mask = route["high_mask"] | route["medium_mask"]
        weights = teacher_outputs.new_zeros(teacher_outputs.shape[0])
        weights[route["high_mask"]] = self.high_loss_weight
        weights[route["medium_mask"]] = self.medium_loss_weight
        weights[route["low_mask"]] = self.low_loss_weight

        if self.augment_times > 0 and refine_mask.any():
            selected_x = x[refine_mask]
            probs_sum = teacher_probs[refine_mask].clone()
            for _ in range(self.augment_times):
                aug_x = augment_tensor_batch(
                    selected_x,
                    self.flip_p,
                    self.brightness,
                    self.contrast,
                    self.noise_std,
                )
                probs_sum = probs_sum + self.teacher(aug_x).softmax(dim=1)
            refined_probs = probs_sum / float(self.augment_times + 1)
            refined_conf, refined_labels = refined_probs.max(dim=1)
            original_conf = route["confidence"][refine_mask]
            selected_positions = torch.where(refine_mask)[0]
            same_label = refined_labels == labels[selected_positions]
            use_refined = same_label | (refined_conf >= original_conf + self.refine_label_margin)
            labels[selected_positions[use_refined]] = refined_labels[use_refined]

            low_positions = torch.where(route["low_mask"])[0]
            if low_positions.numel() > 0:
                selected_is_low = route["low_mask"][selected_positions]
                low_keep = (refined_conf[selected_is_low] >= self.low_conf_threshold) & use_refined[selected_is_low]
                keep_mask[low_positions] = low_keep

        weights = weights * keep_mask.float()
        return labels, weights, route, teacher_features.detach()

    @torch.no_grad()
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor, route):
        if self.proto_align_weight <= 0.0:
            return
        high_positions = torch.where(route["high_mask"])[0]
        if high_positions.numel() == 0:
            return
        high_features = F.normalize(features[high_positions].detach(), dim=1)
        high_labels = labels[high_positions]
        for cls in high_labels.unique():
            cls_mask = high_labels == cls
            cls_index = int(cls.item())
            center = F.normalize(high_features[cls_mask].mean(dim=0), dim=0)
            if self.proto_count[cls_index] <= 0:
                self.proto_bank[cls_index] = center
            else:
                updated = self.proto_bank[cls_index] * self.proto_momentum + center * (1.0 - self.proto_momentum)
                self.proto_bank[cls_index] = F.normalize(updated, dim=0)
            self.proto_count[cls_index] += int(cls_mask.sum().item())

    @torch.no_grad()
    def apply_student_agreement(self, x: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
        if self.student_disagreement_weight >= 1.0 or not (weights > 0).any():
            return weights
        student_outputs, _ = self.student_extractor(x)
        student_labels = student_outputs.argmax(dim=1)
        adjusted = weights.clone()
        adjusted[student_labels != labels] *= self.student_disagreement_weight
        return adjusted

    def cap_update_weights(self, weights: torch.Tensor, route):
        positive = weights > 0
        if not positive.any() or self.max_update_fraction >= 1.0:
            return weights
        max_count = max(1, int(round(weights.shape[0] * self.max_update_fraction)))
        selected = torch.where(positive)[0]
        if selected.numel() <= max_count:
            return weights
        scores = route["reliability"][selected]
        keep = selected[torch.topk(scores, k=max_count, largest=True).indices]
        capped = torch.zeros_like(weights)
        capped[keep] = weights[keep]
        return capped

    def weighted_symmetric_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        total_weight: torch.Tensor | None = None,
    ):
        positive = weights > 0
        if not positive.any():
            return None
        logits = logits[positive]
        targets = targets[positive]
        weights = weights[positive]
        if total_weight is None:
            total_weight = weights.sum()
        weights = weights / total_weight.clamp_min(1e-6)
        ce = F.cross_entropy(logits, targets, reduction="none")
        probs = logits.softmax(dim=1).clamp_min(1e-4)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float().clamp_min(1e-4)
        rce = -(probs * one_hot.log()).sum(dim=1)
        loss = self.sce_alpha * ce + self.sce_beta * rce
        return (loss * weights).sum()

    def prototype_alignment_loss(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        route_mask: torch.Tensor,
        total_weight: torch.Tensor,
    ):
        if self.proto_align_weight <= 0.0 or self.batch_index < self.proto_warmup_batches:
            return None
        valid = (weights > 0) & route_mask
        if not valid.any():
            return None
        targets_valid = targets[valid]
        has_proto = self.proto_count[targets_valid].to(weights.device) >= self.proto_min_count
        if not has_proto.any():
            return None
        features_valid = F.normalize(features[valid][has_proto], dim=1)
        targets_valid = targets_valid[has_proto]
        weights_valid = weights[valid][has_proto] / total_weight.clamp_min(1e-6)
        proto = F.normalize(self.proto_bank[targets_valid].to(features_valid.device), dim=1)
        distance = 1.0 - (features_valid * proto).sum(dim=1)
        return (distance * weights_valid).sum()

    @torch.enable_grad()
    def forward_and_adapt(self, x, predict_after: bool = True):
        labels, weights, route, teacher_features = self.make_teacher_targets(x)
        weights = self.cap_update_weights(weights, route)
        agreement_forward = int(self.student_disagreement_weight < 1.0 and bool((weights > 0).any()))
        weights = self.apply_student_agreement(x, labels, weights)
        selected = torch.where(weights > 0)[0]
        refine_count = int((route["medium_mask"] | route["low_mask"]).sum().item())
        augmentation_calls = self.augment_times if refine_count > 0 else 0
        self.last_adaptation_stats = {
            "batch_size": int(x.shape[0]),
            "high_samples": int(route["high_mask"].sum().item()),
            "medium_samples": int(route["medium_mask"].sum().item()),
            "low_samples": int(route["low_mask"].sum().item()),
            "selected_samples": int(selected.numel()),
            "teacher_forward_calls": 1 + augmentation_calls,
            "teacher_examples": int(x.shape[0]) + augmentation_calls * refine_count,
            "student_agreement_forward_calls": agreement_forward,
            "student_agreement_examples": agreement_forward * int(x.shape[0]),
            "student_adapt_forward_calls": 0,
            "student_adapt_examples": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
        if selected.numel() == 0:
            self.optimizer.zero_grad()
            self.update_prototypes(teacher_features, labels, route)
            self.batch_index += 1
            if predict_after:
                with torch.no_grad():
                    return self.predict(x)
            return None

        total_weight = weights[selected].sum().detach()
        proto_route_mask = route["high_mask"] if self.proto_high_only else (weights > 0)
        self.optimizer.zero_grad()
        for chunk in selected.split(self.adapt_batch_size):
            self.last_adaptation_stats["student_adapt_forward_calls"] += 1
            self.last_adaptation_stats["student_adapt_examples"] += int(chunk.numel())
            outputs, features = self.student_extractor(x[chunk])
            loss = self.weighted_symmetric_cross_entropy(
                outputs,
                labels[chunk],
                weights[chunk],
                total_weight=total_weight,
            )
            proto_loss = self.prototype_alignment_loss(
                features,
                labels[chunk],
                weights[chunk],
                proto_route_mask[chunk],
                total_weight=total_weight,
            )
            if proto_loss is not None:
                loss = proto_loss * self.proto_align_weight if loss is None else loss + proto_loss * self.proto_align_weight
            if loss is None:
                continue
            if not torch.isfinite(loss):
                self.optimizer.zero_grad()
                if predict_after:
                    with torch.no_grad():
                        return self.predict(x)
                return None
            loss.backward()
            self.last_adaptation_stats["backward_calls"] += 1

        self.optimizer.step()
        self.last_adaptation_stats["optimizer_steps"] = 1
        self.optimizer.zero_grad()
        self.update_prototypes(teacher_features, labels, route)
        self.update_teacher()
        self.batch_index += 1
        if predict_after:
            with torch.no_grad():
                return self.predict(x)
        return None

    def adapt_only(self, x):
        """Adapt on ``x`` without recomputing predictions after the update."""
        return self.forward_and_adapt(x, predict_after=False)

    @torch.no_grad()
    def predict(self, x):
        student_outputs = self.student(x)
        if self.prediction_blend <= 0.0:
            return student_outputs
        teacher_outputs = self.teacher(x)
        return (1.0 - self.prediction_blend) * student_outputs + self.prediction_blend * teacher_outputs


def setup_rrtta(student_model, cfg, logger):
    logger.info("test-time adaptation: reliability-routed RRTTA")
    teacher = load_model(
        cfg.rrtta.teacher_arch,
        cfg.ckpt_dir,
        cfg.dataset,
        ThreatModel.corruptions,
    ).cuda()
    teacher = configure_model(teacher)
    teacher.eval()
    teacher.requires_grad_(False)

    student = configure_model(student_model)
    params, names = collect_params(student)
    optimizer = setup_optimizer(params, cfg.optim)
    rrtta_model = RRTTA(teacher, student, optimizer, cfg)
    logger.info(
        "teacher: arch=%s ckpt_dir=%s ema=True momentum=%s",
        cfg.rrtta.teacher_arch,
        cfg.ckpt_dir,
        cfg.rrtta.teacher_momentum,
    )
    logger.info("student: arch=%s ckpt_dir=%s", cfg.arch, cfg.ckpt_dir)
    logger.info("params for student adaptation: %s", names)
    logger.info("optimizer for student adaptation: %s", optimizer)
    logger.info(
        "rrtta settings: high_threshold=%s low_threshold=%s max_update_fraction=%s sce_alpha=%s sce_beta=%s teacher_momentum=%s adapt_batch_size=%s refine_label_margin=%s student_disagreement_weight=%s prediction_blend=%s augment_times=%s flip_p=%s brightness=%s contrast=%s noise_std=%s high_weight=%s medium_weight=%s low_weight=%s low_conf_threshold=%s",
        cfg.reliability.reliability_high_threshold,
        cfg.reliability.reliability_low_threshold,
        cfg.reliability.max_update_fraction,
        cfg.reliability.sce_alpha,
        cfg.reliability.sce_beta,
        cfg.rrtta.teacher_momentum,
        cfg.rrtta.adapt_batch_size,
        cfg.rrtta.refine_label_margin,
        cfg.rrtta.student_disagreement_weight,
        cfg.rrtta.prediction_blend,
        cfg.rrtta.augment_times,
        cfg.rrtta.flip_p,
        cfg.rrtta.brightness,
        cfg.rrtta.contrast,
        cfg.rrtta.noise_std,
        cfg.rrtta.high_loss_weight,
        cfg.rrtta.medium_loss_weight,
        cfg.rrtta.low_loss_weight,
        cfg.rrtta.low_conf_threshold,
    )
    if cfg.rrtta.proto_align_weight > 0.0:
        logger.info(
            "prototype alignment: enabled momentum=%s weight=%s warmup=%s min_count=%s high_only=%s",
            cfg.rrtta.proto_momentum,
            cfg.rrtta.proto_align_weight,
            cfg.rrtta.proto_warmup_batches,
            cfg.rrtta.proto_min_count,
            cfg.rrtta.proto_high_only,
        )
    else:
        logger.info("prototype alignment: disabled")
    return rrtta_model
