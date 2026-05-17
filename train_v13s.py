from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "waterscenes.yaml"
DEFAULT_MODEL = ROOT / "ultralytics" / "cfg" / "models" / "v13" / "yolov13s.yaml"


def parse_args():
    parser = ArgumentParser(description="Train YOLOv13 on the WaterScenes dataset.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv13 model weights or yaml")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--device", default="0", help="Training device, e.g. 0, 0,1, cpu")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Save results to project/name")
    parser.add_argument("--name", default="yolov13", help="Experiment name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default='Achelous++', help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity/team name")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"], help="W&B mode")
    return parser.parse_args()


def init_wandb(args):
    if args.no_wandb or args.wandb_mode == "disabled":
        SETTINGS.update({"wandb": False})
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Weights & Biases is not installed. Install it with: pip install wandb") from exc

    SETTINGS.update({"wandb": True})
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def _get_metric(metrics, *keys):
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _get_mar5095(validator_metrics):
    """Compute mean AR over IoU 0.50:0.95 from the validator's TP matrix."""
    stats = getattr(validator_metrics, "stats", None)
    if not stats or not stats.get("tp") or not stats.get("target_cls") or not stats.get("pred_cls"):
        return None

    tp = np.concatenate(stats["tp"], axis=0)
    target_cls = np.concatenate(stats["target_cls"], axis=0)
    pred_cls = np.concatenate(stats["pred_cls"], axis=0)
    if tp.size == 0 or target_cls.size == 0:
        return 0.0

    recalls = []
    for cls in np.unique(target_cls).astype(int):
        n_targets = (target_cls == cls).sum()
        if n_targets == 0:
            continue
        cls_tp = tp[pred_cls.astype(int) == cls]
        recalls.append(cls_tp.sum(axis=0) / n_targets if cls_tp.size else np.zeros(tp.shape[1]))

    return float(np.mean(recalls)) if recalls else 0.0


def cache_mar5095_on_val_batch_end(validator):
    """Cache mAR50:95 before Ultralytics clears validator.metrics.stats."""
    if validator.batch_i + 1 == len(validator.dataloader):
        validator.mar5095 = _get_mar5095(validator.metrics)


def log_map_metrics_to_wandb(trainer):
    """Log mAP50, mAP75, mAP50-95, AR50 and mAR50:95 to W&B at the end of every fit epoch."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    metrics = trainer.metrics or {}
    log_data = {}

    map50 = _get_metric(metrics, "metrics/mAP50(B)", "metrics/mAP50")
    map5095 = _get_metric(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
    ar50 = _get_metric(metrics, "metrics/AR50(B)", "metrics/AR50", "metrics/recall(B)", "metrics/recall")
    if map50 is not None:
        log_data["mAP50"] = map50
    if map5095 is not None:
        log_data["mAP50-95"] = map5095
    if ar50 is not None:
        log_data["AR50"] = ar50

    validator_metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
    box_metrics = getattr(validator_metrics, "box", None)
    map75 = getattr(box_metrics, "map75", None)
    mar5095 = _get_metric(metrics, "metrics/mAR50:95(B)", "metrics/mAR50-95(B)", "metrics/mAR50:95")
    if mar5095 is None:
        mar5095 = getattr(getattr(trainer, "validator", None), "mar5095", None)
    if mar5095 is None:
        mar5095 = _get_mar5095(validator_metrics)
    if map75 is not None:
        log_data["mAP75"] = float(map75)
    if mar5095 is not None:
        log_data["mAR50:95"] = mar5095
    if "AR50" not in log_data:
        mean_recall = getattr(box_metrics, "mr", None)
        if mean_recall is not None:
            log_data["AR50"] = float(mean_recall)

    if log_data:
        log_data["epoch"] = trainer.epoch + 1
        wandb.log(log_data, step=trainer.epoch + 1, commit=False)


def main():
    args = parse_args()
    init_wandb(args)
    model = YOLO(args.model)
    model.add_callback("on_val_batch_end", cache_mar5095_on_val_batch_end)
    model.add_callback("on_fit_epoch_end", log_map_metrics_to_wandb)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "dfl": 0.0,
        # Disable all built-in data augmentation.
        # "hsv_h": 0.0,
        # "hsv_s": 0.0,
        # "hsv_v": 0.0,
        # "degrees": 0.0,
        # "translate": 0.0,
        # "scale": 0.0,
        # "shear": 0.0,
        # "perspective": 0.0,
        # "flipud": 0.0,
        # "fliplr": 0.0,
        # "bgr": 0.0,
        # "mosaic": 0.0,
        # "mixup": 0.0,
        # "cutmix": 0.0,
        # "copy_paste": 0.0,
        # "auto_augment": None,
        # "erasing": 0.0,
        # "multi_scale": 0.0,
        # "augment": False,
        # "close_mosaic": 0,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
