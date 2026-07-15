import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_ as grad_clipping
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm, trange

import wandb
from src.loader import get_dataset
from src.model import Model
from src.utils import (
    AverageMeter,
    accuracy,
    calculate_flops,
    measure_throughput,
    profile_model,
    setup_experiment,
)


def get_summary(meters: dict) -> str:
    return " ".join([f"{k}: {v.avg:5.7f}" for k, v in meters.items()])


def main(args: argparse.Namespace, config: dict) -> None:
    # Set Seeds
    setup_experiment(config["seed"])

    # Check CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    autocast_factory = {
        "device_type": device.type,
        "enabled": args.fp16,
        "dtype": torch.float16,
    }

    if not args.dry_run and not args.debug:
        os.makedirs(args.logs, exist_ok=True)

        run = wandb.init(
            config=config,
            mode="offline",
            **config["logging"],
        )

        run_dir = Path(run.dir).parent
        config_path = run_dir / "config.json"
        model_path = run_dir / "model.pt"

        with open(config_path, "w") as f:
            json.dump(config, f)

    # Get model ----------------------------------------------------------
    model = Model(**config["model"])

    if args.profile:
        torch.autograd.set_detect_anomaly(True)

        profile_model(model, config["model"]["input_shape"], device)
        measure_throughput(model, config["model"]["input_shape"], device)
        exit(0)

    if args.verbose:
        dummy_inputs = torch.rand(config["model"]["input_shape"])
        summary(model, input_data=dummy_inputs)

        flops, macs = calculate_flops(model, dummy_inputs)
        print(f"Model FLOPs: {flops}, MACs: {macs}")

    # Optimizer ----------------------------------------------------------
    if args.opt_name == "sgd":
        optimizer = SGD(
            model.parameters(), momentum=0.9, nesterov=True, **config["optim"]
        )
    elif args.opt_name == "adamw":
        optimizer = AdamW(model.parameters(), **config["optim"])

    scaler = GradScaler(enabled=args.fp16)

    # Load data ----------------------------------------------------------
    t_dataset = get_dataset(
        args.datatype, split="test" if args.debug else "train", **config["data"]
    )
    v_dataset = get_dataset(args.datatype, split="test", **config["data"])

    t_dataloader = DataLoader(
        t_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=(device == "cuda"),
        prefetch_factor=config["prefetch_factor"],
        shuffle=(not args.debug),
    )

    v_dataloader = DataLoader(
        v_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=(device == "cuda"),
        prefetch_factor=config["prefetch_factor"],
        shuffle=False,
    )

    meters = {
        "t_loss": AverageMeter("t_loss", ":.4e"),
        "v_loss": AverageMeter("v_loss", ":.4e"),
        "t_acc1": AverageMeter("t_acc1", ":6.4f"),
        "t_acc5": AverageMeter("t_acc5", ":6.4f"),
        "v_acc1": AverageMeter("v_acc1", ":6.4f"),
        "v_acc5": AverageMeter("v_acc5", ":6.4f"),
    }

    best_val_loss = np.inf

    model.to(device)

    if args.dry_run or args.profile:
        exit(0)

    # --------------------------------------------------------------------
    for i_epoch in trange(1, int(config["num_epochs"]) + 1, desc="Epoch"):

        torch.cuda.empty_cache()

        for meter in meters.values():
            meter.reset()

        # Training -------------------------------------------------------
        model.train()

        for X, Y_true in tqdm(t_dataloader):

            X = X.to(device, non_blocking=True)
            Y_true = Y_true.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(**autocast_factory):
                Y_pred = model(X)

            loss = F.cross_entropy(Y_pred, Y_true)

            scaler.scale(loss).backward()

            if args.clip:
                scaler.unscale_(optimizer)
                grad_clipping(model.parameters(), config["clip_value"])

            scaler.step(optimizer)
            scaler.update()

            t_acc1, t_acc5 = accuracy(Y_pred, Y_true, topk=(1, 5))

            meters["t_loss"].update(loss.item())
            meters["t_acc1"].update(t_acc1.item())
            meters["t_acc5"].update(t_acc5.item())

            if args.debug:
                break

        # Validation ------------------------------------------------------
        model.eval()

        with torch.inference_mode():
            for X, Y_true in tqdm(v_dataloader):

                X = X.to(device, non_blocking=True)
                Y_true = Y_true.to(device, non_blocking=True)

                with autocast(**autocast_factory):
                    Y_pred = model(X)

                    loss = F.cross_entropy(Y_pred, Y_true)

                v_acc1, v_acc5 = accuracy(Y_pred, Y_true, topk=(1, 5))

                meters["v_loss"].update(loss.item())
                meters["v_acc1"].update(v_acc1.item())
                meters["v_acc5"].update(v_acc5.item())

                if args.debug:
                    break

        # Logging --------------------------------------------------------
        if not args.debug:
            run.log({k: v.avg for k, v in meters.items()}, step=i_epoch)

        if args.verbose:
            print(f"Epoch {i_epoch}: {get_summary(meters)}")

        # Checkpoint -----------------------------------------------------
        if meters["v_loss"].avg < best_val_loss and not args.debug:
            best_val_loss = meters["v_loss"].avg

            if args.verbose:
                print(f"Save weights @ {i_epoch} (v_loss: {best_val_loss})")

            checkpoint = {
                "epoch": i_epoch,
                "opt": type(optimizer).__name__,
                "optimizer": optimizer.state_dict(),
                "v_acc1": meters["v_acc1"].avg,
                "v_acc5": meters["v_acc5"].avg,
                "val_loss": meters["v_loss"].avg,
            }

            checkpoint["model"] = model.state_dict()

            if args.fp16:
                checkpoint["scaler"] = scaler.state_dict()

            torch.save(checkpoint, model_path)

    # --------------------------------------------------------------------
    if not args.debug:
        weights = torch.load(model_path)

        print(f"\n{'='*50}")
        print("Best Model Results")
        print(f"Best model: {weights['val_loss']} @ epoch {weights['epoch']}")
        print(f"v_acc@1: {weights['v_acc1']:.3f}")
        print(f"v_acc@5: {weights['v_acc5']:.3f}")
        print(f"path: {model_path.as_posix()}")
        print(f"{'='*50}\n")

    wandb.finish(exit_code=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument(
        "--datatype",
        type=str,
        choices=["audio", "video", "image"],
        default="image",
        required=True,
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--opt_name",
        type=str,
        required=False,
        default="sgd",
        choices=["adamw", "sgd"],
    )
    parser.add_argument("--verbose", action="store_true")

    # Model-related
    parser.add_argument(
        "--pe_variant",
        type=str,
        required=True,
        choices=["rope", "mrope", "mrope_i", "fourier"],
        default="original",
    )

    args = parser.parse_args()

    # Load configs
    assert args.config.exists(), "--config doesn't exist"
    config = json.load(open(args.config, "r+"))

    # Add for model building
    config["model"]["pe_type"] = args.pe_variant

    try:
        main(args, config)

    except Exception as e:
        wandb.finish(exit_code=-1)
        raise e
