from enum import Enum
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from calflops.calculate_pipline import CalFlopsPipline


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def init_parameters(module: nn.Module, init_scale: float = 0.1) -> None:
    """
    Initiallize module parameters.

    Arguments:
        module -- torch module

    Keyword Arguments:
        init_scale -- parameters initial scale (default: {0.1})
    """

    for m in module.modules():

        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=init_scale)

            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=init_scale)


def freeze(module: nn.Module) -> None:
    """
    Freeze layers.
    """
    for param in module.parameters():
        param.requires_grad = False


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Arguments:
        output -- output logits (batch_size x n_classes)
        target -- ground truth as class labels (batch_size x 1)

    Returns:
        Returns accuracy score
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        target = target.argmax(-1)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_flops(model, args: Tuple[torch.Tensor, ...]):
    """
    Calculate the FLOPs and MACs of a model.

    Arguments:
        model -- model
        args -- arguments

    Returns:
        FLOPs and MACs
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"

    calculate_flops_pipline = CalFlopsPipline(
        model=model, include_backPropagation=False, compute_bp_factor=False
    )
    calculate_flops_pipline.start_flops_calculate()

    _ = model(args)  # forward pass

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()

    calculate_flops_pipline.end_flops_calculate()

    return number_to_string(flops), number_to_string(macs)


def number_to_string(num, units=None, precision=3):
    """
    Convert a number to a string with units.

    Arguments:
        num -- number
        units -- units
        precision -- precision

    Returns:
        String with units
    """
    if units is None:
        if num >= 1e12:
            magnitude, units = 1e12, "T"
        elif num >= 1e9:
            magnitude, units = 1e9, "G"
        elif num >= 1e6:
            magnitude, units = 1e6, "M"
        elif num >= 1e3:
            magnitude, units = 1e3, "K"
        elif num >= 1 or num == 0:
            magnitude, units = 1, ""
        elif num >= 1e-3:
            magnitude, units = 1e-3, "m"
        else:
            magnitude, units = 1e-6, "u"
    else:
        if units == "T":
            magnitude = 1e12
        elif units == "G":
            magnitude = 1e9
        elif units == "M":
            magnitude = 1e6
        elif units == "K":
            magnitude = 1e3
        elif units == "m":
            magnitude = 1e-3
        elif units == "u":
            magnitude = 1e-6
        else:
            magnitude = 1

    return f"{round(num / magnitude, precision):g} {units}"


def log_params_and_gradients(named_parameters, run, i_epoch):
    """
    Log gradients and parameters to wandb at end of each epoch

    Arguments:
        named_parameters -- _description_
        run -- _description_
        i_epoch -- _description_
    """

    for name, params in named_parameters:
        grad = params.grad

        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"NaN/Inf detected in gradients ({name}) @ Epoch {i_epoch}")
            continue

        if grad is not None:
            grads = params.grad.detach().cpu().numpy()
            params = params.detach().cpu().numpy()

            run.log(
                {
                    f"gradients/{name}": wandb.Histogram(grads),
                    f"parameters/{name}": wandb.Histogram(params),
                },
                step=i_epoch,
            )


def setup_experiment(seed):
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def profile_model(model, shape, device) -> None:
    from torch.profiler import profile

    dummy_inputs = torch.randn(shape).to(device)
    model.to(device)

    # Warmup
    with torch.no_grad():
        model(dummy_inputs)

    # Profile
    with profile(record_shapes=True) as prof:
        model(dummy_inputs)

    print(f"\n{'='*50}")
    print("Torch profiling")
    print(f"{'='*50}")
    print(prof.key_averages().table(top_level_events_only=True))
    print(f"{'='*50}\n")


def measure_throughput(model, shape, device, iters=100) -> None:
    import time

    dummy_inputs = torch.randn(shape).to(device)
    model.to(device)

    # Throughput measurement
    batch_size = shape[0]

    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            model(dummy_inputs)

    # Synchronize GPU if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Time forward passes
    with torch.no_grad():
        start_time = time.time()
        for _ in range(iters):
            model(dummy_inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()

    elapsed_time = end_time - start_time
    total_samples = batch_size * iters
    throughput = total_samples / elapsed_time
    avg_latency = (elapsed_time / iters) * 1000  # ms

    print(f"\n{'='*50}")
    print("Throughput Measurement Results")
    print(f"{'='*50}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {iters}")
    print(f"Total time: {elapsed_time:.3f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Average latency: {avg_latency:.2f} ms/batch")
    print(f"{'='*50}\n")
