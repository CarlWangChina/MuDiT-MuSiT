import typing as tp
import torch

def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def is_distributed():
    return world_size() > 1

def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)

def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)

def _check_number_of_params(params: tp.List[torch.Tensor]):
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        raise RuntimeError(f"Mismatch in number of params: ours is {len(params)}, "
                           "at least one worker has a different one.")

def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()

def sync_buffer(buffers, average=True):
    if not is_distributed():
        return
    handles = []
    for buffer in buffers:
        if torch.is_floating_point(buffer.data):
            if average:
                handle = torch.distributed.all_reduce(
                    buffer.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            else:
                handle = torch.distributed.broadcast(
                    buffer.data, src=0, async_op=True)
            handles.append((buffer, handle))
    for buffer, handle in handles:
        handle.wait()
        if average:
            buffer.data /= world_size()

def sync_grad(params):
    if not is_distributed():
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size()

def average_metrics(metrics: tp.Dict[str, float], count=1.):
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))