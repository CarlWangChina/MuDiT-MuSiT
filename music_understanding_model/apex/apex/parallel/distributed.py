import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.parallel.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict
from itertools import chain
import copy
import importlib
from ..multi_tensor_apply import multi_tensor_applier

imported_flatten_impl = False

def import_flatten_impl():
    global flatten_impl, unflatten_impl, imported_flatten_impl
    try:
        import apex_C
        flatten_impl = apex_C.flatten
        unflatten_impl = apex_C.unflatten
    except ImportError:
        print("Warning:  apex was installed without --cpp_ext.  Falling back to Python flatten and unflatten.")
        flatten_impl = torch._utils._flatten_dense_tensors
        unflatten_impl = torch._utils._unflatten_dense_tensors
    imported_flatten_impl = True

def flatten(bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return flatten_impl(bucket)

def unflatten(coalesced, bucket):
    if not imported_flatten_impl:
        import_flatten_impl()
    return unflatten_impl(coalesced, bucket)

def apply_flat_dist_call(bucket, call, extra_args=None):
    coalesced = flatten(bucket)
    if extra_args is not None:
        call(coalesced, *extra_args)
    else:
        call(coalesced)
    if call is dist.all_reduce:
        coalesced /= dist.get_world_size()
        for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
            buf.copy_(synced)

def split_half_float_double(tensors):
    dtypes = ["torch.cuda.HalfTensor", "torch.cuda.FloatTensor", "torch.cuda.DoubleTensor"]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append(bucket)
    return buckets

def split_by_type(tensors):
    buckets = OrderedDict()
    for tensor in tensors:
        tp = tensor.type()
        if tp not in buckets:
            buckets[tp] = []
        buckets[tp].append(tensor)
    return buckets

def flat_dist_call(tensors, call, extra_args=None):
    buckets = split_by_type(tensors)
    for tp in buckets:
        bucket = buckets[tp]
        apply_flat_dist_call(bucket, call, extra_args)

def extract_tensors(maybe_tensor, tensor_list):
    if torch.is_tensor(maybe_tensor):
        tensor_list.append(maybe_tensor)
    else:
        try:
            for item in maybe_tensor:
                extract_tensors(item, tensor_list)
        except TypeError:
            return

class Reducer(object):
    def __init__(self, module_or_grads_list):
        if isinstance(module_or_grads_list, Module):
            self.module = module_or_grads_list
            flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,))
        else:
            self.module = None
            self.grads = []
            extract_tensors(module_or_grads_list, self.grads)

    def reduce(self):
        if self.module:
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
            flat_dist_call(grads, dist.all_reduce)
        else:
            flat_dist_call(self.grads, dist.all_reduce)

class DistributedDataParallel(Module):
    def __init__(self, module, message_size=10000000, delay_allreduce=False, shared_param=None, allreduce_trigger_params=None, retain_allreduce_buffers=False, allreduce_always_fp32=False, gradient_average=True, gradient_predivide_factor=1.0):
        super(DistributedDataParallel, self).__init__()
        if hasattr(dist, "get_backend"):
            self._backend = dist.get_backend()
            if hasattr(dist, "DistBackend"):
                self.backend_enum_holder = dist.DistBackend
            else:
                self.backend_enum_holder = dist.Backend
        else:
            self._backend = dist._backend
            self.backend_enum_holder = dist.dist_backend
        self.warn_on_half = True if self._backend == self.backend_enum_holder.GLOO else False
        if shared_param is not None:
            raise ValueError("shared_param is no longer supported as an option.  It was misleadingly named from the start.  It turns out overlapping communication with computation should work fine with shared parameters.  If you still wish to delay communication to the end of the backward pass, use delay_allreduce=True|False instead.")
        self.world_size = float(dist.get_world_size())
        self.retain_allreduce_buffers = retain_allreduce_buffers
        self.allreduce_always_fp32 = allreduce_always_fp32
        self.gradient_average = gradient_average
        self.gradient_predivide_factor = gradient_predivide_factor
        self.custom_allreduce_triggers = False
        if allreduce_trigger_params is not None:
            if delay_allreduce:
                raise ValueError("Setting allreduce_trigger_params is only valid if delay_allreduce=False.")
            self.custom_allreduce_triggers = True
            self.allreduce_trigger_params = set([id(param) for param in allreduce_trigger_params])
        self.delay_allreduce = delay_allreduce
        self.message_size = message_size
        self.reduction_stream = torch.cuda.Stream()
        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)
        self.module = module
        self._disable_allreduce = False
        if self._backend == self.backend_enum_holder.NCCL:
            for param in self.module.parameters():
                assert param.is_cuda, "NCCL backend only supports model parameters to be on GPU."
        self.active_params = []
        self.param_type_to_tmp_i = {"torch.cuda.HalfTensor": 0, "torch.cuda.FloatTensor": 1, "torch.cuda.DoubleTensor": 2}
        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_scale = amp_C.multi_tensor_scale
            self._overflow_buf = torch.cuda.IntTensor([0])
        self.create_hooks()
        flat_dist_call([param.data for param in self.module.parameters()], dist.broadcast, (0,))

    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        self.reduction_stream = torch.cuda.Stream()
        self.reduction_event = torch.cuda.Event(enable_timing=False, blocking=False)

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        if self._backend != self.backend_enum_holder.NCCL:
            del attrs['reduction_stream']
            del attrs['reduction_event']
        return attrs

    def enable_allreduce(self):
        self._disable_allreduce = False

    def disable_allreduce(self):
        self._disable_allreduce = True

    def sync_bucket_structure(self):
        for tmp_bucket in self.tmp_buckets:
            if len(tmp_bucket) > 0:
                self.active_i_buckets.append(tmp_bucket)
        self.num_buckets = len(self.active_i_buckets)
        self.bucket_sizes = [len(bucket) for bucket in self.active_i_buckets]
        info_tensor = torch.cuda.IntTensor([self.num_buckets] + self.bucket_sizes + list(chain(*self.active_i_buckets)))
        dist.broadcast(info_tensor, 0)
        info = [int(entry) for entry in info_tensor]
        self.num_buckets = info[0]
        self.bucket_sizes = info[1:self.num_buckets + 1]
        self.buckets = [[None for _ in range(self.bucket_sizes[i])] for i in range(self.num_buckets)]
        self.active_i_buckets = [[None for _ in range(self.bucket_sizes[i])] for i in range(self.num_buckets)]
        flattened_buckets = info[self.num_buckets + 1:]
        flat_i = 0
        for bucket_idx in range(self.num_buckets):
            for bucket_loc in range(self.bucket_sizes[bucket_idx]):
                param_i = flattened_buckets[flat_i]
                self.active_i_buckets[bucket_idx][bucket_loc] = param_i
                self.param_id_to_bucket[id(self.active_params[param_i])] = (bucket_idx, bucket_loc)
                flat_i += 1

    def create_hooks(self):
        def allreduce_params():
            if not self.delay_allreduce:
                if self.needs_refresh:
                    self.sync_bucket_structure()
                    self.needs_refresh = False
            self.allreduce_fallback()

        def overlapping_backward_epilogue():
            self.reduction_stream.record_event(self.reduction_event)
            torch.cuda.current_stream().wait_event(self.reduction_event)
            if self.next_bucket != self.num_buckets:
                raise RuntimeError("In epilogue, next_bucket ({}) != num_buckets ({}).  ".format(self.next_bucket, self.num_buckets), "This probably indicates some buckets were not allreduced.")
            for actual, expected in zip(self.buckets_ready_size, self.bucket_sizes):
                if actual != expected:
                    raise RuntimeError("Some param buckets were not allreduced.")

        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                def wrapper(param):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    def allreduce_hook(*unused):
                        if not self._disable_allreduce:
                            if self.delay_allreduce or self.needs_refresh:
                                if not self.delay_allreduce and self.needs_refresh:
                                    active_i = self.param_id_to_active_i[id(param)]
                                    current_type = self.param_type_to_tmp_i[param.type()]
                                    self.tmp_buckets[current_type].append(active_i)
                                    ship_tmp_bucket = False
                                    if self.custom_allreduce_triggers:
                                        if id(param) in self.allreduce_trigger_params:
                                            ship_tmp_bucket = True
                                    else:
                                        self.tmp_numels[current_type] += param.numel()
                                        if self.tmp_numels[current_type] >= self.message_size:
                                            ship_tmp_bucket = True
                                    if ship_tmp_bucket:
                                        self.active_i_buckets.append(self.tmp_buckets[current_type])
                                        self.tmp_buckets[current_type] = []
                                        self.tmp_numels[current_type] = 0
                                if not self.callback_queued:
                                    Variable._execution_engine.queue_callback(allreduce_params)
                                    self.callback_queued = True
                            else:
                                if not self.callback_queued:
                                    Variable._execution_engine.queue_callback(overlapping_backward_epilogue)
                                    self.callback_queued = True
                            self.comm_ready_buckets(param)
                    grad_acc.register_hook(allreduce_hook)
                    self.grad_accs.append(grad_acc)
                wrapper(param)

    def allreduce_bucket(self, bucket):
        tensor = flatten(bucket)
        tensor_to_allreduce = tensor
        if self.allreduce_always_fp32:
            tensor_to_allreduce = tensor.float()
        if self.gradient_predivide_factor != 1.0:
            tensor_to_allreduce.mul_(1. / self.gradient_predivide_factor)
        dist.all_reduce(tensor_to_allreduce)
        if self.gradient_average:
            if self.gradient_predivide_factor != self.world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor / self.world_size)
        if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)
        return tensor

    def allreduce_maybe_retain(self, bucket, bucket_idx=-1):
        allreduced = self.allreduce_bucket(bucket)
        if self.retain_allreduce_buffers:
            if self.allreduce_buffers[bucket_idx] is not None:
                raise RuntimeError("The backward pass is attempting to replace an already-filled allreduce buffer.  This is almost certainly an error.")
            self.allreduce_buffers[bucket_idx] = allreduced
        else:
            if multi_tensor_applier.available:
                multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [unflatten(allreduced, bucket), bucket], 1.0)
            else:
                for buf, synced in zip(bucket, unflatten(allreduced, bucket)):
                    buf.copy_(synced)

    def allreduce_fallback(self):
        grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
        split_buckets = split_half_float_double(grads)
        if self.retain_allreduce_buffers:
            self.allreduce_buffers = [None for _ in range(len(split_buckets))]
            for i, bucket in enumerate(split_buckets):
                self.allreduce_maybe_retain(bucket, i)

    def comm_ready_buckets(self, param):
        bucket_idx, bucket_loc = self.param_id_to_bucket[id(param)]
        if self.buckets[bucket_idx][bucket_loc] is not None:
            raise RuntimeError("The backward pass is attempting to replace an already-filled bucket slot.  This is almost certainly an error.")
        self.buckets[bucket_idx][bucket_loc] = param.grad.data
        self.buckets_ready_size[bucket_idx] += 1
        if self.buckets_ready_size[bucket_idx] == self.bucket_sizes[bucket_idx]:
            if bucket_idx == self.next_bucket:
                torch.cuda.current_stream().record_event(self.reduction_event)
                self.reduction_stream.wait_event(self.reduction_event)
                with torch.cuda.stream(self.reduction_stream):
                    self.allreduce_maybe_retain(self.buckets[bucket_idx], bucket_idx)
                    self.next_bucket += 1
                    if len(self.ready_buckets_not_reduced) > 0:
                        sorted_todo = sorted(self.ready_buckets_not_reduced)
                        for i in sorted_todo:
                            if i > self.next_bucket:
                                break
                            elif i == self.next_bucket:
                                self.allreduce_maybe_retain(self.buckets[i], i)
                                self.ready_buckets_not_reduced.remove(i)
                                self.next_bucket += 1
                            else:
                                raise ValueError("i should always be >= next_bucket")
            else:
                self.ready_buckets_not_reduced.add(bucket_idx)

    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        if not self._disable_allreduce:
            if not self.delay_allreduce:
                param_list = [param for param in self.module.parameters() if param.requires_grad]
                if ((not self.active_params) or (len(param_list) != len(self.active_params)) or any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)])):
                    self.needs_refresh = True
                if self.needs_refresh:
                    self.active_i_buckets = []
                    self.buckets = []
                    self.tmp_buckets = [[], [], []]
                    self.tmp_numels = [0, 0, 0]
                    self.bucket_sizes = []
                    self.param_id_to_active_i = {id(param): i for i, param in enumerate(param_list)}
                    self.param_id_to_bucket = {}
                else:
                    self.buckets = [[None for _ in range(self.bucket_sizes[i])] for i in range(self.num_buckets)]
                    self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                    if self.retain_allreduce_buffers:
                        self.allreduce_buffers = [None for _ in range(self.num_buckets)]
                    self.next_bucket = 0
                    self.ready_buckets_not_reduced = set()
                self.active_params = param_list
            self.callback_queued = False
        return result