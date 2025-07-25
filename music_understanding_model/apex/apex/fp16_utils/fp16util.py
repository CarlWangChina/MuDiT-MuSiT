import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.nn as nn

class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()
    def forward(self, input):
        return input.half()

def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))

def convert_module(module, dtype):
    for param in module.parameters(recurse=False):
        if param is not None:
            if param.data.dtype.is_floating_point:
                param.data = param.data.to(dtype=dtype)
            if param._grad is not None and param._grad.data.dtype.is_floating_point:
                param._grad.data = param._grad.data.to(dtype=dtype)
    for buf in module.buffers(recurse=False):
        if buf is not None and buf.data.dtype.is_floating_point:
            buf.data = buf.data.to(dtype=dtype)

def convert_network(network, dtype):
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine is True:
            continue
        convert_module(module, dtype)
        if isinstance(module, torch.nn.RNNBase) or isinstance(module, torch.nn.modules.rnn.RNNBase):
            module.flatten_parameters()
    return network

class FP16Model(nn.Module):
    def __init__(self, network):
        super(FP16Model, self).__init__()
        self.network = convert_network(network, dtype=torch.half)
    def forward(self, *inputs):
        inputs = tuple(t.half() for t in inputs)
        return self.network(*inputs)

def backwards_debug_hook(grad):
    raise RuntimeError("master_params recieved a gradient in the backward pass!")

def prep_param_lists(model, flat_master=False):
    model_params = [param for param in model.parameters() if param.requires_grad]
    if flat_master:
        try:
            master_params = _flatten_dense_tensors([param.data for param in model_params]).float()
        except:
            print("Error in prep_param_lists:  model may contain a mixture of parameters "
                      "of different types.  Use flat_master=False, or use F16_Optimizer.")
            raise
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        if master_params.grad is None:
            master_params.grad = master_params.new(*master_params.size())
        return model_params, [master_params]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params:
            param.requires_grad = True
        return model_params, master_params

def model_grads_to_master_grads(model_params, master_params, flat_master=False):
    if flat_master:
        master_params[0].grad.data.copy_(
            _flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size()))
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None

def master_params_to_model_params(model_params, master_params, flat_master=False):
    if flat_master:
        for model, master in zip(model_params,
                                  _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 0 and TORCH_MINOR <= 4:
    clip_grad_norm = torch.nn.utils.clip_grad_norm
else:
    clip_grad_norm = torch.nn.utils.clip_grad_norm_