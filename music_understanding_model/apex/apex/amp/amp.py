from . import compat, rnn_compat, utils, wrap
from .handle import AmpHandle, NoOpHandle
from .lists import functional_overrides, torch_overrides, tensor_overrides
from ._amp_state import _amp_state
from .frontend import *
import functools
import itertools
import torch

_DECORATOR_HANDLE = None
_USER_CAST_REGISTRY = set()
_USER_PROMOTE_REGISTRY = set()

def _decorator_helper(orig_fn, cast_fn, wrap_fn):
    def wrapper(*args, **kwargs):
        handle = _DECORATOR_HANDLE
        if handle is None or not handle.is_active():
            return orig_fn(*args, **kwargs)
        inner_cast_fn = utils.verbosify(cast_fn, orig_fn.__name__, handle.verbose)
        return wrap_fn(orig_fn, inner_cast_fn, handle)(*args, **kwargs)
    return wrapper

def half_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=True)
    return _decorator_helper(fn, utils.maybe_half, wrap_fn)

def float_function(fn):
    wrap_fn = functools.partial(wrap.make_cast_wrapper, try_caching=False)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)

def promote_function(fn):
    wrap_fn = functools.partial(wrap.make_promote_wrapper)
    return _decorator_helper(fn, utils.maybe_float, wrap_fn)

def register_half_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_half))

def register_float_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(name, module))
    _USER_CAST_REGISTRY.add((module, name, utils.maybe_float))

def register_promote_function(module, name):
    if not hasattr(module, name):
        raise ValueError('No function named {} in module {}.'.format(name, module))
    _USER_PROMOTE_REGISTRY.add((module, name))

def init(enabled=True, loss_scale="dynamic", enable_caching=True, verbose=False, allow_banned=False):
    global _DECORATOR_HANDLE
    if not enabled:
        handle = NoOpHandle()
        _DECORATOR_HANDLE = handle
        return handle
    handle = AmpHandle(loss_scale, enable_caching, verbose)
    for mod, fn, cast_fn in _USER_CAST_REGISTRY:
        try_caching = (cast_fn == utils.maybe_half)
        wrap.cached_cast(mod, fn, cast_fn, handle, try_caching, verbose)
    _USER_CAST_REGISTRY.clear()
    for mod, fn in _USER_PROMOTE_REGISTRY:
        wrap.promote(mod, fn, handle, verbose)
    _USER_PROMOTE_REGISTRY.clear()
    override_modules = [functional_overrides, torch_overrides, tensor_overrides]
    cast_table = [('FP16_FUNCS', utils.maybe_half), ('FP32_FUNCS', utils.maybe_float)]
    for module, (list_name, cast_fn) in itertools.product(override_modules, cast_table):
        for fn in getattr(module, list_name):
            try_caching = (cast_fn == utils.maybe_half)
            wrap.cached_cast(module.MODULE, fn, cast_fn, handle, try_caching, verbose)
    if compat.tensor_is_float_tensor():
        for fn in tensor_overrides.FP16_FUNCS:
            wrap.cached_cast(torch.cuda.FloatTensor, fn, utils.maybe_half, handle, try_caching=True, verbose=verbose)
        for fn in tensor_overrides.FP32_FUNCS:
            wrap.cached_cast(torch.cuda.HalfTensor, fn, utils.maybe_float, handle, try_caching=False, verbose=verbose)
    promote_modules = [torch_overrides, tensor_overrides]
    promote_table = [('CASTS', wrap.promote), ('SEQUENCE_CASTS', wrap.sequence_promote)]
    for promote_mod, (list_name, promote_fn) in itertools.product(promote_modules, promote_table):
        for fn in getattr(promote_mod, list_name):
            promote_fn(promote_mod.MODULE, fn, handle, verbose)
    if compat.tensor_is_float_tensor():
        for cls, (list_name, promote_fn) in itertools.product([torch.cuda.FloatTensor, torch.cuda.HalfTensor], promote_table):
            for fn in getattr(tensor_overrides, list_name):
                promote_fn(cls, fn, handle, verbose)
    for fn in utils.as_inplace(torch_overrides.FP32_FUNCS):
        wrap.err_if_any_half(torch_overrides.MODULE, fn, handle)
    for fn in utils.as_inplace(tensor_overrides.FP32_FUNCS):
        wrap.err_if_arg0_half(tensor_overrides.MODULE, fn, handle, verbose)
        if compat.tensor_is_float_tensor():
            wrap.err_if_arg0_half(torch.cuda.HalfTensor, fn, handle, verbose)
    for fn in utils.as_inplace(itertools.chain(tensor_overrides.FP16_FUNCS, tensor_overrides.CASTS)):
        wrap.promote_match_arg0(tensor_overrides.MODULE, fn, handle, verbose)
        if compat.tensor_is_float_tensor():
            wrap.promote_match_arg0(torch.cuda.HalfTensor, fn, handle, verbose)
            wrap.promote_match_arg0(torch.cuda.FloatTensor, fn, handle, verbose)
    if rnn_compat.has_old_rnns():
        wrap.rnn_cast(torch.nn.backends.thnn.backend, 'RNN', handle, verbose)
    if not rnn_compat.has_old_rnns():
        torch.nn.modules.rnn._VF = rnn_compat.VariableFunctionsShim()
        for x in rnn_compat.RNN_NAMES:
            wrap.new_rnn_cast(x.upper(), handle, verbose)
    rnn_compat.whitelist_rnn_cells(handle, verbose)
    for fn, err_msg in functional_overrides.BANNED_FUNCS:
        if allow_banned:
            wrap.cached_cast(functional_overrides.MODULE, fn, utils.maybe_float, handle, try_caching=True, verbose=verbose)
        else:
            wrap.err_if_any_half(functional_overrides.MODULE, fn, handle, err_msg)
    _DECORATOR_HANDLE = handle
    _amp_state.handle = handle
    return handle