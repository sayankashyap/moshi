from dataclasses import dataclass
import math
import torch
from torch import nn
from weakref import ref, WeakKeyDictionary, WeakSet

from .compile import torch_compile_lazy


def _quantize_weight_int8(weight: torch.Tensor, block_size: int = 256):
    blocks = int(math.ceil(weight.numel() / block_size))
    if weight.numel() != blocks * block_size:
        weight_tmp = torch.empty(block_size, block_size, device=weight.device, dtype=weight.dtype)
        weight_tmp.view(-1)[:weight.numel()] = weight.view(-1)
        weight_tmp.view(-1)[weight.numel():].zero_()
        weight = weight
    mn, mx = weight.aminmax(dim=1, keepdim=True)
    mx_val = 127
    mn_val = -128

    # we want mx_val * scale + mean = mx (1)
    # and     mn_val * scale + mean = mn (2)
    # so doing (2) - (1) we get
    # scale * (mx_val - mn_val) = mx - mn
    scale = (mx - mn) / (mx_val - mn_val)
    # now we have, if scale is not generate.
    zero = torch.zeros(1, device=weight.device, dtype=weight.dtype)
    invalid = scale == 0
    mean = torch.where(invalid, zero, mx - scale * mx_val)
    q8 = torch.where(invalid, zero, (weight - mean) / scale).round().clamp(mn_val, mx_val).to(torch.int8)
    return q8, scale, mean


@torch_compile_lazy
def _unquantize_weight_int8(q8: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor, size: torch.Size) -> torch.Tensor:
    w = q8.to(mean.dtype)
    torch.addcmul(mean, w, scale, out=w)
    w = w.view(-1)[:size.numel()].view(size)
    return w


@dataclass
class _QuantizedParam:
    _module: ref[nn.Module]
    name: str
    size: torch.Size

    @property
    def module(self) -> nn.Module:
        module = self._module()
        if module is None:
            raise RuntimeError("Module is gone but we are still here.")
        return module

    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @property
    def mean(self) -> torch.Tensor:
        return getattr(self.module, self.name + '_mean')

    @property
    def scale(self) -> torch.Tensor:
        return getattr(self.module, self.name + '_scale')

    @property
    def q8(self) -> torch.Tensor:
        return getattr(self.module, self.name + '_q8')

    def quantize_(self, block_size: int = 256) -> None:
        w = getattr(self.module, self.name)
        q8, scale, mean = _quantize_weight_int8(w, block_size)
        setattr(self.module, self.name + '_q8', nn.Parameter(q8, requires_grad=False))
        setattr(self.module, self.name + '_scale', nn.Parameter(scale, requires_grad=False))
        setattr(self.module, self.name + '_mean', nn.Parameter(mean, requires_grad=False))

    def unquantized_value(self) -> torch.Tensor:
        w = _unquantize_weight_int8(self.q8, self.scale, self.mean, self.size)
        return w

    def unquantize_(self) -> None:
        if hasattr(self.module, self.name):
            return
        w = self.unquantized_value()
        setattr(self.module, self.name, nn.Parameter(w, requires_grad=False))

    def flush_unquantized_(self) -> None:
        delattr(self.module, self.name)


_params_registry: WeakKeyDictionary[nn.Module, dict[str, _QuantizedParam]] = WeakKeyDictionary()
_quantized_registry: WeakSet[nn.Module] = WeakSet()


def quantize_module_int8_(module: nn.Module, min_size_mb: float = 1., block_size: int = 256):
    if module in _quantized_registry:
        return
    params: list[_QuantizedParam] = []

    def pre_hook(mod, args) -> None:
        for param in params:
            param.unquantize_()

    def post_hook(mod, args, output) -> None:
        for param in params:
            param.flush_unquantized_()

    seen: set[nn.Module] = set()
    for child in module.modules():
        if child in seen:
            continue
        elif child in _params_registry:
            # Someone else already quantized this module.
            params.extend(_params_registry[child].values())
            continue

        seen.add(child)
        child_params = {}
        for name, weight in list(child.named_parameters(recurse=False)):
            size = weight.numel() * weight.dtype.itemsize / 1e6
            if size < min_size_mb:
                continue

            param = _QuantizedParam(ref(child), name, weight.size())
            param.quantize_()
            param.flush_unquantized_()
            params.append(param)
            child_params[param.name] = param
        _params_registry[child] = child_params
    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)
    _quantized_registry.add(module)
    return module

def get_size_dtype_device(module: nn.Module, name: str) -> tuple[torch.Size, torch.dtype, torch.device]:
    try:
        tensor = getattr(module, name)
    except AttributeError:
        if module in _params_registry:
            params = _params_registry[module]
            if name in params:
                param = params[name]
                return param.size, param.dtype, param.device
        raise
    else:
        return tensor.size(), tensor.dtype, tensor.device


