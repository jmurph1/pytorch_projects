"""
PyTorch Custom Operator for BitsAndBytes 4-bit matrix multiplication kernel compatibility with torch.compile (i.e. so that it compiles
without graph breaks). Monkey patches that demonstrate the integration are at the bottom.
"""


import torch
from torch import Tensor
from typing import Optional, Tuple, List, Dict, Any
import bitsandbytes as bnb
import bitsandbytes.functional as bnb_F

# Helper Function to prepare to obtain QuantState, by correctly arranging the attributes from list into dict
# (custom_op can only take a List of Tensors, as a representation for QuantState attributes)
def get_quant_state_dict(padSize, quant_state_list):
    qsl_copy = quant_state_list.copy()
    quant_state_dict = {}
    quant_state_dict["absmax"] = qsl_copy.pop(0)
    quant_state_dict["quant_map"] = qsl_copy.pop(0)
    quant_state_dict["nested_absmax"] = qsl_copy.pop(0)
    quant_state_dict["nested_quant_map"] = qsl_copy.pop(0)
    quant_state_dict["quant_state.bitsandbytes__nf4"] = qsl_copy.pop(0)[0:(169-padSize.item())]
    return quant_state_dict

# Define BnB MatMul4Bit as a custom operator
@torch.library.custom_op("unslothProblemC::new_matmul_4bit", mutates_args=())
def new_matmul_4bit(A: Tensor, B: Tensor, in_features: int, out_features: int, padSize: Tensor, quant_state_list: List[Tensor]) -> Tensor:
    quant_state = bnb_F.QuantState.from_dict(qs_dict=get_quant_state_dict(padSize, quant_state_list), device="cuda")
    return torch.nn.functional.linear(A, bnb_F.dequantize_4bit(B, quant_state).to(A.dtype).t())

# Register fake tensor implementation
@torch.library.register_fake("unslothProblemC::new_matmul_4bit")
def _(A, B, in_features, out_features, padSize, quant_state_list):
    output_shape = A.shape[:-1] + (out_features,)
    return torch.empty(output_shape, dtype=A.dtype, device=A.device)

#Define BnB MatMul4Bit.backward as a custom operator
@torch.library.custom_op("unslothProblemC::new_matmul_4bit_backward", mutates_args=())
def new_matmul_4bit_backward(grad_output: Tensor, B: Tensor, in_features: int, padSize: Tensor, quant_state_list: List[Tensor]) -> Tensor:
    quant_state = bnb_F.QuantState.from_dict(qs_dict=get_quant_state_dict(padSize, quant_state_list), device="cuda")
    return torch.matmul(grad_output, bnb_F.dequantize_4bit(B, quant_state).to(grad_output.dtype).t())

# Register fake tensor implementation for backward
@torch.library.register_fake("unslothProblemC::new_matmul_4bit_backward")
def _(grad_output, B, in_features, padSize, quant_state_list):
    output_shape = grad_output.shape[:-1] + (in_features,)
    return torch.empty(output_shape, dtype=grad_output.dtype, device=grad_output.device)

# Define autograd functions
def setup_context(ctx, inputs, output):
    _, B, in_features, _, padSize, quant_state_list = inputs
    ctx.save_for_backward(B)
    ctx.in_features = in_features
    ctx.padSize = padSize
    ctx.quant_state_list = quant_state_list

def backward_fn(ctx, grad_output):
    B, = ctx.saved_tensors
    in_features = ctx.in_features
    padSize = ctx.padSize
    quant_state_list = ctx.quant_state_list
    grad_A = new_matmul_4bit_backward(grad_output, B, in_features, padSize, quant_state_list)
    grad_quant_state_list = [None] * len(quant_state_list)
    return grad_A, None, None, None, None, grad_quant_state_list

# Register autograd
torch.library.register_autograd("unslothProblemC::new_matmul_4bit", backward_fn, setup_context=setup_context)

# Patch bnb.matmul_4bit
bnb.matmul_4bit = new_matmul_4bit


####------------------------------------------------------------------------------------------#############

def serialize_attributes(quant_state):
        """
        returns ordered list of tensors to use in serialization to pass to custom_op
        """
        nonTensor_attr_dict = {
            "shape": tuple(quant_state.shape),
            "quant_type": quant_state.quant_type,
            "blocksize": quant_state.blocksize,
            "dtype": "float16",
            "nested_blocksize": quant_state.state2.blocksize,
            "nested_dtype": "float16",
            "nested_offset": quant_state.offset.item(),
        }
        nonTensor_attr_tensor = bnb.utils.pack_dict_to_tensor(nonTensor_attr_dict)
        # Pad nonTensor_attr_tensor for consistent shape after compilation
        padSize = 169 - len(nonTensor_attr_tensor)
        nonTensor_attr_tensor = torch.cat((nonTensor_attr_tensor, torch.zeros(padSize, dtype=torch.uint8)))

        absmax = quant_state.absmax
        quant_map = quant_state.code
        nested_absmax = quant_state.state2.absmax
        nested_quant_map = quant_state.state2.code.clone()

        return torch.tensor(padSize), [absmax, quant_map, nested_absmax, nested_quant_map, nonTensor_attr_tensor]


def new_Linear4bit_forward(self, x: torch.Tensor):
    if not self.compute_type_is_set:
        self.set_compute_type(x)
        self.compute_type_is_set = True

    inp_dtype = x.dtype
    if self.compute_dtype is not None:
        x = x.to(self.compute_dtype)

    return bnb.matmul_4bit(x, self.weight.transposed, self.in_features, self.out_features, self.weight.padSize, quant_state_list=self.weight.quant_state_serialized).to(inp_dtype)

bnb.nn.Linear4bit.forward = new_Linear4bit_forward

@classmethod
def new_from_prequantized(
    cls,
    data: torch.Tensor,
    quantized_stats: Dict[str, Any],
    requires_grad: bool = False,
    device="cuda",
    module: Optional["Linear4bit"] = None,
    **kwargs,
) -> "bnb.nn.Params4bit":
    self = torch.Tensor._make_subclass(cls, data.to(device))
    self.requires_grad = requires_grad

    self.quant_state = bnb_F.QuantState.from_dict(qs_dict=quantized_stats, device=device)
    self.padSize, self.quant_state_serialized = serialize_attributes(self.quant_state)
    self.transposed = self.t()

    self.blocksize = self.quant_state.blocksize
    self.compress_statistics = self.quant_state.nested
    self.quant_type = self.quant_state.quant_type
    self.bnb_quantized = True

    self.quant_storage = data.dtype
    self.module = module

    if self.module is not None:
        self.module.quant_state = self.quant_state

    return self

bnb.nn.Params4bit.from_prequantized = new_from_prequantized
