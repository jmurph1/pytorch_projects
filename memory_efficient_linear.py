import torch
def transformation_function(batch, linear, labels):
    x = linear(batch).float() # Up projection to large space
    from torch.nn import CrossEntropyLoss
    down_projection_function = CrossEntropyLoss(reduction = "sum")
    # Down projection to small space
    #loss = down_projection_function(x.view(-1, x.shape[-1]), labels)
    loss = down_projection_function(x, labels)
    return loss

class MemoryEfficientLinear(torch.autograd.Function):
    """ 
    Computes the application of a linear layer by micro-batch. The intended use is for the "LM Head", i.e.
    logit computation, or for a large MLP. The size of the micro-batches that are computed (default size 1) can be easily changed
    by altering the for-loops.

    Example usage: MemoryEfficientLinear.apply(input_hidden_states, linear_weight, labels, transformation_function)
    e.g. for Cross-Entropy Loss: 
    loss = torch.sum(MemoryEfficientLinear.apply(hidden_states, self.lm_head, transformation_function, (shift_labels,)))/batch_size
    """
    @staticmethod
    def forward(ctx, X, linear, labels, forward_function):
        outputs = []
        with torch.no_grad():
          for batch, batch_labels in zip(X, labels):
            output = forward_function(batch, linear, batch_labels)
            outputs.append(output)
        ctx.linear = linear
        ctx.forward_function = forward_function
        ctx.save_for_backward(X, labels)
        return torch.stack(outputs)

    @staticmethod
    def backward(ctx, dY):
        X, labels = ctx.saved_tensors
        X_grad = []
        for i, (batch, batch_labels) in enumerate(zip(X, labels)):
            batch = batch.detach().requires_grad_(True)
            with torch.enable_grad():
                output = ctx.forward_function(batch, ctx.linear, batch_labels)
            output.backward(gradient=dY[i])
            X_grad.append(batch.grad)
        return torch.stack(X_grad), None, None, None


#####################-----------------------------------------------------------------------------#########################
# Some benchmarking code adapted from Horace He -- https://gist.github.com/Chillee/22cd93e11b887db1f596ab754d60a899  + Gradient check
# To really see how memory efficient the above autograd compatible function is, recommend setting the number of tokens to
# a high number, e.g. 7000, and observe that even a small GPU can fit the computation, whereas it could only handle 
# a token size of several hundred computed normally.

import torch.nn as nn
torch.set_default_device('cuda')

# Batch Size X Token X 
B, T, D, V = 4, 256, 2048, 128256
lm_head = nn.Linear(D, V).to(torch.float32)
ce = nn.CrossEntropyLoss(reduction="sum")
x = torch.randn(B, T, D, requires_grad=True, dtype=torch.float32)
label = torch.randint(0, V, (B, T)).to(torch.int64)

print(f"Memory Allocated Pre-Activation (Tensors): {torch.cuda.memory_allocated() / 1e9:.3f} GB")
print(f"Memory Reserved Pre-Activation (Cache):   {torch.cuda.memory_reserved() / 1e9:.3f} GB")


def f(x, m, label):
    '''
    Loss calculated with no memory savings.
    '''
    out = ce(m(x).view(-1, V), label.view(-1))
    out.backward()
    grad_x = x.grad.clone()
    return out, grad_x

def efficient_f(x, m, label):
    '''
    Loss calculated without materialization of logits.
    '''
    out = MemoryEfficientLinear.apply(x, m, label, transformation_function)
    average_out = torch.sum(out)
    average_out.backward()
    grad_x = x.grad.clone()
    return average_out, grad_x

def bench(f, name=None, warmup=2, display=True, profile=False, profile_mem=False):
    import time
    from triton.testing import do_bench
    for _ in range(warmup):
        f()

    if profile_mem:
        torch.cuda.memory._record_memory_history()
        f()
        torch.cuda.memory._dump_snapshot(f"{name if name is not None else 'memory'}.pickle")
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.reset_peak_memory_stats()
    ms_per_iter = do_bench(lambda: f())
    if name is None:
        res = ms_per_iter
    else:
        res= f"{name}: {ms_per_iter:.3f}ms"
    if display:
        print(res)
        print(f"Peak mem: {torch.cuda.max_memory_allocated()/1e9:.3f}")
        print()
    return "Complete"


bench(lambda: f(x, lm_head, label), name='Regular')
import gc
gc.collect()
torch.cuda.empty_cache()
bench(lambda: efficient_f(x, lm_head, label), name='MemEfficient')

# Compare Gradients
x.grad.zero_()
_, grad_regular = f(x, lm_head, label)

x.grad.zero_()

_, grad_efficient = efficient_f(x, lm_head, label)

# Compare gradients
if grad_regular is not None and grad_efficient is not None:
    print("\n--- Comparing Gradients ---")
    print(f"Gradients are close: {torch.allclose(grad_regular, grad_efficient)}")
    diff = torch.abs(grad_regular - grad_efficient)
    print(f"Max absolute difference: {torch.max(diff).item()}")
    print(f"Mean absolute difference: {torch.mean(diff).item()}")
else:
    print("Could not compare gradients (one or both are None)")
