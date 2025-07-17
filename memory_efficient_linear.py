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
