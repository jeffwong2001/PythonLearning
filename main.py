import torch

w = torch.tensor(10, requires_grad=True, dtype=torch.float32)
learning_rate = 1e-2

for epoch in range(1,101):
    loss = 2 * w ** 2

    if w.grad is not None:
        w.grad.zero_()

    loss.sum().backward()

    w.data = w.data - learning_rate * w.grad

    print(f'第 {epoch}次以后, w更新以后是{w.data:5f}, loss的值是{loss.data:5f}')