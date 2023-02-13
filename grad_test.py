import torch

torch.set_default_dtype(torch.float64)
batch_size = 128
softmax_dim = 128

def test_multiplicative_grad():
    for i in range(30):
        values = torch.randn(batch_size, softmax_dim)
        logits = torch.randn(batch_size, softmax_dim)
        mask = torch.rand(batch_size, softmax_dim)
        mask = mask / 10**i
        mask = torch.maximum(mask, torch.eye(softmax_dim))
        # print("Mask")
        # print(mask[:5, :5])
        # print(torch.mean(mask))
        mask.requires_grad = True

        weights = torch.softmax(logits, dim=1)
        weights = weights * mask / torch.sum(weights * mask, dim=1, keepdim=True)
        output = torch.sum(values * weights, dim=1)
        loss = torch.sum(output)

        loss.backward()
        print("Grad")
        print(torch.max(torch.abs(mask.grad)))

def test_logsumexp_grad():
    for i in range(20):
        values = torch.randn(batch_size, softmax_dim)
        logits = torch.randn(batch_size, softmax_dim)
        logits.requires_grad = True
        mask = torch.rand(batch_size, softmax_dim)
        mask = mask / 10**i
        mask = torch.maximum(mask, torch.eye(softmax_dim))
        # print("Mask")
        # print(mask[:5, :5])
        # print(torch.mean(mask))
        mask.requires_grad = True

        log_mask = torch.log(mask)
        log_mask.retain_grad()
        new_logits = logits + log_mask
        new_logits.retain_grad()
        weights = torch.softmax(new_logits, dim=1)
        weights.retain_grad()
        # mult_weights = torch.softmax(logits, dim=1) * mask / torch.sum(torch.softmax(logits, dim=1) * mask, dim=1, keepdim=True)
        output = torch.sum(values * weights, dim=1)
        output.retain_grad()
        loss = torch.sum(output)

        loss.backward()
        print("Grad")
        # print(mask.grad[:5, :5])
        print(torch.max(torch.abs(weights.grad)))
        print(torch.max(torch.abs(new_logits.grad)))
        print(torch.max(torch.abs(log_mask.grad)))
        print(torch.max(torch.abs(mask.grad)))
        print(torch.max(torch.abs(logits.grad)))

if __name__ == "__main__":
    # test_multiplicative_grad()
    test_logsumexp_grad()