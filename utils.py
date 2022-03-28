import torch

def get_torch_item(t: torch.Tensor):
    return t.cpu().detach().item()

def get_max(t: torch.Tensor):
    m = torch.max(t)
    return get_torch_item(m)

def get_arg_max(t: torch.Tensor):
    a = torch.argmax(t)
    return get_torch_item(a)

def to_tensor(arr, device="cpu") -> torch.Tensor:
    return torch.Tensor(arr).to(device)

def to_np(t: torch.Tensor):
    return t.cpu().detach().numpy()