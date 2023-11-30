import torch
device = "cuda:1" if torch.cuda.is_available() else "cpu"

a0= torch.tensor([0,1,3])
a1 = torch.tensor(5)
a = torch.tensor({"state": a0, "action": a1})


print(a)
print(a["state"])
print(a["action"])
