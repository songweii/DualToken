import torch
from collections import OrderedDict

src = "/mnt/public/users/songwei/code/DualToken-causal/epoch_7.pt"
dst = "/mnt/public/users/songwei/code/DualToken-causal/ckpt/siglip2-256-rvq4-1152d/epoch_7.pt"

ckpt = torch.load(src, map_location="cpu")
old_sd = ckpt["state_dict"]

new_sd = OrderedDict()
num_changed = 0

for k, v in old_sd.items():
    new_k = k.replace("semantic", "sem").replace("pixel", "pix")
    if new_k != k:
        num_changed += 1
    if new_k in new_sd:
        raise ValueError(f"Key collision after rename: {k} -> {new_k}")
    new_sd[new_k] = v

new_ckpt = {"state_dict": new_sd}
torch.save(new_ckpt, dst)

print(f"original keys: {len(old_sd)}")
print(f"renamed keys: {len(new_sd)}")
print(f"changed keys: {num_changed}")
print(f"top-level fields in new file: {list(new_ckpt.keys())}")
print(f"saved to: {dst}")