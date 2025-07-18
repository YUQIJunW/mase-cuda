import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModel
import tabulate
import logging
import importlib
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_layer_by_name(model, layer_name):
    parts = layer_name.split('.')
    for part in parts:
        if part.isdigit():
            model = model[int(part)]
        else:
            model = getattr(model, part)
    return model

model_name = "JeremiahZ/bert-base-uncased-mrpc"
layer_name = "encoder.layer.0.attention.self.query.weight"
data_types = [
    "mxfp4_E2M1",
    "mxfp6_E2M3",
    "mxfp6_E3M2",
    "mxfp8_E4M3",
    "mxfp8_E5M2"
]

print(f"Loading model: {model_name}")
model = AutoModel.from_pretrained(model_name)

# for name, param in model.named_parameters():
#     print(f"{name:60s} {tuple(param.shape)}")

print(f"Extracting weight from: {layer_name}")
raw_tensor = get_layer_by_name(model, layer_name).detach().cpu()
raw_flat = raw_tensor.numpy().flatten()


from scipy.stats import gaussian_kde
density = gaussian_kde(raw_flat)
xs = np.linspace(min(raw_flat), max(raw_flat), 500)
ys = density(xs)

plt.plot(xs, ys)
plt.title("Weight Value Density Curve")
plt.xlabel("Weight Value")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_tensor = raw_tensor.to(device).to(torch.bfloat16).flatten()

group_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
all_rows = []

for data_type in tqdm(data_types):
    try:
        module = importlib.import_module(f"mase_cuda.{data_type}.linear")
        PackedWeight = getattr(module, "PackedWeight")
    except Exception as e:
        logger.warning(f"Failed to load {data_type}: {e}")
        continue

    for group_size in group_sizes:
        try:
            packed = PackedWeight.pack_simulated(weight_tensor, group_size)
            unpacked = packed.unpack()
            mae = torch.mean(torch.abs(weight_tensor - unpacked)).item()
            all_rows.append([data_type, tuple(weight_tensor.shape), group_size, mae])
        except Exception as e:
            logger.warning(f"Failed at {data_type}, group {group_size}: {e}")


headers = ["DataType", "Shape", "Group Size", "MAE"]
table = tabulate.tabulate(all_rows, headers=headers, tablefmt="pipe")
logger.info(f"\n{table}")
