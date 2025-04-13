# ComfyUI-Flux_Keys_Modifier

This custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) allows you to dynamically modify specific tensor groups within a **UNet-only** model (like FLUX) using a variety of control sliders. The model can be reset between generations or optionally saved.

## ✨ Features

- Apply 7 grouped style modifications to UNet weights
- Individually randomize each group, or randomize all at once
- Reset model to original weights before each run (optional)
- Save modified model as `.safetensors` to the `output/` folder
- Compatible with standard `MODEL` input/output ports in ComfyUI
- Designed for FP8/FP16/FP32 models: safe float8 handling (modifies only what can be restored)

## ⚙️ Parameters

| Parameter         | Description |
|------------------|-------------|
| `unet_model`      | UNet model (from Model Loader) |
| `reset_model`     | Reset modified tensors to their original values before applying changes |
| `randomize_all`   | Randomize all sliders (overrides individual toggles) |
| `save_model`      | Save the modified model to disk |
| `save_filename`   | Name of the saved file (`output/Flux_keys_modified.safetensors` by default) |
| `Keys Group A–G`  | Style sliders (-100 to 200). Adjust rendering behavior |
| `randomize_X`     | Toggle randomization for each slider |

## 💡 Keys Group Legend

These are abstracted tensor categories — not strict visual effects:

- **Keys Group A** → `qkv.weight`
- **Keys Group B** → `mlp.0.weight`
- **Keys Group C** → `norm.key_norm.scale`
- **Keys Group D** → `attn.proj.bias`
- **Keys Group E** → `img_mlp.0.weight`
- **Keys Group F** → `txt_attn.qkv.weight`
- **Keys Group G** → `time_in.in_layer.bias`
