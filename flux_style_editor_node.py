import os
import torch
import random
from safetensors.torch import save_file

class FluxKeyModifier:
    def __init__(self):
        self.STYLE_OPTIONS = {
            "Keys Group A": ("qkv.weight", lambda t, v: t * (1 + v / 100)),
            "Keys Group B": ("mlp.0.weight", lambda t, v: t * (1 + v / 100)),
            "Keys Group C": ("norm.key_norm.scale", lambda t, v: t * (1 + v / 100)),
            "Keys Group D": ("attn.proj.bias", lambda t, v: t + (v / 100)),
            "Keys Group E": ("img_mlp.0.weight", lambda t, v: t * (1 + v / 100)),
            "Keys Group F": ("txt_attn.qkv.weight", lambda t, v: t * (1 + v / 100)),
            "Keys Group G": ("time_in.in_layer.bias", lambda t, v: t + (v / 100)),
        }

        self.RANDOM_RANGES = {
            "Keys Group A": (-20, 20),
            "Keys Group B": (-20, 20),
            "Keys Group C": (-20, 20),
            "Keys Group D": (-20, 20),
            "Keys Group E": (-20, 20),
            "Keys Group F": (-20, 20),
            "Keys Group G": (-20, 20),
        }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = {
            "unet_model": ("MODEL",),
            "reset_model": ("BOOLEAN", {"default": True}),
            "randomize_all": ("BOOLEAN", {"default": False}),
            "save_model": ("BOOLEAN", {"default": False}),
            "save_filename": ("STRING", {"default": "Flux_keys_modified.safetensors"}),
        }
        sliders = {}
        for key in "ABCDEFG":
            group_label = f"Keys Group {key}"
            sliders[f"{group_label}"] = ("FLOAT", {"default": 0.0, "label": group_label, "min": -100.0, "max": 200.0})
            sliders[f"randomize_{group_label}"] = ("BOOLEAN", {"default": False})
        base_inputs.update(sliders)
        return {"required": base_inputs}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "random_values")
    FUNCTION = "apply_styles"
    CATEGORY = "flux/dev"

    def apply_styles(self, unet_model, reset_model=True, randomize_all=False,
                     save_model=False, save_filename="Flux_keys_modified.safetensors",
                     **kwargs):

        print(f"[DEBUG] Received unet_model type: {type(unet_model)}")

        def r(name, do_randomize, current):
            if randomize_all or do_randomize:
                min_v, max_v = self.RANDOM_RANGES[name]
                val = random.uniform(min_v, max_v)
                print(f"[RANDOM] {name}: {val:.2f}")
                return val
            return current

        style_values = {}
        for key in "ABCDEFG":
            group = f"Keys Group {key}"
            val = kwargs.get(group, 0.0)
            rand = kwargs.get(f"randomize_{group}", False)
            style_values[group] = r(group, rand, val)

        display_log = "\n".join(f"{k}: {v:.2f}" for k, v in style_values.items())

        try:
            patcher = unet_model
            base_model = patcher.model
        except AttributeError:
            raise ValueError(f"Provided model is not compatible. Expected a Comfy ModelPatcher. Got: {type(unet_model)}")

        if not hasattr(base_model, "state_dict"):
            raise ValueError("Extracted model does not have .state_dict()")

        all_keys = list(base_model.state_dict().keys())
        key_style_map = {}

        for name, (pattern, _) in self.STYLE_OPTIONS.items():
            if style_values[name] == 0:
                continue
            for k in all_keys:
                if pattern in k and k.startswith("diffusion_model."):
                    key_style_map.setdefault(name, []).append(k)

        all_matched_keys = set(k for ks in key_style_map.values() for k in ks)

        if reset_model:
            if not hasattr(patcher, "__original_state_dict__"):
                patcher.__original_state_dict__ = {}
                for pattern, _ in self.STYLE_OPTIONS.values():
                    for k in all_keys:
                        if pattern in k and k.startswith("diffusion_model.") and k not in patcher.__original_state_dict__:
                            v = base_model.state_dict()[k]
                            if isinstance(v, torch.Tensor):
                                try:
                                    patcher.__original_state_dict__[k] = v.detach().cpu().clone()
                                except Exception as e:
                                    print(f"[WARN] Failed to clone {k}: {e}")
            state_dict = {k: v.clone() for k, v in patcher.__original_state_dict__.items()}
        else:
            state_dict = {k: v for k, v in base_model.state_dict().items()
                          if k in all_matched_keys and isinstance(v, torch.Tensor)}

        modified = 0

        for name, keys in key_style_map.items():
            value = style_values[name]
            transform = self.STYLE_OPTIONS[name][1]

            for key in keys:
                try:
                    tensor = state_dict[key]
                    if not isinstance(tensor, torch.Tensor):
                        continue

                    dtype_name = getattr(tensor.dtype, "name", str(tensor.dtype))

                    if dtype_name.startswith("torch.float8"):
                        try:
                            casted_tensor = tensor.to(torch.float32)
                            new_tensor = transform(casted_tensor, value)
                            new_tensor = new_tensor.to(tensor.dtype)
                        except Exception:
                            print(f"[SKIP] {key}: float8 modification not reversible, skipped.")
                            continue
                    else:
                        new_tensor = transform(tensor, value)

                    state_dict[key] = new_tensor
                    modified += 1
                    print(f"[MOD] {name} → {key}")
                except Exception as e:
                    print(f"[FAIL] {key}: {e}")

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in state_dict and isinstance(state_dict[name], torch.Tensor):
                    try:
                        param.copy_(state_dict[name])
                    except Exception:
                        pass

        if save_model:
            try:
                output_dir = os.path.join(os.getcwd(), "output")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, save_filename)
                print(f"[SAVE] Saving modified model to {output_path}")
                full_state_dict = base_model.state_dict()
                for k in state_dict:
                    if k in full_state_dict:
                        full_state_dict[k] = state_dict[k]
                save_file(full_state_dict, output_path)
                print("[SAVE] Model saved successfully.")
            except Exception as e:
                print(f"[SAVE ERROR] Failed to save model: {e}")

        print(f"✅ {modified} tensors modified." if modified else "⚠️ No tensors modified.")

        return (unet_model, display_log)


NODE_CLASS_MAPPINGS = {
    "FluxKeyModifier": FluxKeyModifier
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKeyModifier": "Flux Keys Modifier 🧪"
}
