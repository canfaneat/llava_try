#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=True, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    # Remove the explicit setting of device_map here, let from_pretrained handle it based on parameters passed or defaults.
    # kwargs = {"device_map": device_map, **kwargs}

    # Keep the logic for handling specific device target if not cuda
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                local_files_only=True,
                **kwargs
            )
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # Non-LoRA case: model_path has projector, model_base has LLM
            print(f'Loading LLM base model from: {model_base} locally')
            # Load base tokenizer locally
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)

            # Load base config (e.g., LlamaConfig) first
            print(f'Loading base model config from: {model_base}')
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True)

            # Load LLaVA specific configuration from model_path
            print(f'Loading LLaVA configuration from: {model_path}')
            llava_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True) # Use trust_remote_code for safety

            # Update base config with LLaVA specific fields
            # Iterate through llava_config and update base_config if the field exists in llava_config
            print("Merging LLaVA config fields into base config...")
            for key, value in llava_config.to_dict().items():
                # Only update fields specifically relevant to LLaVA or mm parts if needed,
                # or simply update all fields present in llava_config.
                # A safer approach might be to list LLaVA specific keys, but updating all is common.
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
                else:
                    # If the key doesn't exist in base_config, add it (necessary for mm fields)
                    setattr(base_config, key, value)

            # Determine the correct LLaVA class based on the merged config
            if 'mpt' in base_config.model_type: # Check merged config for model type
                llava_class = LlavaMptForCausalLM
                print("Loading model as LlavaMptForCausalLM using base path and merged config...")
            elif 'mistral' in base_config.model_type:
                 llava_class = LlavaMistralForCausalLM
                 print("Loading model as LlavaMistralForCausalLM using base path and merged config...")
            else: # Assume Llama
                llava_class = LlavaLlamaForCausalLM
                print("Loading model as LlavaLlamaForCausalLM using base path and merged config...")

            # Load the model using the LLaVA class, pointing to the base path for weights,
            # applying the MERGED config, forcing local files, and explicitly mapping to cuda:0.
            model = llava_class.from_pretrained(
                model_base,             # Load weights FROM BASE path
                low_cpu_mem_usage=True,
                config=base_config,     # Apply the MERGED config
                local_files_only=True,  # Ensure base weights are loaded locally
                # Force mapping to cuda:0, overriding potential auto-mapping or offload
                device_map="cuda:0",
                # Remove offload_folder if explicitly mapping to GPU
                # offload_folder=offload_folder,
                **kwargs
            )

            # Load the mm_projector weights from model_path
            print("Loading mm_projector weights...")
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
            print("MM projector weights loaded.")
        else:
            # model_base is None, load full model from model_path
            print(f"Loading full model from: {model_path} locally")
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    low_cpu_mem_usage=True, 
                    local_files_only=True, # Ensure local loading
                    **kwargs
                )

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        
        if not vision_tower.is_loaded:
            # 强制在目标设备上加载视觉塔
            target_device = device if device == 'cuda' else 'cpu'
            print(f"Attempting to load vision tower directly to device: {target_device}")
            try:
                # 主要尝试：直接加载到目标设备
                vision_tower.load_model(device_map={"": target_device})
                vision_tower.to(dtype=torch.float16) # 确保dtype
                print("Vision tower loaded successfully to target device.")
            except Exception as e:
                print(f"Direct load to target device failed: {e}")
                print("Retrying vision tower load with device_map=None and explicit .to()")
                # 备选方案：加载时不指定复杂映射，然后强制移动
                vision_tower.load_model(device_map=None)
                vision_tower.to(device=target_device, dtype=torch.float16)
                print("Vision tower loaded with fallback method.")

        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

