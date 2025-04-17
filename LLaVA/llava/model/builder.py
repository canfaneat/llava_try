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
from llava.model import * # This should import necessary Llava classes like LlavaLlamaForCausalLM etc.
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower # Ensure this import works


# Ensure default precision is float16 (load_8bit=False, load_4bit=False)
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    # Keep the logic for handling specific device target if not cuda
    if device != "cuda":
        # Force device_map to specified device if not cuda
        kwargs['device_map'] = {"": device}
    else:
        # Pass the intended device_map (e.g., "auto") for CUDA case
        kwargs['device_map'] = device_map

    # Ensure 8bit/4bit flags are processed if passed, but default is False
    if load_8bit:
        print("Loading model in 8bit")
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        print("Loading model in 4bit")
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        print("Loading model in float16")
        # Default to float16 if not 8/4 bit
        kwargs['torch_dtype'] = torch.float16

    # Flash attention (keep logic if needed)
    if use_flash_attn:
        print("Using Flash Attention 2")
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            print("Loading LLaVA LoRA weights")
            from llava.model.language_model.llava_llama import LlavaConfig # Ensure correct import path
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path, local_files_only=True) # Use local files for LoRA config too
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            print('Loading LLaVA from base model...')
            # Load base model first, apply LoRA config, use device_map from kwargs
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, local_files_only=True, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub - attempt local load if possible
                from huggingface_hub import hf_hub_download
                try:
                    print(f"Attempting to load non_lora_trainables.bin locally from {model_path}")
                    cache_file = hf_hub_download(
                        repo_id=model_path, # Assume model_path might be a Hub ID if file not local
                        filename='non_lora_trainables.bin',
                        local_files_only=True, # Try local first
                        cache_dir=kwargs.get('cache_dir'),
                        force_download=kwargs.get('force_download'),
                        proxies=kwargs.get('proxies'),
                        resume_download=kwargs.get('resume_download'),
                        token=kwargs.get('token'),
                        user_agent=kwargs.get('user_agent'),
                        subfolder=kwargs.get('subfolder'),
                    )
                    non_lora_trainables = torch.load(cache_file, map_location='cpu')
                except Exception as e:
                     warnings.warn(f"Could not load non_lora_trainables.bin locally, may need network if from Hub: {e}")
                     # Fallback or re-raise depending on desired behavior
                     raise e # Or handle error

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
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True, local_files_only=True)

            # Load LLaVA specific configuration from model_path
            print(f'Loading LLaVA configuration from: {model_path}')
            llava_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

            # Update base config with LLaVA specific fields
            print("Merging LLaVA config fields into base config...")
            llava_config_dict = llava_config.to_dict()
            for key, value in llava_config_dict.items():
                 # Add or update fields in base_config
                 setattr(base_config, key, value)

            # Determine the correct LLaVA class based on the merged config
            # Use model_type from the merged config
            model_type = getattr(base_config, "model_type", "")
            if 'mpt' in model_type:
                llava_class = LlavaMptForCausalLM
                print("Determined model type: LlavaMptForCausalLM")
            elif 'mistral' in model_type:
                llava_class = LlavaMistralForCausalLM
                print("Determined model type: LlavaMistralForCausalLM")
            else: # Assume Llama
                llava_class = LlavaLlamaForCausalLM
                print("Determined model type: LlavaLlamaForCausalLM")

            print(f"Loading model as {llava_class.__name__} using base path and merged config...")
            # Load the model using the LLaVA class, pointing to the base path for weights,
            # applying the MERGED config, forcing local files, and using device_map='auto'.
            model = llava_class.from_pretrained(
                model_base,             # Load weights FROM BASE path
                low_cpu_mem_usage=True, # Use low_cpu_mem_usage if available
                config=base_config,     # Apply the MERGED config
                local_files_only=True,  # Ensure base weights are loaded locally
                # device_map="auto",      # Use device_map from kwargs which should be "auto" for CUDA
                **kwargs # Pass merged kwargs including device_map and dtype
            )

            # Load the mm_projector weights from model_path
            print("Loading mm_projector weights...")
            projector_path = os.path.join(model_path, 'mm_projector.bin')
            if os.path.exists(projector_path):
                mm_projector_weights = torch.load(projector_path, map_location='cpu')
                mm_projector_weights = {k: v.to(kwargs.get('torch_dtype', torch.float16)) for k, v in mm_projector_weights.items()} # Use target dtype
                model.load_state_dict(mm_projector_weights, strict=False)
                print("MM projector weights loaded.")
            else:
                warnings.warn(f"MM projector file not found at {projector_path}")

        else:
            # model_base is None, load full LLaVA model from model_path
            print(f"Loading full LLaVA model from: {model_path} locally")
            # Determine model type from config in model_path
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            model_type = getattr(config, "model_type", "")

            if 'mpt' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs) # Pass kwargs (contains device_map, dtype)
            elif 'mistral' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs) # Pass kwargs
            else: # Assume Llama
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs) # Pass kwargs
    else:
        # Load language model only (non-LLaVA)
        print(f"Loading non-LLaVA model: {model_name}")
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            print("Loading PEFT model")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, local_files_only=True, **kwargs) # Pass kwargs
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path, local_files_only=True) # Try local for adapter too
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to target dtype...') # Usually float16
            model.to(kwargs.get('torch_dtype', torch.float16))
        else:
            # Standard model
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            model_type = getattr(config, "model_type", "")
            use_fast = False
            if 'mpt' in model_type:
                use_fast = True
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True, **kwargs) # Pass kwargs

    image_processor = None

    # Handle vision tower loading consistently
    if 'llava' in model_name.lower() or getattr(model.config, 'mm_vision_tower', None):
        print("Processing LLaVA/Vision related components...")
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()

        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_config_path_or_hf_id = getattr(model.config, "mm_vision_tower", "openai/clip-vit-large-patch14-336")
                kaggle_clip_path = "/kaggle/input/openai-clip-vit-large-patch14-336" # Target Kaggle path

                # Check if Kaggle path exists and seems valid
                final_vision_path = vision_config_path_or_hf_id # Default to config value
                if os.path.exists(kaggle_clip_path) and os.path.isfile(os.path.join(kaggle_clip_path, 'config.json')):
                     print(f"Attempting to use CLIP model from Kaggle Input: {kaggle_clip_path}")
                     final_vision_path = kaggle_clip_path
                else:
                    print(f"Kaggle Input path for CLIP ('{kaggle_clip_path}') not found or invalid, using value from config: '{vision_config_path_or_hf_id}'")

                print(f"Loading vision tower from: {final_vision_path}")
                # Update internal name before loading if using local path
                vision_tower.vision_tower_name = final_vision_path
                # Load vision tower without specific device_map, let it follow the main model's map
                # Pass target dtype from main kwargs
                vision_tower.load_model(device_map=None, torch_dtype=kwargs.get('torch_dtype', torch.float16))
                print("Vision tower loaded.")
            image_processor = vision_tower.image_processor
        else:
            warnings.warn("Vision tower is None, image processing won't be available.")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048 # Default LLaMA context length

    return tokenizer, model, image_processor, context_len
