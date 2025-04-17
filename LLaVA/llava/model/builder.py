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
# Make sure necessary LLaVA classes are imported correctly
from llava.model import LlavaLlamaForCausalLM, LlavaMistralForCausalLM, LlavaMptForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower # Ensure this import works


# Ensure default precision is float16 (load_8bit=False, load_4bit=False)
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):

    # Handle device mapping for CUDA vs non-CUDA
    if device != "cuda":
        kwargs['device_map'] = {"": device}
        print(f"Warning: Loading model on device '{device}'. CUDA acceleration will be unavailable.")
    else:
        # Pass the intended device_map (e.g., "auto") for CUDA case
        kwargs['device_map'] = device_map
        print(f"Using device_map: {kwargs['device_map']}")

    # Handle quantization flags
    if load_8bit:
        print("Loading model in 8bit")
        kwargs['load_in_8bit'] = True
        # Ensure torch_dtype is not set when using 8bit
        kwargs.pop('torch_dtype', None)
    elif load_4bit:
        print("Loading model in 4bit")
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        # Ensure torch_dtype is not set when using 4bit
        kwargs.pop('torch_dtype', None)
    else:
        print("Loading model in float16")
        kwargs['torch_dtype'] = torch.float16

    # Flash attention (Optional)
    if use_flash_attn:
        # Check if FlashAttention is available and compatible before setting
        # (Simple check here, more robust checks might be needed)
        try:
            import flash_attn
            print("Setting attn_implementation to flash_attention_2")
            kwargs['attn_implementation'] = 'flash_attention_2'
        except ImportError:
            print("FlashAttention not installed, using default attention.")


    if 'llava' in model_name.lower():
        # Load LLaVA model variations
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('LoRA model specified but no model_base provided. Loading LoRA might fail.')
            # Attempt to load as full model from model_path, but behavior might be undefined
            print(f"Warning: Attempting to load LoRA model from {model_path} without base.")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
        elif 'lora' in model_name.lower() and model_base is not None:
            # Load LLaVA LoRA model
            print("Loading LLaVA LoRA weights")
            from llava.model.language_model.llava_llama import LlavaConfig # Ensure correct import path
            # Load base tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            # Load base config
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True, local_files_only=True)
            # Load LoRA specific config from model_path
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path, local_files_only=True)
            # Merge LoRA specific attributes into base config (carefully)
            print("Merging LoRA config fields into base config...")
            for key, value in lora_cfg_pretrained.to_dict().items():
                 setattr(base_config, key, value) # Assume LoRA config keys are needed

            print('Loading LLaVA from base model...')
            # Load base model weights using the merged config
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=base_config, # Use the merged config
                local_files_only=True,
                **kwargs # Pass device_map, dtype etc.
            )

            # Handle token embeddings resizing if needed (copied from original logic)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            # Load non-LoRA trainable weights (projector etc.)
            print('Loading additional LLaVA weights (non-lora trainables)...')
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
            else:
                 warnings.warn(f"non_lora_trainables.bin not found locally at {non_lora_path}. If loading from Hub, ensure network access.")
                 # Optional: Attempt Hub download if needed, requires careful path handling
                 non_lora_trainables = {} # Assign empty dict if not found

            # Clean keys and load state dict
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            # Load LoRA adapter weights
            from peft import PeftModel
            print('Loading LoRA adapter weights...')
            model = PeftModel.from_pretrained(model, model_path, local_files_only=True) # Specify local loading for adapter
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('LoRA Model is loaded and merged...')

        elif model_base is not None:
            # Non-LoRA case: model_path has projector, model_base has LLM weights + base config
            print(f'Loading LLM base model from: {model_base} locally')
            # Load base tokenizer locally
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)

            # Load base config (e.g., LlamaConfig) first
            print(f'Loading base model config from: {model_base}')
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True, local_files_only=True)

            # Load LLaVA specific configuration from model_path
            print(f'Loading LLaVA configuration from: {model_path}')
            llava_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

            # **Crucial Change:** Update base config with ONLY LLaVA-specific multimodal attributes
            # Avoid overwriting core LLM config attributes like decoder/generation params
            print("Carefully merging necessary LLaVA config fields into base config...")
            llava_specific_keys = [
                "mm_vision_tower", "mm_hidden_size", "mm_projector_type",
                "mm_vision_select_layer", "mm_vision_select_feature",
                "mm_use_im_start_end", "mm_use_im_patch_token",
                "image_aspect_ratio", # Add other relevant mm_ keys if needed
            ]
            for key in llava_specific_keys:
                if hasattr(llava_config, key):
                    value = getattr(llava_config, key)
                    print(f"  Updating {key} = {value}")
                    setattr(base_config, key, value)

            # Determine the correct LLaVA class based on the *base* config's model_type
            # (The base config should dictate the LLM architecture)
            model_type = getattr(base_config, "model_type", "")
            if 'mpt' in model_type:
                llava_class = LlavaMptForCausalLM
                print("Determined base model type: MPT")
            elif 'mistral' in model_type:
                llava_class = LlavaMistralForCausalLM
                print("Determined base model type: Mistral")
            else: # Assume Llama
                llava_class = LlavaLlamaForCausalLM
                print("Determined base model type: Llama")

            print(f"Loading model as {llava_class.__name__} using base path and updated base config...")
            # Load the model using the LLaVA class, pointing to the base path for weights,
            # applying the *updated* base_config (which contains core LLM + necessary MM fields).
            model = llava_class.from_pretrained(
                model_base,             # Load weights FROM BASE path
                low_cpu_mem_usage=True,
                config=base_config,     # Apply the *updated* base_config
                local_files_only=True,  # Ensure base weights are loaded locally
                # device_map="auto",      # Use device_map from kwargs
                **kwargs                # Pass merged kwargs including device_map and dtype
            )

            # Load the mm_projector weights from model_path
            print("Loading mm_projector weights...")
            projector_path = os.path.join(model_path, 'mm_projector.bin')
            if os.path.exists(projector_path):
                mm_projector_weights = torch.load(projector_path, map_location='cpu')
                # Use target dtype specified in kwargs or default to float16
                target_dtype = kwargs.get('torch_dtype', torch.float16)
                mm_projector_weights = {k: v.to(target_dtype) for k, v in mm_projector_weights.items()}
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

            # Select class based on model type found in the full model's config
            if 'mpt' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
            elif 'mistral' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
            else: # Assume Llama
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
    else:
        # Load standard language model only (non-LLaVA)
        print(f"Loading standard non-LLaVA model: {model_name}")
        if model_base is not None: # This case seems less likely for non-LLaVA, maybe PEFT?
            from peft import PeftModel
            print("Loading PEFT model")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=True)
            # Load base model using AutoClass
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, local_files_only=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to target dtype...')
            model.to(kwargs.get('torch_dtype', torch.float16))
        else:
            # Standard model from model_path
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            model_type = getattr(config, "model_type", "")
            use_fast = 'mpt' in model_type # Example heuristic
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True, **kwargs)

    image_processor = None

    # Handle vision tower loading consistently for LLaVA models
    if 'llava' in model_name.lower() or getattr(model.config, 'mm_vision_tower', None):
        print("Processing LLaVA/Vision related components...")
        # Add tokens if necessary (check model config)
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            added_tokens = tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if added_tokens > 0: print(f"Added {added_tokens} special image patch token.")
        if mm_use_im_start_end:
            added_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            if added_tokens > 0: print(f"Added {added_tokens} special image start/end tokens.")
        # Important: Resize token embeddings after adding tokens
        model.resize_token_embeddings(len(tokenizer))

        # Get vision tower instance from the loaded model
        vision_tower = model.get_vision_tower()

        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_config_path_or_hf_id = getattr(model.config, "mm_vision_tower", "openai/clip-vit-large-patch14-336")
                kaggle_clip_path = "/kaggle/input/openai-clip-vit-large-patch14-336" # Target Kaggle path

                final_vision_path = vision_config_path_or_hf_id # Default to config value
                # Check if Kaggle path exists and seems valid (e.g., contains config.json)
                if os.path.exists(kaggle_clip_path) and os.path.isfile(os.path.join(kaggle_clip_path, 'config.json')):
                     print(f"Attempting to use CLIP model from Kaggle Input: {kaggle_clip_path}")
                     final_vision_path = kaggle_clip_path
                     # Update the vision_tower's internal path reference before loading
                     vision_tower.vision_tower_name = final_vision_path
                else:
                    print(f"Kaggle Input path for CLIP ('{kaggle_clip_path}') not found or invalid, using value from config: '{vision_config_path_or_hf_id}'")
                    # Ensure the name reflects the actual source being used
                    vision_tower.vision_tower_name = vision_config_path_or_hf_id

                print(f"Loading vision tower weights from: {vision_tower.vision_tower_name}")
                # Load vision tower weights. Let it infer device map from main model if possible (pass None)
                # Pass target dtype from main kwargs
                vision_tower.load_model(device_map=None, torch_dtype=kwargs.get('torch_dtype', torch.float16))
                print("Vision tower weights loaded.")
            image_processor = vision_tower.image_processor
        else:
            warnings.warn("Vision tower is None, image processing won't be available.")

    # Determine context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        # Provide a reasonable default or check specific model type defaults
        context_len = getattr(model.config, "max_position_embeddings", 2048) # Check max_position_embeddings too
        print(f"Warning: max_sequence_length not found in config, using max_position_embeddings or default: {context_len}")

    return tokenizer, model, image_processor, context_len