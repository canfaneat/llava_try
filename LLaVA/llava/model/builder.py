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
# Ensure necessary LLaVA classes are imported correctly
from llava.model import LlavaLlamaForCausalLM, LlavaMistralForCausalLM, LlavaMptForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# Explicitly import CLIPVisionTower
try:
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
except ImportError as e:
    warnings.warn(f"Could not import CLIPVisionTower, vision capabilities might be affected: {e}")
    CLIPVisionTower = None # Define as None to avoid subsequent NameError


# *** 修改: 默认加载方式改为 8-bit ***
def load_pretrained_model(model_path, model_base, model_name, load_8bit=True, load_4bit=False, device_map="cuda:0", device="cuda", use_flash_attn=False, **kwargs):

    # Handle device mapping for CUDA vs non-CUDA
    if device != "cuda":
        kwargs['device_map'] = {"": device}
        print(f"Warning: Loading model on device '{device}'. CUDA acceleration will be unavailable.")
    else:
        # Use the intended device_map (should be "auto" for T4x2)
        kwargs['device_map'] = device_map
        print(f"Using device_map: {kwargs['device_map']}")

    # Handle quantization flags (defaulting to 8-bit)
    if load_8bit:
        print("Loading model in 8bit")
        kwargs['load_in_8bit'] = True
        kwargs.pop('torch_dtype', None) # 8bit incompatible with torch_dtype
    elif load_4bit:
        print("Loading model in 4bit")
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        kwargs.pop('torch_dtype', None) # 4bit incompatible with torch_dtype
    else:
        # Fallback to float16 if neither 8bit nor 4bit is explicitly True
        print("Loading model in float16")
        kwargs['torch_dtype'] = torch.float16

    # Flash attention (Optional - likely incompatible with 8/4 bit)
    if use_flash_attn:
        if not load_8bit and not load_4bit:
            try:
                import flash_attn
                print("Setting attn_implementation to flash_attention_2")
                kwargs['attn_implementation'] = 'flash_attention_2'
            except ImportError:
                print("FlashAttention specified but not installed, using default attention.")
        else:
            print("FlashAttention specified but loading in 8/4 bit, ignoring.")

    # Helper to check if path is local or HF identifier
    def is_local(path):
        # Check for common path separators or if the path actually exists locally
        return path is not None and ('/' in path or '\' in path or os.path.exists(path))

    model_path_is_local = is_local(model_path)
    model_base_is_local = is_local(model_base)

    # Corrected print statements with proper f-string syntax
    print(f"DEBUG: model_path='{model_path}' (local: {model_path_is_local})")
    print(f"DEBUG: model_base='{model_base}' (local: {model_base_is_local})")

    if 'llava' in model_name.lower():
        # Load LLaVA model variations
        if 'lora' in model_name.lower() and model_base is not None:
            # --- LoRA LLaVA 模型加载 --- 
            print("Loading LLaVA LoRA weights")
            from llava.model.language_model.llava_llama import LlavaConfig
            # Allow downloading tokenizer/config from HF Hub if model_base is ID
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=model_base_is_local)
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True, local_files_only=model_base_is_local)
            # LoRA config usually local
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path, local_files_only=model_path_is_local)
            print("Merging LoRA config fields into base config...")
            for key, value in lora_cfg_pretrained.to_dict().items():
                 setattr(base_config, key, value)

            print('Loading LLaVA from base model...')
            # Allow downloading base model from HF Hub if model_base is ID
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=base_config, local_files_only=model_base_is_local, **kwargs
            )
            # Handle token embeddings resizing
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            # Load non-LoRA trainable weights (usually local)
            print('Loading additional LLaVA weights (non-lora trainables)...')
            non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_path):
                non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)
            else: warnings.warn(f"non_lora_trainables.bin not found at {non_lora_path}.")
            # Load LoRA adapter weights (usually local)
            from peft import PeftModel
            print('Loading LoRA adapter weights...')
            model = PeftModel.from_pretrained(model, model_path, local_files_only=model_path_is_local)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('LoRA Model is loaded and merged...')
            if not kwargs.get('load_in_8bit') and not kwargs.get('load_in_4bit'):
                model.to(kwargs.get('torch_dtype', torch.float16))

        elif model_base is not None:
            # --- 非 LoRA LLaVA 模型加载 --- 
            print(f'Loading LLM base from: {model_base} (local only: {model_base_is_local})')
            # Allow downloading tokenizer/config from HF Hub if model_base is ID
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=model_base_is_local)
            base_config = AutoConfig.from_pretrained(model_base, trust_remote_code=True, local_files_only=model_base_is_local)

            print(f'Loading LLaVA delta config from: {model_path} (local only: {model_path_is_local})')
            # LLaVA delta config/weights usually loaded from model_path (can be HF ID or local)
            llava_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=model_path_is_local)

            # 确定并强制设置最终的视觉塔路径
            print("准备合并配置，并确定视觉塔路径...")
            llava_specific_keys_to_merge = [
                "mm_hidden_size", "mm_projector_type", "mm_vision_select_layer",
                "mm_vision_select_feature", "mm_use_im_start_end",
                "mm_use_im_patch_token", "image_aspect_ratio",
            ]
            # Vision tower path from llava_config, assume it can be HF ID or local
            final_vision_tower_path = getattr(llava_config, "mm_vision_tower", "openai/clip-vit-large-patch14-336")
            vision_tower_is_local = is_local(final_vision_tower_path)
            print(f"  Detected vision tower path from config: '{final_vision_tower_path}' (local: {vision_tower_is_local})")
            # We don't override with Kaggle path here anymore, rely on the config value

            # 更新基础配置
            for key in llava_specific_keys_to_merge:
                if hasattr(llava_config, key):
                    value = getattr(llava_config, key)
                    print(f"  合并: base_config.{key} = {value}")
                    setattr(base_config, key, value)
            print(f"  最终设置: base_config.mm_vision_tower = {final_vision_tower_path}")
            setattr(base_config, "mm_vision_tower", final_vision_tower_path)

            # 确定 LLaVA 类
            model_type = getattr(base_config, "model_type", "")
            if 'mpt' in model_type: llava_class = LlavaMptForCausalLM
            elif 'mistral' in model_type: llava_class = LlavaMistralForCausalLM
            else: llava_class = LlavaLlamaForCausalLM
            print(f"确定模型类: {llava_class.__name__} (基于基础模型类型 '{model_type}')")

            print(f"加载模型权重自: {model_base} (local only: {model_base_is_local})，使用更新后的基础配置...")
            # Allow downloading base model from HF Hub if model_base is ID
            model = llava_class.from_pretrained(
                model_base,
                config=base_config,
                low_cpu_mem_usage=True,
                local_files_only=model_base_is_local,
                **kwargs # 传递 device_map, load_in_8bit 等
            )

            # 加载 mm_projector 权重 (usually from model_path, can be HF ID or local)
            print(f"加载 mm_projector 权重 from {model_path} (local only: {model_path_is_local})...")
            # Use AutoModel to load projector weights, allows loading from HF Hub or local path
            try:
                 # We need to load the state dict, not the whole model here
                 # Temporarily load the full model specified by model_path to get the projector state_dict
                 # This assumes model_path points to a full LLaVA model containing the projector
                 print(f"  (临时加载 {model_path} 以获取投影层状态字典...)")
                 temp_full_model = llava_class.from_pretrained(model_path, local_files_only=model_path_is_local)
                 mm_projector_weights = {k: v for k, v in temp_full_model.state_dict().items() if 'mm_projector' in k}
                 del temp_full_model # Free memory
                 if mm_projector_weights:
                     target_dtype = torch.float16 if kwargs.get('load_in_8bit') or kwargs.get('load_in_4bit') else kwargs.get('torch_dtype', torch.float16)
                     print(f"  转换投影层权重到 {target_dtype}")
                     mm_projector_weights = {k: v.to(target_dtype) for k, v in mm_projector_weights.items()}
                     load_info = model.load_state_dict(mm_projector_weights, strict=False)
                     print(f"  MM projector 权重加载完成. Info: {load_info}")
                 else:
                     warnings.warn(f"在 {model_path} 中未找到 'mm_projector' 权重.")
            except Exception as e:
                 warnings.warn(f"从 {model_path} 加载 MM projector 权重时出错: {e}. 可能需要手动加载或检查路径。")

        else:
            # --- 加载完整 LLaVA 模型 (无 base) --- (Assume model_path is HF ID or local)
            print(f"加载完整 LLaVA 模型自: {model_path} (local only: {model_path_is_local})")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=model_path_is_local)
            model_type = getattr(config, "model_type", "")
            if 'mpt' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=model_path_is_local)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=model_path_is_local, **kwargs)
            elif 'mistral' in model_type:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=model_path_is_local)
                model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=model_path_is_local, **kwargs)
            else: # Assume Llama
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=model_path_is_local)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=model_path_is_local, **kwargs)

    # --- 非 LLaVA 标准模型加载 --- 
    else:
        print(f"加载标准非 LLaVA 模型: {model_name} (from {model_path if model_base is None else model_base})")
        if model_base is not None:
            # LoRA for standard models (model_base=HF ID/local, model_path=local LoRA weights)
            from peft import PeftModel
            print("Loading PEFT model")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, local_files_only=model_base_is_local)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, local_files_only=model_base_is_local, **kwargs)
            print(f"Loading LoRA weights from {model_path} (local only: {model_path_is_local})")
            model = PeftModel.from_pretrained(model, model_path, local_files_only=model_path_is_local)
            print(f"Merging weights")
            model = model.merge_and_unload()
            if not kwargs.get('load_in_8bit') and not kwargs.get('load_in_4bit'):
                 print('Convert merged PEFT model to target dtype...')
                 model.to(kwargs.get('torch_dtype', torch.float16))
        else:
            # Standard model from model_path (HF ID or local)
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=model_path_is_local)
            model_type = getattr(config, "model_type", "")
            use_fast = 'mpt' in model_type
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, local_files_only=model_path_is_local)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=model_path_is_local, **kwargs)

    # --- 视觉塔和图像处理器加载 --- 
    image_processor = None
    if 'llava' in model_name.lower() or getattr(model.config, 'mm_vision_tower', None):
        print("处理 LLaVA 视觉组件...")
        # 添加特殊 token
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        added_tokens = 0
        if mm_use_im_patch_token:
            if DEFAULT_IMAGE_PATCH_TOKEN not in tokenizer.get_vocab():
                 added_tokens += tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            else:
                 print(f"DEBUG: Token {DEFAULT_IMAGE_PATCH_TOKEN} already in vocab.")
        if mm_use_im_start_end:
            tokens_to_add = []
            if DEFAULT_IM_START_TOKEN not in tokenizer.get_vocab(): tokens_to_add.append(DEFAULT_IM_START_TOKEN)
            if DEFAULT_IM_END_TOKEN not in tokenizer.get_vocab(): tokens_to_add.append(DEFAULT_IM_END_TOKEN)
            if tokens_to_add:
                 added_tokens += tokenizer.add_tokens(tokens_to_add, special_tokens=True)
            else:
                 print(f"DEBUG: Tokens {DEFAULT_IM_START_TOKEN}, {DEFAULT_IM_END_TOKEN} already in vocab.")
        if added_tokens > 0:
            print(f"添加了 {added_tokens} 个特殊 token。调整嵌入层大小...")
            model.resize_token_embeddings(len(tokenizer))

        # 获取视觉塔实例
        vision_tower = model.get_vision_tower()

        if vision_tower is not None and CLIPVisionTower is not None:
            if not vision_tower.is_loaded:
                vision_tower_path = getattr(model.config, "mm_vision_tower") # Should have correct HF ID or path
                vision_tower_is_local = is_local(vision_tower_path)
                print(f"尝试从以下路径加载视觉塔: {vision_tower_path} (local only: {vision_tower_is_local})")
                try:
                    # 加载视觉塔, 允许从 HF 下载, 使用 float16
                    vision_tower.load_model(
                        device_map=None, # 跟随主模型映射
                        torch_dtype=torch.float16,
                        local_files_only=vision_tower_is_local
                    )
                    print("视觉塔加载成功。")
                    image_processor = vision_tower.image_processor
                except Exception as e:
                     print(f"加载视觉塔时出错: {e}")
                     warnings.warn("视觉塔加载失败，图像处理将不可用。")
            else:
                print("视觉塔先前已加载。")
                image_processor = vision_tower.image_processor
        elif CLIPVisionTower is None:
             warnings.warn("CLIPVisionTower 未导入，无法加载视觉塔。")
        else: # vision_tower is None
            warnings.warn("从模型获取的 vision_tower 为 None，图像处理将不可用。")

    # 确定上下文长度
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = getattr(model.config, "max_position_embeddings", 2048)
        print(f"未在配置中找到 max_sequence_length，使用 max_position_embeddings 或默认值: {context_len}")

    print("模型加载流程结束。")
    return tokenizer, model, image_processor, context_len