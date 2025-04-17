import torch
import torch.nn as nn
import os

# 添加这些代码来强制使用离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # 添加本地处理器配置文件的逻辑
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        except Exception as e:
            print(f"无法加载图像处理器，尝试创建默认处理器: {e}")
            # 创建默认的CLIP处理器配置
            processor_config = {
                "crop_size": 336,
                "do_center_crop": True,
                "do_normalize": True,
                "do_resize": True,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "resample": 3,
                "size": 336
            }
            from transformers.image_processing_utils import BaseImageProcessor
            self.image_processor = BaseImageProcessor(**processor_config)
        
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map, local_files_only=True)
        except Exception as e:
            print(f"加载视觉模型失败: {e}")
            raise
            
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args=None, **kwargs):
        super().__init__()
    
        self.is_loaded = False
    
        self.vision_tower_name = vision_tower
    
        # 修改这里，使用本地已有的CLIP模型
        if "openai/clip-vit-large-patch14-336" in vision_tower:
            local_path = "D:/pcdpm_project/LLaVA_fork/models/clip-vit-large-patch14-336"
            if os.path.exists(local_path):
                print(f"使用本地CLIP模型: {local_path}")
                self.vision_tower_name = local_path
        # 保留原来的代码以兼容其他CLIP模型
        elif "openai/clip-vit-large-patch14" in vision_tower:
            local_path = "D:/pcdpm_project/LLaVA_fork/models/clip-vit-large-patch14"
            if os.path.exists(local_path):
                print(f"使用本地CLIP模型: {local_path}")
                self.vision_tower_name = local_path
        self.select_layer = args.mm_vision_select_layer if hasattr(args, 'mm_vision_select_layer') else -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
    
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
    
        # 添加本地处理器配置文件的逻辑
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name, local_files_only=True)
        except Exception as e:
            print(f"无法加载图像处理器，尝试创建默认处理器: {e}")
            # 创建默认的CLIP处理器配置
            processor_config = {
                "crop_size": 336,
                "do_center_crop": True,
                "do_normalize": True,
                "do_resize": True,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "resample": 3,
                "size": 336
            }
            from transformers.image_processing_utils import BaseImageProcessor
            self.image_processor = BaseImageProcessor(**processor_config)
        
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map, local_files_only=True)
        except Exception as e:
            print(f"加载视觉模型失败: {e}")
            raise
            
        self.vision_tower.requires_grad_(False)
        
        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
