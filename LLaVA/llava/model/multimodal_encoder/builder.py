import os
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', None)
    
    # 添加调试信息
    print(f"构建视觉塔，配置: {vision_tower}")
    
    if vision_tower is None:
        print("警告: mm_vision_tower 为 None")
        return None
    
    # 使用本地CLIP模型路径
    if "openai/clip-vit-large-patch14-336" in vision_tower:
        local_path = "D:/pcdpm_project/LLaVA_fork/models/clip-vit-large-patch14-336"
        if os.path.exists(local_path):
            print(f"使用本地CLIP-336模型: {local_path}")
            vision_tower = local_path
    
    # 确保select_layer存在
    if not hasattr(vision_tower_cfg, 'mm_vision_select_layer'):
        vision_tower_cfg.mm_vision_select_layer = -2
        print(f"设置默认mm_vision_select_layer: {vision_tower_cfg.mm_vision_select_layer}")
    
    # 创建视觉塔实例
    try:
        vision_tower = CLIPVisionTower(vision_tower, vision_tower_cfg)
        print("视觉塔创建成功")
        return vision_tower
    except Exception as e:
        print(f"创建视觉塔时出错: {e}")
        return None
