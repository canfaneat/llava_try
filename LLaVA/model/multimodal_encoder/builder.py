# 使用本地CLIP模型路径 - 注释掉硬编码路径，依赖更上层的逻辑
# if "openai/clip-vit-large-patch14-336" in vision_tower:
#     local_path = "D:/pcdpm_project/LLaVA_fork/models/clip-vit-large-patch14-336"
#     if os.path.exists(local_path):
#         print(f"使用本地CLIP-336模型: {local_path}")
#         vision_tower = local_path 