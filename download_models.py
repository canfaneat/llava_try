from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CLIPModel, CLIPProcessor

# 下载 Vicuna-7B
vicuna = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    cache_dir="./models/vicuna-7b-v1.5"
)
vicuna_tokenizer = AutoTokenizer.from_pretrained(
    "lmsys/vicuna-7b-v1.5",
    cache_dir="./models/vicuna-7b-v1.5"
)

# 下载 CLIP-ViT
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir="./models/clip-vit-large-patch14"
)
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14",
    cache_dir="./models/clip-vit-large-patch14"
)

print("✅ 模型下载完成，存放在 ./models/ 路径下。")
