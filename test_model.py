import os
os.environ["NPY_DISABLE_CPU_FEATURES"] = "1"  # 禁用numpy CPU扩展
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

# 验证CUDA状态
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(r"D:\pcdpm_project\LLaVA_fork\models\vicuna-7b-v1.5")

# 创建输入并移动到GPU
input_text = "Hello, my name is"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 加载模型到GPU（添加离线目录和4位量化）
model = LlamaForCausalLM.from_pretrained(
    r"D:\pcdpm_project\LLaVA_fork\models\vicuna-7b-v1.5",
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    offload_folder="d:\\pcdpm_project\\temp_offload",  # 新增离线目录
    load_in_4bit=True  # 启用4位量化
).eval()

# 打印模型设备信息
print(f"模型设备: {next(model.parameters()).device}")

# 生成输出
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 解码并打印结果
print("\n生成结果:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
