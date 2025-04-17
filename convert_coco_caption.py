import json
import os

# 输入原始 Karpathy 格式路径
input_path = r"D:\caption_datasets\dataset_coco.json"  # 使用原始字符串避免路径问题

# 输出 LLaVA 格式路径
output_path = "data/coco_llava_format.json"

# 加载原始 COCO Caption 数据
with open(input_path, 'r') as f:
    raw_data = json.load(f)

converted = []
for entry in raw_data['images']:
    if entry['split'] != 'train':
        continue  # 只处理训练集
    image_id = entry['filename']  # 例如 COCO_train2014_000000581929.jpg
    for sentence in entry['sentences']:
        caption = sentence['raw']
        # 修改为 LLaVA 格式，添加 conversations 字段
        converted.append({
            'id': f"{image_id}_{sentence.get('sentid', 0)}",
            'image': image_id,
            'conversations': [
                {"from": "human", "value": "<image>\n描述这张图片"},
                {"from": "assistant", "value": caption}
            ]
        })

# 保存转换后文件
os.makedirs("data", exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"✅ 成功转换 {len(converted)} 条数据，输出文件：{output_path}")
