from llava.train.train import train

if __name__ == "__main__":
    train()
     #train(attn_implementation="flash_attention_2")
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
#
# from llava.train.train import train
#
# if __name__ == "__main__":
#     train(
#         model_name_or_path="./models/vicuna-7b-v1.5",
#         vision_tower="./models/clip-vit-large-patch14",
#         mm_projector_type="mlp",
#         freeze_backbone=True,
#         output_dir="./output/baseline_coco_mlp",
#         data_path="./data/coco_llava_format.json",
#         image_folder="./data/train2014",
#         num_train_epochs=3,
#         learning_rate=1e-4,
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=8,
#         bf16=True,
#         save_strategy="epoch",
#         evaluation_strategy="epoch",
#         save_total_limit=2,
#         remove_unused_columns=False
#     )

