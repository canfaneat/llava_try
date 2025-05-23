import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    # 添加调试信息
    print(f"模型加载成功: {model_name}")
    print(f"设备: {args.device}")
    print(f"图像处理器: {type(image_processor).__name__}")

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    # 调整图像大小，尝试使用标准尺寸
    image = image.resize((224, 224))
    image_size = image.size
    print(f"图像加载成功，尺寸: {image_size}")
    
    # 添加异常处理
    try:
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        # 检查图像张量是否包含无效值
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            print("警告: 图像张量包含NaN或Inf值，尝试修复...")
            # 替换无效值
            image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"处理图像时出错: {e}")
        # 尝试使用更简单的方法处理图像
        print("尝试使用备用图像处理方法...")
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
            ])
            image_tensor = transform(image).unsqueeze(0).to(model.device, dtype=torch.float16)
        except Exception as e2:
            print(f"备用图像处理也失败: {e2}")
            raise

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        try:
            with torch.inference_mode():
                # 尝试使用最安全的生成参数
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,  # 使用贪婪解码
                    num_beams=1,      # 禁用束搜索
                    max_new_tokens=min(args.max_new_tokens, 100),
                    streamer=streamer,
                    use_cache=True)
        except RuntimeError as e:
            print(f"\n生成时出错: {e}")
            print("尝试不使用图像直接生成文本...")
            
            # 最后的尝试：不使用图像，只生成文本回复
            try:
                with torch.inference_mode():
                    # 创建一个简单的提示
                    simple_prompt = f"用户请求描述一张图片，但由于技术原因无法处理图像。请回复：'抱歉，我无法处理这张图片。'"
                    simple_input_ids = tokenizer(simple_prompt, return_tensors='pt').input_ids.to(model.device)
                    output_ids = model.generate(
                        simple_input_ids,
                        max_new_tokens=50,
                        do_sample=False,
                        streamer=streamer)
            except Exception as e2:
                print(f"最终尝试也失败: {e2}")
                outputs = "抱歉，无法生成回复。"
                conv.messages[-1][-1] = outputs
                continue

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
