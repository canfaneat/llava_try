import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loading tokenizer, model, image_processor from: {model_path}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    print(f"Image processor loaded: {image_processor is not None}") # Check if processor loaded

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    print(f"Starting evaluation loop for {len(questions)} questions...")
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        # DEBUG: Added limit for quick testing - REMOVE for full run
        if index >= 5:
           print("DEBUG: Reached sample limit for testing.")
           break

        print(f"--- Processing index: {row['index']} ---") # Print separator and index

        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']

            # --- Image Loading Debug ---
            image = None # Initialize to None
            image_base64 = None
            try:
                image_base64 = row['image']
                if pd.isna(image_base64): # Check if NaN before loading
                    print("DEBUG: Image base64 data is NaN/missing in input TSV!")
                else:
                    # print(f"DEBUG: Attempting to load image from base64 (first 30 chars): {str(image_base64)[:30]}...") # Optionally print snippet
                    image = load_image_from_base64(image_base64)
                    print(f"DEBUG: Image loaded via base64: {image is not None}") # Confirm image loaded
                    if image:
                        print(f"DEBUG: Image size: {image.size}, mode: {image.mode}") # Print size and mode
            except KeyError:
                print("DEBUG: 'image' column not found in input TSV!")
            except Exception as e:
                print(f"DEBUG: Error loading image from base64: {e}")

            if image is None:
                print("!!! CRITICAL: Image object is None after loading attempt !!!")
            # --- End Image Loading Debug ---

            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question

            # --- Prompt and Token Construction Debug ---
            image_token_inserted = False
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                image_token_inserted = True
            elif DEFAULT_IMAGE_TOKEN in qs: # Check if already present
                 print("DEBUG: DEFAULT_IMAGE_TOKEN found in text prompt, maybe not needed to add again?")
                 image_token_inserted = True
            else:
                # Only add image token if an image is supposed to be present
                if image is not None:
                     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                     image_token_inserted = True
                else:
                     print("DEBUG: No image loaded, skipping adding DEFAULT_IMAGE_TOKEN.")

            print(f"DEBUG: Image token inserted into prompt: {image_token_inserted}")
            print(f"DEBUG: Input Text (qs, first 100 chars): {qs[:100]}")

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # print(f"DEBUG: Full Prompt: {prompt}") # Can uncomment for very verbose debugging

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            print(f"DEBUG: Input IDs shape: {input_ids.shape}")
            # --- End Prompt Debug ---


            # --- Image Processing and Model Input Debug ---
            image_tensor = None # Initialize
            image_tensor_input = None # Initialize tensor for model
            if image is not None and image_processor is not None:
                 try:
                     print(f"DEBUG: Processing image with processor: {type(image_processor)}")
                     image_tensor = process_images([image], image_processor, model.config)[0]
                     print(f"DEBUG: Image tensor processed: {image_tensor is not None}") # Confirm tensor generated
                     if image_tensor is not None:
                         print(f"DEBUG: Raw Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
                         # Prepare tensor for model (unsqueezing, type conversion, device placement)
                         image_tensor_input = image_tensor.unsqueeze(0).half().cuda()
                         print(f"DEBUG: Image tensor passed to model: shape={image_tensor_input.shape}, dtype={image_tensor_input.dtype}, device={image_tensor_input.device}")
                 except Exception as e:
                     print(f"DEBUG: Error processing image: {e}")
                     image_tensor_input = None # Ensure it's None if processing fails
            elif image is None:
                 print("DEBUG: Skipping image tensor creation because Image object is None.")
                 image_tensor_input = None
            else: # image_processor is None
                 print("DEBUG: Skipping image tensor creation because image_processor is None.")
                 image_tensor_input = None

            # Double-check before generate call
            if image_tensor_input is None:
                 print("!!! WARNING: image_tensor_input is None before calling model.generate. Model will likely run text-only. !!!")
            # --- End Image Processing Debug ---


            # --- Model Generation Debug ---
            outputs = "[GENERATION SKIPPED/ERROR]" # Default in case of error
            try:
                with torch.inference_mode():
                    print("DEBUG: Calling model.generate...")
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor_input, # Pass the prepared tensor (or None)
                        image_sizes=[image.size] if image is not None else None, # Pass size only if image exists
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                print("DEBUG: model.generate finished.")
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(f"DEBUG: Raw Model Output (text): {outputs}") # Print raw output
            except Exception as e:
                print(f"!!! ERROR during model.generate: {e} !!!")
            # --- End Model Generation Debug ---

            ans_id = shortuuid.uuid()
            # --- Output Saving Debug ---
            print(f"DEBUG: Saving result for question_id: {idx}")
            ans_data = {"question_id": idx,
                        "round_id": round_idx,
                        "prompt": cur_prompt, # Original question prompt without image tokens
                        "text": outputs, # Raw text output from model
                        "options": options,
                        "option_char": cur_option_char,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {}}
            # print(f"DEBUG: Data being saved: {ans_data}") # Very verbose
            ans_file.write(json.dumps(ans_data) + "\n")
            # --- End Output Saving Debug ---
            ans_file.flush()

            # rotate options
            if num_rounds > 1: # Only rotate if doing multiple rounds per question
                 options = options[1:] + options[:1]
                 cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()
    print("Evaluation loop finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="") # Original code had this, maybe relevant if not using base64?
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2) # Note: Was 0 in your script, using 0.2 from original code default
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    # Override temperature if passed via mmbench.sh
    import sys
    if '--temperature' in sys.argv:
         temp_index = sys.argv.index('--temperature') + 1
         if temp_index < len(sys.argv):
              try:
                   args.temperature = float(sys.argv[temp_index])
                   print(f"DEBUG: Overriding temperature from command line: {args.temperature}")
              except ValueError:
                   pass # Keep default if conversion fails


    eval_model(args)
