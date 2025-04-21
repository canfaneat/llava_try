import os
import json
import argparse
import pandas as pd
import re # Import regular expression module

def parse_pred_ans(pred_text):
    """
    Extracts the first single uppercase letter (A, B, C, D) found in the prediction text.
    Handles simple cases like "A", "A.", "Answer: A", etc.
    Improved to also find the first standalone letter if not at the beginning.
    Returns the letter or None if no valid option is found.
    """
    if not isinstance(pred_text, str):
        return None # Handle cases where input is not a string
    pred_text = pred_text.strip()

    # 1. Try common patterns at the beginning (most reliable)
    match = re.search(r'^[(\s]*([A-D])[.)\s]*', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Fallback: Check if the text itself is just one of the options (case insensitive)
    if pred_text.upper() in ['A', 'B', 'C', 'D']:
         return pred_text.upper()

    # 3. Fallback: Find the first occurrence of A/B/C/D as a standalone letter
    #    (surrounded by non-alphanumeric chars or start/end of string)
    #    This helps find "... answer is A ..." or "... option B." but avoids matching words like "Apple"
    match = re.search(r'(?<![a-zA-Z0-9])([A-D])(?![a-zA-Z0-9])', pred_text)
    if match:
        return match.group(1).upper() # Return the first such occurrence

    return None # Return None if no valid option letter is extracted


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred_line in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred_line)
        # Extract the single letter answer from the raw text
        parsed_ans = parse_pred_ans(pred['text'])
        # Assign the parsed answer (or None if parsing failed) to the prediction column
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = parsed_ans

    cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
