import os
import json
import argparse
import pandas as pd
import re # Import regular expression module

def parse_pred_ans(pred_text):
    """
    Extracts the first single uppercase letter (A, B, C, D) found in the prediction text.
    Handles simple cases like "A", "A.", "Answer: A", etc.
    Returns the letter or None if no valid option is found.
    """
    pred_text = pred_text.strip()
    # Try to find a single capital letter A, B, C, or D, possibly surrounded by noise
    # Example patterns: "A.", "A", "(A)", "Answer: A"
    match = re.search(r'^[(\s]*([A-D])[.)\s]*', pred_text, re.IGNORECASE)
    if match:
        return match.group(1).upper() # Return the extracted letter in uppercase

    # Fallback: Check if the text itself is just one of the options
    if pred_text in ['A', 'B', 'C', 'D']:
         return pred_text

    # More robust fallback: Check if the first non-whitespace character is A, B, C, or D
    if pred_text and pred_text[0] in ['A', 'B', 'C', 'D']:
        return pred_text[0]

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
