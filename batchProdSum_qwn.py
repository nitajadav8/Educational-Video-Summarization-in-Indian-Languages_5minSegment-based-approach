#################################################################################################
# batch processing script for generating multilingual summaries 
# - Reads multiple CSV files from input directory
#################################################################################################

from unsloth import FastLanguageModel
import pandas as pd
import torch
import os
import glob
import argparse

print("CUDA Available:", torch.cuda.is_available())

# -------------------------------
# ARGUMENTS (model + directory)
# -------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, required=True,
                    help="HF model name (e.g., unsloth/Qwen2.5-3B-Instruct)")
parser.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing input CSV files")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory to save outputs")

args = parser.parse_args()

model_name = args.model_name
input_dir = args.input_dir
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
max_seq_length = 32768
lora_rank = 16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    #max_lora_rank = lora_rank
)

FastLanguageModel.for_inference(model)

# -------------------------------
# SUMMARY FUNCTION
# -------------------------------
def get_summary(text, language):
    if isinstance(text, list):
        text = text[0]

    prompt = f"""Summarize the following educational video transcript content with maximum 20% of content length.
Generate a human-like abstractive summary in {language}. Important: 
Remove ASR noise or transcription errors. Do not make any translation errors. Do not mix languages. Keep it simple and clear.

Content:
{text}

Summary:
"""

    messages = [
        {"role": "system", "content": "You are an expert multilingual educational summarizer."},
        {"role": "user", "content": prompt},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        use_cache=False,
        repetition_penalty=1.15,    
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()

# -------------------------------
# PROCESS ALL FILES
# -------------------------------

def process_files(input_dir, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    for input_path in csv_files:
        print(f"\nProcessing: {input_path}")

        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, f"summary_{filename}")

        df = pd.read_csv(input_path)

        # ---- Load already processed IDs ----
        processed_ids = set()
        if os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                if "uid" in existing_df.columns:
                    processed_ids = set(existing_df["uid"].astype(str))
                print(f"Resuming: {len(processed_ids)} rows already processed")
            except Exception as e:
                print(f"Warning: Could not read existing output ({e})")

        write_header = not os.path.exists(output_path)

        buffer = []
        flush_every = 10  # tune

        # ---- Process rows ----
        for idx, row in df.iterrows():

            name = str(row.get("merged_name", ""))
            text = row.get("merged_segment_text", "")

            # ---- Unique ID ----
            uid = f"{name}_{idx}"

            if uid in processed_ids:
                continue

            if not isinstance(text, str) or not text.strip():
                continue

            try:
                result = {
                    "uid": uid,
                    "m_name": name,
                    "ai_en_sum": get_summary(text, "English"),
                    "ai_h_sum": get_summary(text, "Hindi"),
                    "ai_g_sum": get_summary(text, "Gujarati"),
                    "ai_te_sum": get_summary(text, "Telugu"),
                    "ai_model": model_name
                }

            except Exception as e:
                print(f"Error at row {idx}: {e}")

                result = {
                    "uid": uid,
                    "m_name": name,
                    "ai_en_sum": "",
                    "ai_h_sum": "",
                    "ai_g_sum": "",
                    "ai_te_sum": "",
                    "ai_model": model_name
                }

            buffer.append(result)
            processed_ids.add(uid)

            # ---- Flush periodically ----
            if len(buffer) >= flush_every:
                pd.DataFrame(buffer).to_csv(
                    output_path,
                    mode='a',
                    header=write_header,
                    index=False,
                    encoding="utf-8"
                )
                write_header = False
                buffer.clear()

        # ---- Final flush ----
        if buffer:
            pd.DataFrame(buffer).to_csv(
                output_path,
                mode='a',
                header=write_header,
                index=False,
                encoding="utf-8"
            )

        print(f"Saved → {output_path}")

def main():
    process_files(input_dir, output_dir, model_name)
    # csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # for input_path in csv_files:

    #     print(f"\nProcessing: {input_path}")

    #     df = pd.read_csv(input_path)
    #     output = []

    #     for _, row in df.iterrows():
    #         name = row["merged_name"]
    #         text = row["merged_segment_text"]

    #         if not isinstance(text, str) or not text.strip():
    #             continue

    #         try:
    #             output.append({
    #                 "m_name": name,
    #                 "ai_en_sum": get_summary(text, "English"),
    #                 "ai_h_sum": get_summary(text, "Hindi"),
    #                 "ai_g_sum": get_summary(text, "Gujarati"),
    #                 "ai_te_sum": get_summary(text, "Telugu"),
    #                 "ai_model": model_name
    #             })

    #         except Exception as e:
    #             output.append({
    #                 "m_name": name,
    #                 "ai_en_sum": "",
    #                 "ai_h_sum": "",
    #                 "ai_g_sum": "",
    #                 "ai_te_sum": "",
    #                 "ai_model": model_name
    #             })

    #     # Save output per file
    #     filename = os.path.basename(input_path)
    #     output_path = os.path.join(output_dir, f"summary_{filename}")

    #     out_df = pd.DataFrame(output)
    #     out_df.to_csv(output_path, index=False, encoding="utf-8")

    #     print(f"Saved → {output_path}")

if __name__ == "__main__":
    main()