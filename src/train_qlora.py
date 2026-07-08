"""QLoRA supervised fine-tuning of Llama-3.1-8B-Instruct on data/train.jsonl.

Run on the IEOR server (CUDA GPU, A5000 24GB). bitsandbytes 4-bit needs CUDA — this will
NOT run on the Mac (no CUDA). See the handoff notes for the exact server commands.

Pipeline:
  1. Load base model in 4-bit (NF4) — this is the "Q" in QLoRA.
  2. Attach LoRA adapters (PEFT) to q_proj/v_proj — only these ~0.1% of params train.
  3. Format each chat example (incl. tool-call examples) via the tokenizer's chat template.
  4. Train with TRL SFTTrainer, checkpoint every 50 steps.
  5. Save the adapter; optionally merge into the base and push the merged model to HF Hub.

Base model is gated: request access at huggingface.co/meta-llama/Llama-3.1-8B-Instruct and
run `huggingface-cli login` with an HF_TOKEN that has been granted access, BEFORE running.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "domaingpt-qlora"

def load_examples() -> list[dict]:
    with open(TRAIN_PATH) as f:
        return [json.loads(line) for line in f]


def format_example(example: dict, tokenizer) -> str:
    """Render one chat example to a single training string via the chat template.
    All assistant targets are plain text content (tool-selection targets are the
    Llama-3.1 tool-call JSON as text), so a straight chat-template render is enough.

    Some template variants append a trailing empty assistant generation prompt even with
    add_generation_prompt=False; trim anything after the final <|eot_id|> so the SFT target
    ends cleanly on the assistant's turn."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )
    eot = "<|eot_id|>"
    if eot in text:
        text = text[: text.rfind(eot) + len(eot)]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None, help="e.g. prashantgautam8077/domaingpt-v1")
    parser.add_argument("--merge", action="store_true", help="merge adapter into base after training")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. QLoRA 4-bit requires CUDA — run this on the IEOR server.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 1. 4-bit quantized base model (the "Q" in QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # --- 2. LoRA adapters (only these train) ---
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 3. Dataset ---
    raw = load_examples()
    texts = [format_example(ex, tokenizer) for ex in raw]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Loaded {len(dataset)} training examples")
    print("--- sample formatted example ---")
    print(texts[0][:800])
    print("--------------------------------")

    # --- 4. Train ---
    # TRL's SFTTrainer/SFTConfig API has churned a lot across versions:
    #   - max_seq_length was renamed to max_length in SFTConfig
    #   - dataset_text_field moved from SFTTrainer kwargs into SFTConfig
    #   - tokenizer= became processing_class=
    # Instead of guessing the version, introspect the actual signatures and pass only the
    # kwargs each accepts. Robust across old and new TRL.
    import inspect
    from trl import SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common_args = dict(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    config_kwargs = dict(common_args)
    # sequence-length kwarg name differs by version
    if "max_length" in sft_params:
        config_kwargs["max_length"] = 1024
    elif "max_seq_length" in sft_params:
        config_kwargs["max_seq_length"] = 1024
    # dataset_text_field lives in SFTConfig in newer TRL
    if "dataset_text_field" in sft_params:
        config_kwargs["dataset_text_field"] = "text"
    sft_config = SFTConfig(**config_kwargs)

    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs = dict(model=model, args=sft_config, train_dataset=dataset)
    # tokenizer kwarg name differs by version
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer_kwargs[tok_key] = tokenizer
    # if this TRL still wants dataset_text_field on the trainer, provide it there too
    if "dataset_text_field" in trainer_params and "dataset_text_field" not in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    # --- 5. Save adapter ---
    adapter_dir = OUTPUT_DIR / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")

    if args.merge or args.push_to_hub:
        from peft import PeftModel
        print("Merging adapter into base weights (loads base in fp16, needs more RAM/VRAM)...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged_dir = OUTPUT_DIR / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Saved merged model to {merged_dir}")

        if args.push_to_hub:
            if not args.hub_model_id:
                raise SystemExit("--push_to_hub requires --hub_model_id")
            merged.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))
            tokenizer.push_to_hub(args.hub_model_id, token=os.environ.get("HF_TOKEN"))
            print(f"Pushed merged model to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
