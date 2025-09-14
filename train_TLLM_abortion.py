from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, json, random

OPT_TOKENS = [
  "<OPT_STRONG_ANTI>", "<OPT_ANTI>", "<OPT_NEUTRAL>", "<OPT_PRO>", "<OPT_STRONG_PRO>"
]
opt2id = {}

model_name = "meta-llama/llama-3.1-8b" # or "mistralai/Mistral-7B-v0.3" or "QWEN 2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.add_special_tokens({"additional_special_tokens": OPT_TOKENS})

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
model.resize_token_embeddings(len(tokenizer))

# Optional QLoRA
# model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj"]
)
model = get_peft_model(model, lora_cfg)

# map special tokens to ids AFTER resize
opt2id = {opt: tokenizer.convert_tokens_to_ids(opt) for opt in OPT_TOKENS}
opt_ids = torch.tensor([opt2id[o] for o in OPT_TOKENS], device=model.device)

# --- Dataset ---
class TLLMRowDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, shuffle_opts=True):
        self.recs = [json.loads(x) for x in open(jsonl_path)]
        self.shuffle_opts = shuffle_opts
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        # Expect fields: prompt_text, to_dist (list float), weight (float)
        prompt = r["prompt_text"]
        # Randomize option order to reduce position bias
        order = list(range(len(OPT_TOKENS)))
        if self.shuffle_opts:
            random.shuffle(order)
        # Reorder target distribution accordingly
        to_dist = [r["to_dist"][j] for j in order]
        opt_tokens_ordered = [OPT_TOKENS[j] for j in order]

        prompt_with_opts = prompt + "Options: " + " ".join(opt_tokens_ordered) + "\nAnswer:\n"
        enc = tokenizer(prompt_with_opts, return_tensors="pt", truncation=True, max_length=1024)
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "to_dist": torch.tensor(to_dist, dtype=torch.float),
            "weight": torch.tensor(r.get("weight", 1.0), dtype=torch.float),
            "order": torch.tensor(order, dtype=torch.long),
        }

def data_collate(batch):
    # pad
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    input_ids = []
    attn = []
    for x in batch:
        pad = maxlen - x["input_ids"].shape[0]
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), tokenizer.pad_token_id or tokenizer.eos_token_id)]))
        attn.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "to_dist": torch.stack([x["to_dist"] for x in batch]),
        "weight": torch.stack([x["weight"] for x in batch]),
        "order": torch.stack([x["order"] for x in batch]),
    }

# --- Custom Trainer with KL loss over option tokens ---
import torch.nn.functional as F
class KLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        to_dist = inputs["to_dist"].to(model.device)   # [B, K]
        weight = inputs["weight"].to(model.device)     # [B]
        order  = inputs["order"].to(model.device)      # [B, K]

        out = model(input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device))
        # last token position per sequence
        last_idx = attention_mask.sum(dim=1) - 1  # [B]
        last_hidden = out.logits[torch.arange(out.logits.size(0)), last_idx]  # [B, V]

        # fetch logits for option tokens in the *ordered* list
        # remap global opt_ids according to per-example 'order'
        opt_ids_ordered = torch.stack([opt_ids[o] for o in order])  # [B,K]
        opt_logits = last_hidden.gather(1, opt_ids_ordered)          # [B,K]
        p_llm = F.softmax(opt_logits, dim=1)                         # [B,K]

        # forward KL: sum p_human * (log p_human - log p_llm)
        p_h = (to_dist / (to_dist.sum(dim=1, keepdim=True) + 1e-12)).clamp_min(1e-8)
        loss_vec = (p_h * (p_h.log() - (p_llm + 1e-12).log())).sum(dim=1)  # [B]
        # weight by n_from etc.
        loss = (weight * loss_vec).mean()

        return (loss, out) if return_outputs else loss

args = TrainingArguments(
    output_dir="tllm_abortion_transitions",
    learning_rate=2e-4,  # LoRA can take a higher LR
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    save_total_limit=2,
)

train_ds = TLLMRowDataset("train_rows.jsonl", shuffle_opts=True)
eval_ds  = TLLMRowDataset("val_rows.jsonl",   shuffle_opts=False)

trainer = KLTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collate,
)
trainer.train()
model.save_pretrained("tllm_abortion_transitions_lora")
tokenizer.save_pretrained("tllm_abortion_transitions_lora")
