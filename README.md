# LoRA Fine-Tuning Pipeline (End-to-End)

## Overview

This project fine-tunes a pre-trained language model using LoRA (Low-Rank Adaptation) to adapt model behavior efficiently without updating full model weights.

---

## 1. Load Base Model

* Load a pre-trained model and tokenizer from Hugging Face
* The base model remains frozen during training

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

---

## 2. Prepare Dataset

* Create data in **Instruction → Response** format
* This helps the model learn conversational behavior

Example:

```text
### Instruction: I feel stressed
### Response: It sounds like you're overwhelmed...
```

---

## 3. Tokenization

* Convert text into numerical tokens

```python
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize)
```

* Output:

  * `input_ids` → tokenized text
  * `attention_mask` → identifies valid tokens

---

## 4. Add Labels

* Labels define the target output for training

```python
def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

dataset = dataset.map(add_labels)
```

* Enables next-token prediction learning

---

## 5. Apply LoRA

* Add trainable adapters to attention layers

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

* Only LoRA parameters are updated
* Base model remains unchanged

---

## 6. Define Training Configuration

* Control how training runs

```python
training_args = TrainingArguments(
    output_dir="./lora-act",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50
)
```

---

## 7. Train Model

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

* Model learns patterns using loss minimization
* Only LoRA layers are updated

---

## 8. Save Model

```python
model.save_pretrained("act-lora-final")
tokenizer.save_pretrained("act-lora-final")
```

* Saves trained LoRA adapters

---

## 9. Load and Test

```python
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "act-lora-final")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
```

---

## Key Concepts

* **Tokenization** → text to numerical format
* **Labels** → define target sequence
* **LoRA** → efficient fine-tuning via small adapters
* **Attention (Q, V)** → modified to change behavior
* **Autoregressive training** → predicts next token

---

## Summary

The pipeline follows:

```text
Load Model → Prepare Dataset → Tokenize → Add Labels
→ Apply LoRA → Configure Training → Train → Save → Test
```

This approach enables efficient fine-tuning with minimal compute while preserving base model knowledge.
