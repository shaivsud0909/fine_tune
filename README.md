# LoRA Fine-Tuning Pipeline 

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

  * `input
