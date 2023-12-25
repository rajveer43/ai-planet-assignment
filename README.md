# ai-planet-assignment

Access the code here: https://www.kaggle.com/code/rajveerrathod/falcon7b

## Introduction

This repository contains code for fine-tuning a language model using the `BitsAndBytes` library. The goal is to generate informative responses to user queries related to marine toxins, medical queries, and other relevant topics.

## Installation
To set up the environment, run the following commands in your Python environment:

```bash
pip install -Uqqq pip
pip install -qqq bitsandbytes==0.39.0
pip install -qqq torch==2.0.1
pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc
pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f
pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71
pip install -qqq datasets==2.12.0
pip install -qqq loralib==0.1.1
pip install -qqq einops==0.6.1
```

## Usage

Fine-tuning the Model
Clone the repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

### Model Inference
To use the fine-tuned model for generating responses, you can interact with it using the provided Jupyter notebook. Simply open the notebook and follow the instructions for generating responses based on user prompts.

## Model Information
The pre-trained model used for fine-tuning is "tiiuae/falcon-7b-instruct". The fine-tuned model will be saved in the "trained-model" directory.

### Model Archtirecture:

```bash
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(49216, 3072, padding_idx=0)
        (layers): ModuleList(
          (0-23): 24 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=3072, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (k_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
              (v_proj): lora.Linear4bit(
                (base_layer): Linear4bit(in_features=3072, out_features=3072, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3072, out_features=16, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=16, out_features=3072, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (o_proj): Linear4bit(in_features=3072, out_features=3072, bias=False)
              (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
              (up_proj): Linear4bit(in_features=3072, out_features=8192, bias=False)
              (down_proj): Linear4bit(in_features=8192, out_features=3072, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
          )
        )
        (norm): LlamaRMSNorm()
      )
      (lm_head): Linear(in_features=3072, out_features=49216, bias=False)
    )
  )
)
```

## Datasets
The training data is sourced from the "keivalya/MedQuad-MedicalQnADataset". It contains medical questions paired with corresponding answers.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
```
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 1
```

### Additional Information

- The `generate_promp`t function creates conversation prompts in the format "<human>: [question]\n<assistant>: [answer]".
- The model is fine-tuned using the `transformers library`, with specific configurations for `BitsAndBytes` quantization and `Peft` training.
- The fine-tuned model can be pushed to the Hugging Face Model Hub for easy sharing and access.
