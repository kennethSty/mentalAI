from src._0_data_preparation.Tokenizer import EmoBertTokenizer, PsychBertTokenizer, GPTTokenizer

GPT_BASEMODEL_CONFIG = {
    "model_size": "124M",
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
}

BERT_CONFIG = {
    "out_emb_dim": 768
}

FINETUNE_CONFIG = {
    "max_length": 499,
    "batch_size": 8,
    "num_epochs":1,
    "eval_freq":500,
    "checkpoint_freq":2000,
    "eval_iter":20,
    "lr":5e-5,
    "weight_decay":0.1
}

FINETUNE_EVAL_CONFIG = {
    "batch_size": 2,
    "num_eval_batches": 100,
}

SUICIDE_DS_CONFIG = {
    "max_token_length": 499,
    "train_frac": 0.75,
    "val_frac": 0.1
}

TOKENIZER_CONFIG = {
    "gpt2": GPTTokenizer(),
    "psychbert": PsychBertTokenizer(),
    "emobert": EmoBertTokenizer()
}


