import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class PsychBertClassifier(nn.Module):
    def __init__(self, hf_model_identifier="mnaylor/psychbert-cased", num_classes = 2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_identifier, from_flax=True)

    def forward(self, x):
        batch_size, n_tokens = x.shape
        return self.model(x).logits
