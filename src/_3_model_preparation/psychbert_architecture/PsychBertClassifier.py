import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class PsychBertClassifier(nn.Module):
    def __init__(self, hf_model_identifier="mnaylor/psychbert-cased"):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model_identifier, num_labels = 2, from_flax=True)
        self.classif_config = {
            "id2label": {0: "not suicidal", 1: "suicidal"},
            "label2id": {"not suicidal": 0, "suicidal": 1},
            "num_labels": 2
        }

    def forward(self, x, attention_mask):
        #batch_size, n_tokens = x.shape
        return self.model(x, attention_mask=attention_mask).logits

    def __str__(self):
        return "Psychbert classifier"