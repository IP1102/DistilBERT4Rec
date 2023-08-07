import torch.nn as nn
from transformers import DistilBertModel

class DistilSequentialRecommender(nn.Module):
    def __init__(self, distilbert_model_name, sequence_length, hidden_size, num_classes):
        super(DistilSequentialRecommender, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name)
        self.pooling = nn.AvgPool1d(sequence_length)
        self.fc = nn.Linear(self.distilbert.config.dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        distilbert_output = self.distilbert(input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.pooling(distilbert_output.permute(0, 2, 1)).squeeze(2)
        fc_output = self.fc(pooled_output)
        logits = self.output_layer(fc_output)
        return logits