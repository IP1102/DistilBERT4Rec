import torch.nn as nn, os
from transformers import BertModel


class SequentialRecommender(nn.Module):
    def __init__(self, bert_model_name, sequence_length, hidden_size, num_classes):
        super(SequentialRecommender, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.pooling = nn.AvgPool1d(sequence_length)
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        pooled_output = self.pooling(bert_output.permute(0, 2, 1)).squeeze(2)
        fc_output = self.fc(pooled_output)
        logits = self.output_layer(fc_output)
        return logits

