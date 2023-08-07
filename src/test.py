import torch

from .bert_model import BertModel
from .distil_bert import DistilSequentialRecommender
from torchmetrics import Precision, Recall
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

class TestModel:

    def __init__(self, model_path, data_loader) -> None:
        self.model = torch.load(model_path)
        self.dataloader = data_loader

    def test(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self.model.to(device)
        model.eval()

        for batch_input_ids, batch_attention_mask, batch_labels in self.dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            with torch.no_grad():
                output = model(batch_input_ids, batch_attention_mask)
                probabilities = F.softmax(output, dim=1)  # Apply softmax along dimension 1
                predicted_labels = torch.argmax(probabilities, dim=1)                

            print(precision_score(precision_score(batch_labels.cpu(), predicted_labels.cpu(), average=None)))
            print(precision_score(recall_score(batch_labels.cpu(), predicted_labels.cpu(), average=None)))

        


