import pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class Data:

    def __init__(self, path) -> None:
        self.path = path

    @classmethod
    def data_loader(cls, batch_size):

        input_ids, labels = cls.__load_ml_1m()
        X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.25, random_state=42)

        # Train
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        attn_train = [[1]*10]*X_train.shape[0]
        X_train_dataset = TensorDataset(X_train, attn_train, y_train)
        X_train_loader = DataLoader(X_train_dataset, batch_size=batch_size, shuffle=True)

        return X_train_loader


    def __load_ml_1m(self):

        df = pd.read_csv(self.path)
        sorted_grouped_data = df.sort_values('timestamp').groupby('userId')['movieId'].apply(list)
        input_ids, labels = [], []

        for inp in sorted_grouped_data:
            if len(inp)<=10:
                continue

            input_ids.append(inp[:10])
            labels.append(*inp[11:12])
        
        return input_ids, labels