import pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class Data:

    @classmethod
    def data_loader(cls,path, batch_size):

        cls.path = path

        input_ids, labels, num_unique_movies = cls.__load_ml_1m()
        X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.25, random_state=42)

        # Train
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        attn_train = [[1]*10]*X_train.shape[0]
        attn_train = torch.tensor(attn_train)
        X_train_dataset = TensorDataset(X_train, attn_train, y_train)
        X_train_loader = DataLoader(X_train_dataset, batch_size=batch_size, shuffle=True)

        return X_train_loader, num_unique_movies

    @classmethod
    def __load_ml_1m(cls):

        df = pd.read_csv(cls.path)
        sorted_grouped_data = df.sort_values('timestamp').groupby('userId')['movieId'].apply(list)
        input_ids, labels = [], []
        num_unique_movies = len(set(df['movieId']))
        for inp in sorted_grouped_data:
            if len(inp)<=10:
                continue

            input_ids.append(inp[:10])
            labels.append(*inp[11:12])
        
        return input_ids, labels, num_unique_movies