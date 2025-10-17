import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# ===============================
# CLASE DEL DATASET
# ===============================
class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ruta = row['ruta_npy']
        if not ruta.endswith('.npy'):
            ruta += '.npy'

        vol = np.load(ruta)
        vol = (vol - vol.mean()) / vol.std()
        vol = torch.tensor(vol).unsqueeze(0).float()  # (1, D, H, W)
        label = torch.tensor(row['label_progresion']).float()
        return vol, label


# ===============================
# FUNCIÓN PARA CREAR LOS SPLITS
# ===============================
def crear_dataloaders(path_csv, batch_size=4):
    df = pd.read_csv(path_csv)

    # Corregir rutas si es necesario
    df['ruta_npy'] = df['ruta_npy'].apply(lambda r: r if r.endswith('.npy') else r + '.npy')
    df = df[df['ruta_npy'].apply(os.path.exists)]

    # Nivel de sujeto
    df_subjects = df.groupby('sujeto_id')['label_progresion'].first().reset_index()
    X = df_subjects[['sujeto_id']]
    y = df_subjects['label_progresion']

    # Split 80/20 (trainval / test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(sss1.split(X, y))

    trainval_subjects = X.iloc[trainval_idx]['sujeto_id']
    test_subjects = X.iloc[test_idx]['sujeto_id']

    # Split 75/25 (train / val)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=24)
    X_trainval = X.iloc[trainval_idx]
    y_trainval = y.iloc[trainval_idx]
    train_idx, val_idx = next(sss2.split(X_trainval, y_trainval))

    train_subjects = X_trainval.iloc[train_idx]['sujeto_id']
    val_subjects = X_trainval.iloc[val_idx]['sujeto_id']

    # Filtrar imágenes
    X_train = df[df['sujeto_id'].isin(train_subjects)]
    X_val   = df[df['sujeto_id'].isin(val_subjects)]
    X_test  = df[df['sujeto_id'].isin(test_subjects)]

    # Crear datasets y dataloaders
    train_ds = MRIDataset(X_train)
    val_ds   = MRIDataset(X_val)
    test_ds  = MRIDataset(X_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    print(f"Train: {len(train_ds)} imágenes | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader
