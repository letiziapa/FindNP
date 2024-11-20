import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def complete_data_preprocessing(coeff, sm, data, features):
    
    sm['label'] = 1
    coeff['label'] = 0

    Xsm = sm[features]
    Xcoeff = coeff[features]
    Xdata = data[features]
    X = pd.concat([Xsm, Xcoeff])
    y = pd.concat([sm['label'], coeff['label']])

    Xsm_mean = Xsm.mean()
    Xsm_std = Xsm.std()

    Xsm_scaled = (Xsm - Xsm_mean)/Xsm_std
    Xcoeff_scaled = (Xcoeff - Xsm_mean)/Xsm_std
    Xdata_scaled = (Xdata - Xsm_mean)/Xsm_std
    X_scaled = (X - Xsm_mean)/Xsm_std

    #transforming to torch tensors
    Xsm_scaled = torch.tensor(Xsm_scaled.values, dtype=torch.float32)
    Xcoeff_scaled = torch.tensor(Xcoeff_scaled.values, dtype=torch.float32)
    Xdata_scaled = torch.tensor(Xdata_scaled.values, dtype=torch.float32)
    
    return  Xcoeff_scaled, X_scaled, Xsm_scaled, Xdata_scaled, y

def parton_data_preprocessing(c : str, path: str, scaler ='robust'):

    if c == 'c8dt':
        coeff = pd.read_json(path + '/c8dt.json')
    elif c == 'c8qt':
        coeff = pd.read_json(path + '/c8qt.json')
    elif c == 'c8dt2':
        coeff = pd.read_json(path + '/c8dt2.json')
    elif c == 'c8qt2':
        coeff = pd.read_json(path + '/c8qt2.json')
    else:
        raise ValueError('Invalid coefficient. Please select c8dt, c8qt, c8dt2 or c8qt2')

    sm = pd.read_json(path + '/sm.json')

    sm['label'] = 1
    coeff['label'] = 0

    features = ['mtt', 'ytt']
    Xsm = sm[features]
    Xcoeff = coeff[features]

    X = pd.concat([Xsm, Xcoeff])
    y = pd.concat([sm['label'], coeff['label']])

    if scaler == 'robust':
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        Xsm = scaler.transform(Xsm)
        Xcoeff = scaler.transform(Xcoeff)
    elif scaler == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        Xsm = scaler.transform(Xsm)
        Xcoeff = scaler.transform(Xcoeff)
    else:
        raise ValueError('Invalid scaler. Please select robust or standard')
    return X, Xsm, Xcoeff, y

def dataset_loader(X, y, test_size=0.2, random_state=42, batch_size = 200000, device = 'cpu'):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self, input_size = 2):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(self.input_size, 50)
        self.layer2 = nn.Linear(50, 70)
        self.layer3 = nn.Linear(70, 150)
        self.layer4 = nn.Linear(150, 100)
        self.layer5 = nn.Linear(100, 50)
        self.layer6 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.sigmoid(self.layer6(x))
        
        return x

def train_test_model(model, criterion, optimizer, train_loader, test_loader, epochs=200, save = False, path = 'weights/model_weights.pth'):
    avg_train_losses = []
    avg_test_losses = []
    for epoch in range(epochs):
        losses = []
        for batch, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        avg_loss = np.average(losses)
        avg_train_losses.append(avg_loss)
        if epoch % 10 == 0:

            print(f'Epoch {epoch}\nTrain Loss {avg_loss}')
        model.eval()
        with torch.no_grad():
            for batch, (x,y) in enumerate(test_loader):
                y_pred = model(x).flatten()
                test_loss = criterion(y_pred, y)
                avg_test_losses.append(test_loss.item())
            if epoch % 10 == 0:
                print(f'Test Loss {test_loss.item()}')
                print('-----------------------------------')
            if save:
                torch.save(model.state_dict(), path)
            else:
                pass
           
    return avg_train_losses, avg_test_losses

def plot_confidence_regions(samples, params, confidence_levels, save = False, path = 'confidence_regions.pdf'):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for i, confidence_level in enumerate(confidence_levels):
        sns.kdeplot(
            x=samples[:, 0],
            y=samples[:, 1],
            levels=[1 - confidence_level / 100.0, 1.0],
            bw_adjust=1.2,
            ax=ax[i],
            fill=True,
            alpha=0.9,
            color="blue",
        )
        ax[i].set_title(f"{confidence_level}% Confidence Region", fontsize=21, fontname = 'cmr10')
        ax[i].set_xlabel(params[0], fontsize=22, fontname = 'cmr10')
        ax[i].set_ylabel(params[1], fontsize=22, fontname = 'cmr10')
        ax[i].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    if save:
        plt.savefig(path)
   
    else:
        plt.show()