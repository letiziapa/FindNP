"""
@file funcs.py
@brief Module containing functions for different parts of the analysis.

@details This module contains the approprate tools to pre-process the data, train and 
evaluate the neural networks.
It also provides the function to compute the confidence intervals in the nested sampling results
at different levels.
@author Created by Letizia Palmas (lp645)
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def complete_data_preprocessing(coeff, sm, data, features):
    """
    @brief Pre-process the full-feature datasets.

    @details This function assigns label 0 to the Effective Field Theory data
    and label 1 to the Standard Model data.
    It then constructs the datasets on which the Neural Networks will be trained
    by concatenating the Monte Carlo data of the SM and EFT.
    Each point in the datasets are standardised equivalently by subtracting the mean of the SM data
    and scaling by its standard deviation.
    The scaled datasets are then converted to Torch Tensors for training.

    @param coeff (pandas.DataFrame)
        Pandas DataFrame corresponding to the Monte Carlo samples of the selected Wilson coefficient
    @param sm (pandas.DataFrame)
        Pandas DataFrame of the Monte Carlo generated samples of the Standard Model data
    @param data (pandas.DataFrame)
        Pandas DataFrame of the observed data
    @param features (list)
        Features to be included in the X training data
    
    @return Xcoeff_scaled (torch.tensor)
        Tensor of the EFT scaled features
    @return X_scaled (torch.tensor)
        Tensor of the combined EFT and SM features to be used in the neural network training
    @return Xsm_scaled (torch.tensor)
        Tensor of the SM scaled features
    @return Xdata_scaled (torch.tensor)
        Tensor of the scaled observed data
    @return y (torch.tensor)
        Tensor containing the labels to use in supervised learning classification
    
    """
    #define the labels
    sm['label'] = 1
    coeff['label'] = 0

    #define training and classification set
    Xsm = sm[features]
    Xcoeff = coeff[features]
    Xdata = data[features]
    X = pd.concat([Xsm, Xcoeff])
    y = pd.concat([sm['label'], coeff['label']])

    Xsm_mean = Xsm.mean()
    Xsm_std = Xsm.std()

    #standardisation
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
    """
    @brief Load and process the parton level data.

    @details This function reads JSON files containing parton level data for different Wilson coefficients
    and the Standard Model.
    It assigns label 1 to the SM and label 0 to the EFT data.
    It then concatenates the X datasets and applies the chosen scaling.
    If 'scaler' = 'robust', the data is scaled according to the median and interquartile range
    of the SM + EFT data.
    If 'scaler' = 'standard', the data is standardised using the mean and standard deviation
    of the SM + EFT data.

    @param c (str)
        Coefficient identifier. Valid options are 'c8dt', 'c8qt', 'c8dt2', 'c8qt2'.
    @param path (str)
        The folder where the JSON files are located.
    @param scaler (str) 
        The type of scaling to apply to the data. Options are 'robust' for RobustScaler 
        and 'standard' for StandardScaler. Defaults to 'robust'.

    @return  X (numpy.ndarray)
        The scaled features of both SM and coefficient data combined.
    @return Xsm (numpy.ndarray)
        The scaled features of the SM data.
    @return Xcoeff (numpy.ndarray)
        The scaled features of the EFT data.
    @return y (pandas.Series)
        The labels for the data, with 1 for SM and 0 for EFT data.
    """
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
    """
    @brief Prepare the dataset for training.
    @details This function splits the X and y datasets into training and testing sets
    using the specified test size and random state.
    It then converts them into Pytorch tensors, ensuring they are of the correct float32 data type and 
    sends them to the specified device.
    The tensors are then wrapped into DataLoader objects, setting the batch size 
    and enabling shuffling for the training set to facilitate batch processing during model training.

    @param X (numpy.ndarray or pandas.DataFrame) The input features for the dataset.
    @param y (numpy.ndarray or pandas.Series) The target labels for the dataset.
    @param test_size (float) The fraction of the dataset that will be allocated to the test set. Defaults to 0.2.
    @param random_state (int) Controls the shuffling applied to the data before applying the split. 
    Specify the random state to reproduce the same output across multiple function calls. Defaults to 42.
    @param batch_size (int): The size of the batches for loading the data. Defaults to 200000.
    @param device (str): The device to which the tensors will be sent ('cpu' or 'cuda'). Defaults to 'cpu'.

    @return train_loader DataLoader object containing the training set.
    @return test_loader DataLoader object containing the testing set.

    """

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
    """
    @brief Class containing the MultiLayer Perceptron (MLP) used in the analysis.
    @details The MLP class creates a neural network model consisting of six linear layers 
    with ReLU activations for the first five layers and a sigmoid activation for the output layer.
    This architecture is specifically designed for binary classification tasks
    due to the range of values that the sigmoid function outputs spans.

    Attributes:
    input_size (int): The size of the input features. Defaults to 2.
    layer1 to layer6 (nn.Linear): Linear layers of the neural network.

    Methods:
    forward(x): Defines the forward pass of the model.

    Parameters:
    x (torch.Tensor): The input tensor containing the features.

    Returns:
    torch.Tensor: The output tensor after passing through the layers and activations.

   
    """
    def __init__(self, input_size = 2):
        """
        @brief Function that initialises the MLP.
        @details The MLP consists of 6 linear layers of variable size. The output is always unidimensional.
        @param input_size (int)  The size of the input features. Defaults to 2, which reflects the structure 
        of parton level data. It should be changed to 14 in the integrated analysis.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(self.input_size, 50)
        self.layer2 = nn.Linear(50, 70)
        self.layer3 = nn.Linear(70, 150)
        self.layer4 = nn.Linear(150, 100)
        self.layer5 = nn.Linear(100, 50)
        self.layer6 = nn.Linear(50, 1)

    def forward(self, x):
        """
        @brief Forward pass of the MLP.
        @details The first 5 layers are passed through a Rectified Linear Unit (ReLU) activation function.
        The final layer is passed through a sigmoid function to ensure that the output of the neural network
        lies between 0 and 1.
        @param x (torch.Tensor) The input tensor containing the features.
        @return x (torch.Tensor) The output tensor after passing through the layers and activations.
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = torch.sigmoid(self.layer6(x))
        
        return x

def train_test_model(model, criterion, optimizer, train_loader, test_loader, epochs=200, save = False, path = 'weights/model_weights.pth'):
    """
    @brief Function for training and optimising the MLP
    @details The function trains the model for a specified number of epochs,
    optimises it using the specified optimiser
    evaluating it on the test data at each epoch.
    It calculates the average train and test losses per epoch and saves them to a list. 
    The losses are printed every 10 epochs. 
    The model weights can be saved to the specified path if 'save' is set to True.

    @param model (torch.nn.Module) The model to be trained.
    @param criterion The loss function.
    @param optimizer The optimizer for updating the model weights.
    @param train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
    @param test_loader (torch.utils.data.DataLoader): DataLoader for the testing data.
    @param epochs (int) The number of epochs to train the model. Defaults to 200.
    @param save (bool) If True, saves the model weights after training. Defaults to False.
    @param path (str) Path where the model weights should be saved if 'save' = True. Defaults to 'weights/model_weights.pth'.
    
    @return avg_train_losses (list) A list of the average training losses per epoch.
    @return avg_test_losses (list) A list of the average testing losses per epoch.
    """
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

def find_nsigma(pvalue):
    """
    @brief Find the number of standard deviations.
    @details Create a function to find the n-sigma value corresponding to a given p-value in a normal distribution.
    The n-sigma value is a way to express the probability of a certain event in terms of the number of standard
    deviations (sigma) away from the mean of a normal distribution. This function returns another function that
    calculates the difference between the calculated p-value for a given n-sigma and the provided p-value.

    @param pvalue (float) The p-value used to calculate the corresponding n-sigma.
    The p-value should be between 0 and 1.

    @return func (function) A function that takes n-sigma (x) as input and returns the difference between the
    calculated p-value for this n-sigma value and the provided p-value. 
    """
    def func(x):
        return (1 - (scipy.stats.norm.cdf(x) - scipy.stats.norm.cdf(-x))) - pvalue

    return func

def plot_confidence_regions(samples, params, confidence_levels, save = True, path = 'plots/confidence_regions.png'):
    """
    @brief Plot 2D confidence regions at different levels
    @details The function takes the samples produced by Nested Sampling and
    creates a figure with three subplots, each showing the parameters confidence regions for different confidence levels.
    The confidence regions are plotted using kernel density estimation with seaborn's kdeplot function. 
    The plots can be saved to a file if 'save' is set to True.

    @param samples (numpy.ndarray) The samples used to plot the confidence regions. 
    @param params (list of str): The names of the parameters corresponding to the samples.
    @param confidence_levels (list of float): The confidence levels to plot. Each value should be between 0 and 100.
    @param save (bool) If True, the plot will be saved to the specified path. Defaults to False.
    @param path (str) The file path where the plot should be saved if `save` is True. Defaults to 'plots/confidence_regions.png'.
    """
 
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    #create subplots
    for i, confidence_level in enumerate(confidence_levels):
        #use kde to plot the areas corresponding to the confidence levels
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
