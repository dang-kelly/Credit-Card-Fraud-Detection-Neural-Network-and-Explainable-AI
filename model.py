import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from numpy.random import seed

import imblearn
from imblearn.under_sampling import RandomUnderSampler  

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

## Set random seeds for reproducibility
seed(42)
torch.manual_seed(42)

# ------------- Data Preprocessing -------------

## Import dataset
df = pd.read_csv('creditcard.csv')
print(df.head())

## Create Train-test 80/20 split
X = df.drop('Class', axis=1).values
Y = df['Class'].values 
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

## Normalize Xtrain, and apply the transformations to Xtest
scalar = MinMaxScaler()
Xtrain = scalar.fit_transform(Xtrain)
Xtest = scalar.transform(Xtest)

## Check missing values in Xtrain, compute fill values and apply them to Xtrain
print("Missing values in Xtrain:", np.sum(np.isnan(Xtrain)))
    # None
## Check missing values in Xtest
print("Missing values in Xtest:", np.sum(np.isnan(Xtest)))
    # None

## Handle class imbalance by undersampling
# Visualize class imabalance for whole dataset
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution Before Undersampling')
plt.tight_layout()
plt.savefig('Class_Distribution_Before_Undersampling.png', dpi=300, bbox_inches="tight")
plt.show()
    # Data is highly unabalnced, majority of data classified as 0
    # We will only undersample on the training data and test on the original, observed imabalance for honest results

# Apply RandomUnderSampler to training data
undersampler = RandomUnderSampler(sampling_strategy=0.52, random_state=42)
Xtrain, Ytrain = undersampler.fit_resample(Xtrain, Ytrain)

# Visualize class imabalance for training dataset after undersampling
plt.figure(figsize=(6, 4))
sns.countplot(x=Ytrain, data=pd.DataFrame(Ytrain, columns=['Class']))
plt.title('Class Distribution of Training Data After Undersampling')
plt.tight_layout()
plt.savefig('Training_Class_Distribution_After_Undersampling.png', dpi=300, bbox_inches="tight")
plt.show()



#------------- Create Torch Datasets and Dataloaders -------------

# Choose batch size of 100
bs = 100
train_ds = torch.utils.data.TensorDataset(
    torch.tensor(Xtrain).float(),
    torch.tensor(Ytrain).float().unsqueeze(1)   # add dimension to second position to match output shape of model
)
test_ds = torch.utils.data.TensorDataset(
    torch.tensor(Xtest).float(),
    torch.tensor(Ytest).float().unsqueeze(1)
)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False)



#------------- Model Building -------------

# Create feedforward neural network
class FFNN(nn.Module):
    def __init__(self, input_n, hidden_n, output_n, dr):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_n, hidden_n),   # 16 learned features from 30 input features
            nn.ReLU(),
            nn.Linear(hidden_n, hidden_n),  # 16 more abstract features learned from previous 16
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_n, output_n)
        )
    def forward(self, x):
        return self.net(x)

# Define loss function with optimization
def loss_batch(model, loss_f, xb, yb, opt=None):
    loss = loss_f(model(xb), yb)   
    if opt is not None:             # Training mode
        loss.backward()             # Compute gradient of loss wrt to weights in linear layers
        opt.step()                  # Update weights in direction that reduces loss
        opt.zero_grad()             # Set gradient to 0 for next x-batch
    return loss.item(), len(xb)     # Return loss as float and number of samples in x-batch

def create_model(input_n, hidden_n, output_n, dr):
    model = FFNN(input_n, hidden_n, output_n, dr)
    return model



#------------- Model Training, Testing, and Evaluation -------------

def train(model, epochs, loss_f, opt, train_dl, device):
    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0, 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, n = loss_batch(model, loss_f, xb, yb, opt)  # Compute loss and update weights for each batch
            total_loss += loss * n
            total_n += n
        train_loss = total_loss / total_n
    
    print(f'Training Loss: {train_loss:.4f}')


def predict(model, xb, device):
    with torch.no_grad():
        xb = xb.to(device)
        outputs = model(xb)
        probabilities = torch.sigmoid(outputs)        # Convert raw linear output (not bounded b/w 0 and 1) to a probability
        predictions = (probabilities >= 0.48).float()  # Convert probability to a binary prediction
    return probabilities.item(), int(predictions.item())   


def test(model, test_dl, device):
    model.eval()
    y_pred = []
    targets = []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            probabilities = torch.sigmoid(outputs)        # Convert raw linear output (not bounded b/w 0 and 1) to a probability
            predictions = (probabilities >= 0.48).float()  # Convert probability to a binary prediction

            y_pred.extend(predictions.cpu().numpy())      # Move tensor from GPU to CPU and convert tensor to numpy array, then extend y_pred list
            targets.extend(yb.cpu().numpy())

    y_pred = np.array(y_pred)       # Convert to numpy arrays for sklearn metrics
    targets = np.array(targets)

    # Evaluation metrics

    print(f'Confusion Matrix: \n {metrics.confusion_matrix(targets, y_pred)}\n')
    print(f'Accuracy: {metrics.accuracy_score(targets, y_pred)}\n')
    print(f'Precision: {metrics.precision_score(targets, y_pred)}\n')
    print(f'Recall: {metrics.recall_score(targets, y_pred)}\n')
    print(f'F1 Score: {metrics.f1_score(targets, y_pred)}\n')



#------------- Save Model -------------

def train_and_save(model, epochs, loss_f, opt, train_dl, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model.to(device)
    
    train(model, epochs, loss_f, opt, train_dl, device)    # Train
    torch.save(model.state_dict(), path)                   # Save learned weights to path
    print(f'Model saved to: {path}\n')

    return device



#------------- Main -------------

def main():
    # Set model parameters
    input_n = Xtrain.shape[1]
    hidden_n = 16
    output_n = 1
    dr = 0.5

    # Create model, train and save, then test
    model = create_model(input_n, hidden_n, output_n, dr)
    pos_weight = torch.tensor([len(Ytrain[Ytrain == 0]) / len(Ytrain[Ytrain == 1])])
    device = train_and_save(
        model,
        epochs=100,
        loss_f=nn.BCEWithLogitsLoss(pos_weight=pos_weight),     # Standard loss function for binary classification
                                                                # Add class-weighted loss
        opt=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9),
        train_dl=train_dl,
        path='FFNN_CCFD.pth'
    )
    
    test(model, test_dl, device)
    
if __name__ == '__main__':
    main()
