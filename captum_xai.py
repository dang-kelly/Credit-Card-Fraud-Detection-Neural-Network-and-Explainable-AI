import numpy as np
import matplotlib.pyplot as plt

import torch

import captum
from captum.attr import (
    Lime, 
    GradientShap, 
    IntegratedGradients
)

from model import df, Xtest, Ytest, FFNN, create_model, predict

# Visualization function
def visualize_attribution(attribution, method, save_path="attribution"):
    plt.figure(figsize=(12, 5))
    features = df.drop('Class', axis=1).columns
    plt.bar(features, attribution.cpu().numpy()[0])
    plt.xlabel('Feature Index')
    plt.ylabel('Attribution Value')
    plt.title(f'{method} Feature Attribution on First Test Sample')
    plt.tight_layout()
    plt.savefig(f'{method}_{save_path}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# Captum function: attribution methods and corresponding attribution plots
def run_captum(model, test_ds_obs, method, device=None):
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    x, y = test_ds_obs
    x = x.unsqueeze(0).float()  # add batch size of 1 for a single sample: (1, 30)
    x, y = x.to(device), y.to(device) 

    with torch.no_grad():
        prob, pred = predict(model, x, device)
    print(f'Probability of fraud: {prob:.4f}')
    print(f'Class Prediction: {pred}')  

    titles = {
        'Lime': 'LIME Explanation',
        'GradientShap': 'Gradient SHAP Explanation',
        'IntegratedGradients': 'Integrated Gradients Explanation'
    }

    baseline = torch.tensor(Xtest[Ytest == 0].mean(axis=0)).float().unsqueeze(0).to(device)

    if method == 'Lime':
        explainer = Lime(model)
        attribution = explainer.attribute(x, n_samples=1000)
    elif method == 'GradientShap':
        explainer = GradientShap(model)
        attribution = explainer.attribute(x, baselines=baseline, n_samples=50, stdevs=0.1)
    else:
        explainer = IntegratedGradients(model)
        attribution = explainer.attribute(x, baselines=baseline, n_steps=50)
    
    print(f'{titles[method]} Attribution: {attribution.cpu().numpy()}')
    visualize_attribution(attribution, titles[method])


def main():
    model = create_model(input_n=30, hidden_n=16, output_n=1, dr=0.5)
    model.load_state_dict(torch.load('FFNN_CCFD.pth'))

    test_ds_obs = (torch.tensor(Xtest[0]).float(), torch.tensor(Ytest[0]).float())

    methods = ['Lime', 'GradientShap', 'IntegratedGradients']

    for method in methods:
        print(f'\nRunning {method}:')
        run_captum(model, test_ds_obs, method=method)

if __name__ == '__main__':
    main()
