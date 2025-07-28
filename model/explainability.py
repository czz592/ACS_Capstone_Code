# explainability.py
# Version: 1.0
# xai.py

import shap
from captum.attr import IntegratedGradients
import torch
import matplotlib.pyplot as plt
import numpy as np


def explain_with_shap(model, dataloader, feature_names, class_names, num_samples=100):
    model.eval()
    device = next(model.parameters()).device

    # Collect background and test data from loader
    X_all = []
    for X, _ in dataloader:
        X_all.append(X.to(device))
        if sum(x.shape[0] for x in X_all) >= num_samples * 2:
            break

    X_all = torch.cat(X_all, dim=0)
    background = X_all[:num_samples]
    test_data = X_all[num_samples:2*num_samples]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(test_data)

    shap.summary_plot(shap_values, test_data.cpu().numpy(),
                      feature_names=feature_names,
                      class_names=class_names)
    return shap_values


def explain_with_captum(model, dataloader, feature_names, target=0, num_samples=100):
    model.eval()
    device = next(model.parameters()).device

    # Collect test and baseline data
    X_all = []
    for X, _ in dataloader:
        X_all.append(X.to(device))
        if sum(x.shape[0] for x in X_all) >= num_samples * 2:
            break

    X_all = torch.cat(X_all, dim=0)
    baseline = X_all[:num_samples]
    test_data = X_all[num_samples:2*num_samples]

    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(
        test_data, baselines=baseline, target=target, return_convergence_delta=True)

    mean_attr = attr.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_attr)), mean_attr)
    plt.xticks(range(len(mean_attr)), feature_names, rotation=90)
    plt.title("Integrated Gradients Feature Attribution")
    plt.tight_layout()
    plt.show()

    return mean_attr, delta
