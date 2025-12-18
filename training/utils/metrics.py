import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score

def calculate_metrics(preds, targets):
    # New
    #targets = targets.flatten()
    #preds = preds.flatten()
    
    preds = (preds[:, 0, :, :] > 0.5).cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()
    # Convert predictions and targets to binary format and transfer to CPU
    #preds = (preds[0] > 0.5).cpu().numpy().flatten()
    #targets = targets[:, 0, :, :].cpu().numpy().flatten()  # Convert one-hot encoded targets back to original shape
    #preds = preds.argmax(dim=1).cpu().numpy().flatten()
    #targets = targets.argmax(dim=1).cpu().numpy().flatten()  # Convert one-hot encoded targets back to original shape

    # Calculate metrics
    precision = precision_score(targets, preds, average='binary', zero_division=1)
    recall = recall_score(targets, preds, average='binary', zero_division=1)
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='binary', zero_division=1)
    iou = jaccard_score(targets, preds, average='binary')

    return precision, recall, accuracy, f1, iou