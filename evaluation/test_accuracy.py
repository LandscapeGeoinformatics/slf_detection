import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# define sites and ids
site_ids = {
    "center": "64531",
    "east": "64564",
    "south": "54452",
    "west": "52981"
}

sites = ["center", "east", "south", "west"]

# define base path
base_label = "/landscape_elements/working/test_sites/label"
base_pred = "/prediction"


# loop open
test_data = {}

for site in sites:
    site_id = site_ids[site]
    site_name = f"{site_id}_{site}"

    label_path = f"{base_label}/label_{site_name}.tif"
    pred_path = f"{base_pred}/{site_name}_pred_mask.tif"

    label = np.array(Image.open(label_path))
    prediction = np.array(Image.open(pred_path))

    test_data[site] = {"label": label, "prediction": prediction}

print(test_data["center"]["prediction"].shape)

# concatenate each array
label = np.concatenate([d["label"] for d in test_data.values()])
prediction = np.concatenate([d["prediction"] for d in test_data.values()])

prediction = np.where(prediction==0, 0, prediction) #255

# Ground truth: binary mask
y_true = label.flatten()

# Predicted probabilities: same shape as label
y_score = prediction.flatten()

print("Label min/max:", y_true.min(), y_true.max())
print("Predicted min/max:", y_score.min(), y_score.max())

# change NA value to 0 if not already done
#y_score = np.where(y_score == 255, 0, y_score)

#center

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
avg_precision = average_precision_score(y_true, y_score)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png", dpi=300)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig("pr_curve.png", dpi=300)
plt.show()

# Best PR threshold (max F1-score, class imbalance, few positive class)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # avoid division by zero
best_idx = np.argmax(f1_scores)
best_threshold_pr = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
print(f"Best threshold: {best_threshold_pr:.3f}")
print(f"Best F1-score: {best_f1:.3f}")

## All metrics

predicted_bin = y_score > best_threshold_pr #0.766#best_thresh


TP = np.logical_and(predicted_bin == 1, y_true == 1).sum()
FP = np.logical_and(predicted_bin == 1, y_true == 0).sum()
TN = np.logical_and(predicted_bin == 0, y_true == 0).sum()
FN = np.logical_and(predicted_bin == 0, y_true == 1).sum()

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

# Intersection: both prediction and label are 1
intersection = np.logical_and(predicted_bin == 1, y_true == 1).sum()

# Union: either prediction or label is 1
union = np.logical_or(predicted_bin == 1, y_true == 1).sum()

# Avoid divide by zero
iou = intersection / union if union != 0 else 0

print(f"True Positive: {TP}")
print(f"False Positive: {FP}")
print(f"True Negative: {TN}")
print(f"False Negative: {FN}")


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


print(f"IoU: {iou:.4f}")

# loop over sites for per-site metrics
results = {}

for site, data in test_data.items():
    label = data["label"]
    prediction = data["prediction"]

    # change NA value (255) to 0
    prediction = np.where(prediction == 255, 0, prediction)

    # flatten
    y_true = label.flatten()
    y_score = prediction.flatten()

    # binarize prediction using best threshold
    predicted_bin = (y_score > best_threshold_pr).astype(int)

    # confusion matrix
    TP = np.logical_and(predicted_bin == 1, y_true == 1).sum()
    FP = np.logical_and(predicted_bin == 1, y_true == 0).sum()
    TN = np.logical_and(predicted_bin == 0, y_true == 0).sum()
    FN = np.logical_and(predicted_bin == 0, y_true == 1).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    intersection = np.logical_and(predicted_bin == 1, y_true == 1).sum()
    union = np.logical_or(predicted_bin == 1, y_true == 1).sum()
    iou = intersection / union if union != 0 else 0

    results[site] = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
    }

# show results
for site, metrics in results.items():
    print(f"\n--- {site.upper()} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# convert to dataframe for summary / export
df_results = pd.DataFrame(results).T
print(df_results)

# optional: export to CSV
df_results.to_csv("site_metrics.csv", index=True)