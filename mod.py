from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
model.train(
    data="C:/Users/Rohan Kanegaonkar/OneDrive/Desktop/final_mo/detasets.yaml",  # Path to the dataset configuration file
    epochs=1,            # Number of training epochs
    imgsz=640,            # Image size for training
    batch=16,             # Batch size
    name="my_project"  # Name of the model run
)

# Validate the model
metrics = model.val()
print(metrics)  # Outputs precision, recall, mAP, etc.


results = model.predict(source="C:/Users/Rohan Kanegaonkar/OneDrive/Desktop/final_mo/frames/images", save=True)


import os
import numpy as np
# Load ground truth labels
def load_labels(label_dir):
    labels = []
    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file)) as f:
            labels.append([line.strip().split() for line in f.readlines()])
    return labels

# Load predictions
def load_predictions(pred_dir):
    predictions = []
    for file in os.listdir(pred_dir):
        with open(os.path.join(pred_dir, file)) as f:
            predictions.append([line.strip().split() for line in f.readlines()])
    return predictions

ground_truth = load_labels("path/to/ground_truth_labels")
predictions = load_predictions("runs/predict/exp/labels")


from sklearn.metrics import confusion_matrix

# Assuming y_true and y_pred are lists of ground truth and predicted class IDs
y_true = [...]  # Replace with actual ground truth class IDs
y_pred = [...]  # Replace with predicted class IDs from predictions

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

# Display metrics
print(f"Accuracy: {accuracy}")
print(f"Specificity: {specificity}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")



def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "f1_score": 2 * (tp / (tp + fp) * tp / (tp + fn)) / (tp / (tp + fp) + tp / (tp + fn))
    }
    return metrics

metrics = calculate_metrics(y_true, y_pred)
print(metrics)
