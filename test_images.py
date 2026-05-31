import os
import cv2
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics import confusion_matrix, classification_report

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Analysis Mode: Using device", device)

# Load Models
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model = joblib.load("models/face_classifier.pkl")

test_folder = "test"
results = []
y_true = []
y_pred = []
confidences = []

print("\n--- ANALYZING TEST DATASET ---")

for image_name in os.listdir(test_folder):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
    
    image_path = os.path.join(test_folder, image_name)
    
    # Extract ground truth from filename (e.g., 'rahman_1.jpg' -> 'rahman')
    # If the filename doesn't contain a known name, we label as 'other' for comparison
    true_label = 'unknown'
    if 'rahman' in image_name.lower(): true_label = 'rahman'
    elif 'ghulam' in image_name.lower(): true_label = 'unknown' # We treat Ghulam as unknown since he wasn't in training
    
    try:
        img = Image.open(image_path).convert('RGB')
        face = mtcnn(img)
        
        if face is None:
            results.append(f"{image_name}: No Face Detected")
            continue

        face = face.unsqueeze(0).to(device)
        embedding = resnet(face).detach().cpu().numpy()
        probs = model.predict_proba(embedding)[0]
        
        confidence = np.max(probs)
        # Access classes from the kneighborsclassifier step of the pipeline
        classes = model.named_steps['kneighborsclassifier'].classes_
        pred_label = classes[np.argmax(probs)]
        
        # Apply threshold logic
        final_pred = pred_label if confidence > 0.6 else "unknown"
        
        y_true.append(true_label)
        y_pred.append(final_pred)
        confidences.append(confidence)
        
        results.append(f"{image_name} | True: {true_label} | Pred: {final_pred} ({confidence:.2f})")
        print(f"Processed {image_name}")

    except Exception as e:
        print(f"Error with {image_name}: {e}")

# SAVE TEXT REPORT
with open("test_results_report.txt", "w") as f:
    f.write("\n".join(results))

# --- GENERATE COMPARISON CHARTS ---
print("\n--- GENERATING CHARTS ---")
plt.style.use('dark_background')

# 1. Confidence Distribution Chart
plt.figure(figsize=(10, 6))
sns.histplot(confidences, bins=15, kde=True, color='#10b981')
plt.axvline(0.8, color='red', linestyle='--', label='Threshold (0.8)')
plt.title("Model Confidence Distribution (All Test Images)")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("report_confidence_dist.png")

# 2. Confusion Matrix
labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
plt.title("Confusion Matrix: Actual vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("report_confusion_matrix.png")

# 3. Accuracy by category
categories = ['rahman', 'unknown']
cat_acc = []
for cat in categories:
    total = sum(1 for t in y_true if t == cat)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p == cat)
    cat_acc.append(correct/total if total > 0 else 0)

plt.figure(figsize=(8, 6))
plt.bar(categories, cat_acc, color=['#10b981', '#334155'])
plt.ylim(0, 1.1)
plt.title("Accuracy by Category")
plt.ylabel("Accuracy %")
plt.savefig("report_category_accuracy.png")

print("\n✅ Report Generated!")
print("Files saved: test_results_report.txt, report_confidence_dist.png, report_confusion_matrix.png, report_category_accuracy.png")
