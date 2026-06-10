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

test_folder = "test_cropped"
results = []
y_pred = []
confidences = []

print("\n--- ANALYZING TEST DATASET ---")

for image_name in os.listdir(test_folder):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
    
    image_path = os.path.join(test_folder, image_name)
    
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
        
        y_pred.append(final_pred)
        confidences.append(confidence)
        
        results.append(f"{image_name} -> Prediction: {final_pred} (Score: {confidence:.2f})")
        print(f"Processed {image_name}")

    except Exception as e:
        print(f"Error with {image_name}: {e}")

# SAVE TEXT REPORT
with open("test_results_report.txt", "w") as f:
    f.write("\n".join(results))

# --- GENERATE PERFORMANCE CHARTS ---
print("\n--- GENERATING CHARTS ---")
plt.style.use('dark_background')

# 1. Confidence Distribution Chart (Shows overall model certainty)
plt.figure(figsize=(10, 6))
sns.histplot(confidences, bins=15, kde=True, color='#10b981')
plt.axvline(0.6, color='red', linestyle='--', label='Threshold (0.6)')
plt.title("Model Prediction Confidence (All Test Images)")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("report_confidence_dist.png")

# 2. Prediction Frequency (Shows what the model is identifying)
plt.figure(figsize=(10, 6))
unique_preds, counts = np.unique(y_pred, return_counts=True)
plt.bar(unique_preds, counts, color='#10b981')
plt.title("Total Identification Counts")
plt.ylabel("Number of Images")
plt.savefig("report_prediction_counts.png")

# # 2. Confusion Matrix
# labels = sorted(list(set(y_true) | set(y_pred)))
# cm = confusion_matrix(y_true, y_pred, labels=labels)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
# plt.title("Confusion Matrix: Actual vs Predicted")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.savefig("report_confusion_matrix.png")


print("\n✅ Report Generated!")
print("Files saved: test_results_report.txt, report_confidence_dist.png, report_prediction_counts.png")
