import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 50)
print("HANDWRITTEN DIGIT RECOGNITION")
print("=" * 50)

print("\nLoading Digits Dataset...")
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")
print(f"Image Size: 8x8 pixels (flattened to 64)")
print(f"Number of Classes: {len(np.unique(y))}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample Training Images", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sample_images.png"), dpi=150)
plt.close()
print("Saved: sample_images.png")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nBuilding Neural Network Model (MLPClassifier)")
print("Hidden Layers: (256, 128, 64)")
print("Activation: ReLU")
print("Optimizer: Adam")
print("Max Iterations: 50")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=50,
    batch_size=256,
    random_state=42,
    verbose=True
)

print("\nTraining Model...")
model.fit(X_train_scaled, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(model.loss_curve_, label='Training Loss')
ax.set_title('Model Loss Curve')
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
plt.close()
print("Saved: training_history.png")

y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

report = classification_report(y_test, y_pred)
print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(report)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Predictions vs Actual", fontsize=14)
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    pred_label = y_pred[idx]
    true_label = y_test[idx]
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"Pred: {pred_label} | True: {true_label}", color=color)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "predictions.png"), dpi=150)
plt.close()
print("Saved: predictions.png")

with open(os.path.join(output_dir, "Output.txt"), "w", encoding="utf-8") as f:
    f.write("HANDWRITTEN DIGIT RECOGNITION - OUTPUT REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write("Dataset: Sklearn Digits (8x8 handwritten digits)\n")
    f.write(f"Total Samples: {X.shape[0]}\n")
    f.write(f"Training Samples: {X_train.shape[0]}\n")
    f.write(f"Testing Samples: {X_test.shape[0]}\n")
    f.write(f"Image Size: 8x8 pixels (64 features)\n")
    f.write(f"Number of Classes: 10 (digits 0-9)\n\n")
    f.write("Model: Multi-Layer Perceptron (MLPClassifier)\n")
    f.write("Hidden Layers: (256, 128, 64)\n")
    f.write("Activation: ReLU\n")
    f.write("Optimizer: Adam\n")
    f.write("Max Iterations: 50\n\n")
    f.write(f"Training Accuracy: {train_accuracy * 100:.2f}%\n")
    f.write(f"Testing Accuracy:  {test_accuracy * 100:.2f}%\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 50 + "\n")
    f.write(report)
    f.write("\n\nSAVED FILES:\n")
    f.write("-" * 50 + "\n")
    f.write("1. sample_images.png    - Sample training images\n")
    f.write("2. training_history.png - Loss curve plot\n")
    f.write("3. confusion_matrix.png - Confusion matrix heatmap\n")
    f.write("4. predictions.png      - Model predictions vs actual\n")
    f.write("5. Output.txt           - This report\n")

print("\n" + "=" * 50)
print("ALL FILES SAVED SUCCESSFULLY!")
print("=" * 50)
