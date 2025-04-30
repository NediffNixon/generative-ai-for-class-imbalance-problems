# Generative Adversarial Networks (GANs) for class imbalance problems in Machine Learning

This project explores a hybrid data augmentation strategy using **SMOTE (Synthetic Minority Over-sampling Technique)** followed by a **Generative Adversarial Network (GAN)** to address severe class imbalance in classification tasks.

The goal is to improve model generalisation on minority classes by leveraging the strengths of both classical oversampling and generative modelling.

> 🏅 This work received **Distinction** as part of my academic submission at the University of Nottingham.

---

## 🧠 Overview

Class imbalance remains a major challenge in many real-world datasets, where some classes are underrepresented, leading to biased models. This project proposes:

1. **Step 1 – SMOTE**: Generate initial synthetic samples using nearest neighbours interpolation.
2. **Step 2 – GAN**: Further enhance the minority class with more realistic, high-variance synthetic samples learned by a GAN model trained only on the minority class.

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/NediffNixon/generative-ai-for-class-imbalance-problems.git
cd generative-ai-for-class-imbalance-problems

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## 📊 Example Scripts
- Example notebook scripts (in "example_notebooks" folder) are available to test the implementation.
- Data correpoinding to the notebook scripts are in "data" folder.

## 📝 Notes
- SMOTE works well when classes are not extremely sparse, but it can create noisy points if not tuned carefully (e.g., number of neighbors).

- The GAN is trained only on the minority class, not the full dataset. This ensures it focuses on capturing that class's structure.

- You may need to balance diversity vs. overfitting when generating samples from the GAN—early stopping and validation is key.

- The synthetic data should always be validated visually or statistically before training.

- This approach is not always better than SMOTE or GANs alone—evaluate based on your dataset.

## 🛠️ Technologies Used
Python

scikit-learn

imbalanced-learn (SMOTE)

TensorFlow (GAN implementation)

## 📄 License
This project is licensed under the MIT License.
