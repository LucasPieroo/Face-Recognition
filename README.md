# 🏆 Face Recognition Competition - IFSP Campinas (Submission: Lucas Piero, CP3016315)  
**Kaggle Competition Link**: [IFSP D3APL 2023 Face Recognition](https://www.kaggle.com/competitions/ifsp-d3apl-2023-face-recognition/overview)  

---

## 📖 Introduction  
This project presents our approach to the **Face Recognition Competition** organized by the Federal Institute of São Paulo (IFSP) - Campinas Campus, as part of the Applied Data Science course (D3APL 2023). The goal is to build a neural network model capable of accurately identifying faces of 83 individuals using the PubFig83 dataset.  

The final solution leverages advanced deep learning techniques, including transfer learning, data augmentation, and model optimization.  
- **Exploratory notebooks**:  
  - `Brainstorm.ipynb`: Iterative testing of architectures and preprocessing strategies.  


---

## 🎯 Objective  
Develop a **machine learning model** (primarily neural networks) to classify faces into 83 distinct categories with high accuracy, laying the groundwork for future real-time recognition systems.  

---

## 🗄️ Dataset  
**PubFig83 Dataset**:  
- **13,840 images** of 83 celebrities, resized to **100x100 pixels** and eye-aligned.  
- **Training Data**: 12,180 labeled images (`train.csv`).  
- **Test Data**: 1,660 unlabeled images (`test.csv`) for competition evaluation.  

---

## 🛠️ Methodology  

### 1. **Data Preprocessing**  
- **Resizing & Normalization**: Images standardized to 100x100 pixels, normalized to `[0, 1]`.  
- **Data Augmentation**: Techniques included rotation (±30°), horizontal flipping, zoom (20%), and shifts (10%) to reduce overfitting.  

### 2. **Model Selection & Training**  
#### **Initial Approach: Custom CNN**  
- Achieved **46% validation accuracy**, highlighting limitations in feature extraction for complex facial patterns.  

#### **Transfer Learning Models**  
Tested multiple architectures for comparison:  
| Model          | Validation Accuracy |  
|----------------|---------------------|  
| **VGGFace**    | **97.7%**           |  
| **DeepFace**   | **95.8%**           |  
| Xception       | ~85%                |  
| DenseNet201    | ~82%                |  
| MobileNetV2    | ~75%                |  

**Why VGGFace/DeepFace?**  
- Pre-trained on large face datasets, enabling robust feature extraction.  
- Fine-tuning focused on adapting to the PubFig83 distribution.  

#### **Final Submission**  
- **VGGFace** achieved **98.62% accuracy** on the competition test set.  

---

## 📊 Results & Insights  


**Key Takeaways**:  
1. **Transfer Learning Dominance**: Pre-trained models (VGGFace, DeepFace) outperformed custom CNNs by **~50%**, emphasizing the value of leveraging existing knowledge.  
2. **VGGFace Superiority**: Achieved the highest accuracy (97.7% validation, 98.6% test) due to its architecture optimized for facial features.  
3. **Overfitting Mitigation**: Data augmentation and dropout layers were critical for generalization.  

---

## 🚀 Next Steps  
1. **Real-Time Recognition**:  
   - Deploy the model in a real-time application using OpenCV for live face detection.  
   - Optimize inference speed with TensorRT or ONNX.  
2. **Data Expansion**:  
   - Generate synthetic data (e.g., GANs) to improve robustness.  
   - Test with personal photos to evaluate real-world performance.  
3. **Hyperparameter Tuning**:  
   - Experiment with learning rate schedules and advanced optimizers (e.g., AdamW).  
4. **Model Interpretability**:  
   - Use Grad-CAM to visualize critical facial regions for predictions.  

---

## 🛠️ Technologies Used  
- **Frameworks**: TensorFlow, Keras, OpenCV.  
- **Pre-trained Models**: VGGFace, DeepFace, Xception, DenseNet, MobileNet.  
- **Optimization**: Data augmentation, dropout, batch normalization.  
- **Hardware**: GPU acceleration (NVIDIA CUDA).  
