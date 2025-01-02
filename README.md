README for Brain Tumor Classification Project

Overview

This project aims to classify brain tumor images using both traditional machine learning (ML) algorithms and deep learning (DL) techniques. We will use a variety of features extracted from medical images, such as mean, variance, entropy, skewness, and contrast, along with convolutional neural networks (CNNs) for classification tasks. The dataset includes images with tumor and non-tumor classes, and we will explore models such as Logistic Regression, Decision Trees, and CNNs to predict the presence of a brain tumor.

Technologies Used
	•	Python: Programming language for data analysis and machine learning
	•	TensorFlow: Deep learning framework for CNN models and transfer learning
	•	scikit-learn: For traditional machine learning algorithms like Logistic Regression and Decision Trees
	•	pandas: Data manipulation and analysis
	•	NumPy: Numerical operations
	•	matplotlib: For visualizing data and model performance
	•	prettytable: To format evaluation results

Requirements

The following Python libraries are used in this project:
# Install dependencies
pip install -r requirements.txt
Dataset

The dataset consists of brain tumor images and corresponding metadata, including first and second order statistical features. The features in the dataset include:
	•	Mean, Variance, Standard Deviation: Measures of central tendency and spread.
	•	Entropy, Skewness, Kurtosis: Measures of image randomness and shape.
	•	Contrast, Energy, ASM, Homogeneity: Texture features that quantify local image variations.

The images are labeled as Tumor or Non Tumor.

Data Preprocessing

For Machine Learning (ML):
	•	Feature Extraction: Statistical features such as mean, variance, skewness, and entropy are extracted from the images and stored in the CSV file (Brain Tumor.csv).
	•	Data Normalization: The features are standardized using StandardScaler to bring all features to the same scale.
	•	Splitting the Dataset: The data is split into training and testing sets using an 80-20 split.

For Deep Learning (DL):
	•	Image Preprocessing: Images are loaded and resized to a consistent size of 240x240 pixels for CNN models. Pixel values are normalized by scaling them to the range [0, 1].
	•	Data Augmentation: For the training dataset, images are augmented (flipped, rotated, zoomed) to improve model robustness.

Models

1. Traditional Machine Learning Models
	•	Logistic Regression: A linear classifier used for predicting the presence of a tumor.
	•	Decision Tree: A non-linear classifier that builds a decision tree based on the features.
	•	Random Forest: An ensemble of decision trees used to improve classification accuracy.

2. Deep Learning Models
	•	CNN from Scratch: A custom CNN model is built using multiple convolutional and pooling layers, followed by dense layers for classification.
	•	Transfer Learning: A pre-trained ResNet50 model is used as the base model, and additional layers are added for classification. Only the last few layers of ResNet50 are fine-tuned.
	•	Ensemble Model: A combination of predictions from machine learning and deep learning models (Logistic Regression, Decision Tree, Random Forest, and CNN).

Model Evaluation

Each model is evaluated using various metrics such as:
	•	Accuracy: The percentage of correct predictions.
	•	Precision: The percentage of correctly predicted tumors out of all predicted tumors.
	•	Recall: The percentage of correctly predicted tumors out of all actual tumors.
	•	F1 Score: The harmonic mean of precision and recall.
	•	Sensitivity: The true positive rate, i.e., how many actual tumors were correctly identified.
	•	AUC-ROC: The area under the receiver operating characteristic curve, a measure of the model’s ability to distinguish between classes.
Example Output:

For the Logistic Regression Model:
![Accuracy Output](https://github.com/vijay1612/BrainTumour/blob/main/IMG_D745AB07BD19-1.jpeg)
Training and Testing

For CNN models:
	•	CNN from Scratch:
	•	Accuracy: 91.0%
	•	Precision: 89.4%
	•	Recall: 90.5%
	•	F1 Score: 90.0%
	•	ResNet Transfer Learning:
	•	Accuracy: 85.5%
	•	Precision: 80.5%
	•	Recall: 89.3%
	•	F1 Score: 84.7%
	•	ML + DL Ensemble:
	•	Accuracy: 94.7%
	•	Precision: 93.5%
	•	Recall: 94.7%
	•	F1 Score: 94.1%
 Conclusion

This project demonstrates the use of both traditional machine learning and deep learning techniques for classifying brain tumor images. We used a combination of image-based features and CNN architectures to classify the data, achieving impressive results with both approaches. The ensemble model combining machine learning and deep learning models provided the best performance.
