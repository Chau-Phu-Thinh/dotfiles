Classification in machine learning is a predictive modeling process by which machine learning models use classification algorithms to predict the correct label for input data. 

As AI models learn to analyze and classify data in their training datasets, they become more proficient at identifying various data types, discovering trends and making more accurate predictions. 

At the end of the model training process, the model’s performance is evaluated by using test data. After the model performs consistently well, it’s introduced to unseen real-world data. The trained neural networks apply what they learned during training to make successful predictions with new data. 

# What are classification models?
A classification model is a type of machine learning model that sorts data points into predefined groups called classes. Classifiers learn class characteristics from input data, then learn to assign possible classes to new unseen data according to those learned characteristics.
# What are classification algorithms?
A classification algorithm is a categorization-focused [[machine learning algorithm]] that sorts input data into different classes or categories. Artificial intelligence (AI) models use classification algorithms to process input datasets against a specified classifier that sets the criteria for how the data should be sorted. Classification algorithms are widely used in data science for forecasting patterns and predicting outcomes. 
# **How do classification models work**
All follow the same general two-step data classification process:
1. Learning
2. Classification
## Step 1: Learning
Classification has traditionally been a type of [supervised machine learning](https://www.ibm.com/topics/supervised-learning), which means it uses [labeled data](https://www.ibm.com/topics/data-labeling) to train models. In supervised learning, each data point in the training data contains input variables (also known as independent variables or features), and an output variable, or label. 

In classification training, the model’s job is to understand the relationships between features and class labels, then apply those criteria to future datasets. Classification models use each data point’s features along with its class label to decode what features define each class. In mathematical terms, the model considers each data point as a tuple _x_. A tuple is an ordered numerical sequence that is represented as _x = (x1,x2,x3…xn)._

Each value in the tuple is a feature of the data point. By mapping training data with this equation, a model learns which features are associated with each class label. 

The purpose of training is to minimize errors during predictive modeling. [Gradient descent](https://www.ibm.com/topics/gradient-descent) algorithms train models by minimizing the gap between predicted and actual results. Models can later be [fine-tuned](https://www.ibm.com/topics/fine-tuning) with more training to perform more specific tasks. 

[Unsupervised learning](https://www.ibm.com/topics/unsupervised-learning) approaches to classification problems have been a key focus of recent research. Unsupervised learning methods enable models to discover patterns in unlabeled data by themselves. The lack of labels is what differentiates [unsupervised learning and supervised learning](https://www.ibm.com/think/topics/supervised-vs-unsupervised-learning). 

Meanwhile, [semisupervised learning](https://www.ibm.com/topics/semi-supervised-learning) combines labeled and unlabeled data to train models for classification and regression purposes. In situations where obtaining large datasets of labeled data is not feasible, semisupervised learning is a viable alternative.