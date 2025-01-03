# **Implementing Privacy-Preserving Techniques in Machine Learning Using Differential Privacy**

A Python-based project demonstrating the application of **Differential Privacy (DP)** in machine learning models to ensure robust data protection while maintaining predictive performance. This project investigates trade-offs between privacy and utility using the **Iris dataset** as a case study, employing TensorFlow Privacy for implementing differential privacy in logistic regression.

---

## **Abstract**
This project explores differential privacy to address growing concerns around data privacy in machine learning. The study compares baseline and DP-enhanced logistic regression models under varying privacy parameters and evaluates their resilience against membership inference attacks. Key findings highlight the delicate balance between privacy and model utility.

---

## **Features**
- **Privacy-Preserving Machine Learning**: Differential privacy applied to logistic regression using DP-SGD.
- **Trade-Off Analysis**: Detailed evaluation of privacy-utility trade-offs across various noise multiplier settings.
- **Membership Inference Attack Defense**: Robust testing against privacy attacks to validate DP efficacy.
- **Scalable Implementation**: Efficient processing with minimal computational overhead.

---

## **Requirements**
1. **Python 3.8 or later**  
2. **Required Libraries**:
   ```bash
   pip install -r requirements.txt

## **Usage**

### **Launch the Notebook**

1. Run the Jupyter Notebook to start the workflow:  
    ```bash
   jupyter notebook "Differential_Privacy_Iris.ipynb"  
3. Preprocess the dataset and prepare TensorFlow datasets:  
   ```bash
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    
    def create_tf_dataset(X, y, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.shuffle(len(X)).batch(batch_size)
    
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=32)
    test_dataset = create_tf_dataset(X_test, y_test, batch_size=32)


3. Train baseline and differentially private models:  
   1. Use the baseline logistic regression model.  
   2. Enable DP-SGD with configurable privacy parameters.  
4. Evaluate privacy-utility trade-offs and test for membership inference attack resistance.

## **Dependencies**

* Python 3.8+  
* TensorFlow  
* TensorFlow Privacy  
* Scikit-learn

## **Implementation Details**

* **Dataset:** Iris dataset transformed into a binary classification task.  
* **Preprocessing:** StandardScaler for feature normalization.  
* **Model Architecture:**  
  * Baseline: Logistic regression without privacy guarantees.  
  * Differentially Private: Logistic regression using DP-SGD.  
* **Privacy Mechanisms:**  
  * Gradient clipping and Gaussian noise addition.  
  * Configurable privacy parameters: noise multiplier, L2 norm clipping, and microbatching.  
* **Evaluation Metrics:**  
  * Model accuracy, privacy budget (ε), membership inference attack resistance, training time.

## **Results**

* **Model Performance:** Near-baseline accuracy achieved with low noise levels; higher noise strengthens privacy at the cost of accuracy.  
* **Privacy-Utility Trade-offs:**  
  * Optimal balance at ε \= 11.25–22.19, achieving \~84%-92% accuracy.  
  * Strong privacy (ε \= 7.46) leads to 76% accuracy.  
* **Membership Inference Resistance:** Attack success rate reduces as noise increases, approaching random guessing with higher privacy guarantees.  
* **Computational Overhead:** \~53% increase in training time with negligible memory overhead.

## **Contributors**

* Team Members: Stephen, Steve, Linson, Shreyas  
* Course: Advanced Computer Security

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.
