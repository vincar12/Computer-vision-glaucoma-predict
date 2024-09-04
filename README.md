# Glaucoma Prediction Computer Vision

### 1. Project Overview
The goal of this project is to create a Computer Vision Model with Artificial Neural Networks(ANN) to predict whether an image has symptoms of glaucoma. The model aims to catch symptoms of glaucoma with a great accuracy.  
Deployment: [Huggingface](https://huggingface.co/spaces/vincar12/glaucoma-predict)

---

### 2. Project Structure
```bash
├── deployment/                     # Contains deployment files
├── P2G7_Vincar.ipynb               # Jupyter notebook for analysis and model creation
├── P2G7_Vincar_Inference.ipynb     # Jupyter notebook for model inference
├── README.md                       # Project README file
```

---

### 3. Workflow
1. **Data Loading & Cleaning**:
   - Load the original data from CSV and clean column names.

2. **Exploratory Data Analysis (EDA)**:
   - Explore the dataset to better understand the data.

3. **Preprocessing**: Preprocessing the images from the dataset in order for the model to be able to process and learn the data.
   - **Resizing**: Setting the image height and width so the model can learn on a smaller size which helps in computing time.
   - **Augmentation & Scaling**: Generating images that are altered such as rotated, zoomed, flipped, etc. to artificially increase the data learned by the model so that it has a much more varied data to learn. Rescaling is also essential so that the image can be processed by the model.

4. **ANN Training**:
   - This section will have training and evaluation of the Artificial Neural Network model on the dataset that has been preprocessed.

5. **ANN Improvement**:
   - Improving the model by modifying the neural networks to produce better performance.

6. **Model Saving**:
   - Saving the model by freezing the trainable parameters so the model does not keep learning on data for inferencing.

7. **Conclusion & Recommendations**:
   - Provide conclusions from the project.

8. **Inference**:
   - Inference is performed in a separate notebook to reduce the risk of data leakage.

---

### 4. Results and Conclusion
- Based on the dataset it can be said that the data is quite balanced with all classes represented relatively equal.
- Classification of the dataset is divided into 4 classes which is then used for training by the model.
- Model has a moderately good accuracy of 79% in predicting which class of glaucoma an image is.
- The model used for saving is the improved model_sequential_gamma which demonstrates a much higher score and a more stable metrics compared to the first model which exhibited signs of overfitting.
- Possible improvements are: experimenting on neurons, batch size, regularizer, and learning rate.

---

### 5. References
Dataset: [Glaucoma Detection](https://universe.roboflow.com/isp005/glaucoma-detection-sttfw/dataset/1)

Tools: Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn, TensorFlow, Keras, Streamlit.
