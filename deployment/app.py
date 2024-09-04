import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import load_model

def run():
    #title
    st.title('Glaucoma Prediction')
    #line
    st.markdown('---')
    #membuat subheader
    st.subheader('Model deployment to predict glaucoma based on uploaded image')

    #uploader
    upl = st.file_uploader('Upload image', type=['jpeg', 'jpg', 'png', 'webp'])

    #load model
    model = load_model('model_sequential.keras')

    #function to load & preprocess
    def import_and_predict(image_data, model):
        image = load_img(image_data, target_size=(160, 160))
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  
        img_array = img_array / 255.0  

        #predict image
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)
        class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        predicted_class_label = class_labels[predicted_class_index[0]]

        #display result
        result = f'Prediction: {predicted_class_label}'
        proba = f'Prediction Probabilities: {prediction[0]}'

        return result, proba

    # if no upload
    if upl is None:
        st.text('Please upload an image file')
    else:
        result, proba = import_and_predict(upl, model)
        st.image(upl)
        st.write(result)
        st.write(proba)

if __name__ == '__main__':
    run()