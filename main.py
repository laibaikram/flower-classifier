import requests
import tensorflow as tf
from PIL import Image

def classify_flowers(image_path):
    # Load the pre-trained InceptionV3 model
    model = tf.keras.applications.InceptionV3()
    
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((299, 299))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    
    # Perform flower classification
    predictions = model.predict(image)
    
    # Decode the predictions
    decode_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=5)
    
    # Return the flower labels and probabilities
    labels = [label for (_, label, _) in decode_predictions[0]]
    probabilities = [probability for (_, _, probability) in decode_predictions[0]]
    
    return labels, probabilities

def main():
    # Specify the path to the image
    image_path = 'flower.jpg'
    
    # Perform flower classification
    labels, probabilities = classify_flowers(image_path)
    
    # Print the results
    print("Flower Classification Results:")
    for label, probability in zip(labels, probabilities):
        print(f"Label: {label}, Probability: {probability}")

if __name__ == '__main__':
    main()
