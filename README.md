# Flower Classification

The Flower Classification project is a Python application that uses a pre-trained machine learning model to classify the flowers present in an image. It utilizes the TensorFlow library and the InceptionV3 model for image classification.

## Prerequisites

Before running the Flower Classification project, ensure that you have the following installed:

- Python 3 (https://www.python.org/downloads/)
- TensorFlow library (install using `pip install tensorflow`)
- Requests library (install using `pip install requests`)
- PIL library (install using `pip install pillow`)

## Getting Started

1. Clone the repository or download the source code.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the following command to install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have an image that you want to classify. The image should be in a common format like JPEG or PNG.

## Usage

1. Place your image in the project directory.

2. Open the `main.py` file in a text editor.

3. Specify the path to your image by replacing `'flower.jpg'` with the path to your own image in the `main` function.

4. Save the `main.py` file.

5. Run the following command to start the classification:

   ```bash
   python main.py
   ```

6. The application will load the pre-trained InceptionV3 model and classify the flowers in the image.

7. The classification results, including the predicted flower labels and their corresponding probabilities, will be displayed in the console.

## Customization

- You can modify the code in `main.py` to incorporate your own image classification model or utilize different pre-trained models available in TensorFlow.

- The project uses the InceptionV3 model for flower classification. If you want to classify objects other than flowers, you may need to use a different model trained on a different dataset.
