Eye Recognition using CNN
This project utilizes Convolutional Neural Networks (CNN) to recognize eyes from images. The model is trained using machine learning techniques to identify and classify eye regions in facial images.

Overview
Eye recognition is a crucial component in many computer vision applications, such as facial recognition systems, driver monitoring systems, and security systems. By leveraging deep learning techniques like CNNs, this project aims to build an efficient and accurate eye recognition model.

Installation
Clone the repository:

bash
git clone https://github.com/Hanjiho0316/cnn.git
Navigate to the project directory:


Train the CNN model using the following command:

bash
python train_model.py
Once the model is trained, use the following command to test eye recognition on new images:

bash
python recognize_eyes.py --image <path_to_image>
Features
Eye Detection: Detects and locates eyes in images.
CNN Model: Built using Convolutional Neural Networks for robust feature extraction.
Real-time Testing: Use trained models to recognize eyes in new images.
Requirements
Python 3.x
TensorFlow or PyTorch (depending on which framework you use)
OpenCV (for image processing)

in the folder the npy file is acting like dataset
npy file is just array
