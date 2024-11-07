# Face Detection & Recognition using CNN and Transfer Learning

## Overview
This project involves developing a face recognition system using deep learning and transfer learning techniques. The system performs face detection and recognition on the Pins Face Recognition dataset from Kaggle. The main steps include dataset preprocessing, face detection, face cropping, model training, validation, and live recognition using a webcam.

## Steps

### 1. Dataset Preprocessing
- **Download Dataset**: Download the dataset from Kaggle using the following link: [Pins Face Recognition Dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition). You will need to have a Kaggle account and the Kaggle CLI set up to download the dataset programmatically. Ensure that the dataset is downloaded to the appropriate directory for seamless integration with the preprocessing steps.

- **Extract Dataset**: Extract the dataset into a directory called `data`. This directory will contain subdirectories, each representing a different individual with images of that person. Proper organization is crucial to facilitate training, as each subdirectory represents a unique label or class that the model will learn to recognize.

- **Prepare Cropped Faces Directory**: Create a directory to store cropped faces for each class. This directory will have the same structure as the original dataset, with folders for each individual. The purpose of creating a cropped faces directory is to ensure that the training data is standardized and only contains relevant facial information, which significantly improves recognition accuracy.

### 2. Face Detection and Cropping
- **Load OpenCV Pre-trained DNN Model**: Download the OpenCV DNN model and architecture files. These files contain the weights and the configuration needed to detect faces in images.
  - **Model Weights**: The weights file (`opencv_face_detector_uint8.pb`) contains the pre-trained parameters of the model. This model has been trained on a large dataset of facial images to detect faces with high accuracy.
  - **Model Architecture**: The architecture file (`opencv_face_detector.pbtxt`) defines the structure of the model used for face detection, including the layers and how they are connected.

- **Face Cropping**: For each image in the dataset, use the OpenCV DNN model to detect faces. Crop the detected faces and save them in the output directory with the same class structure. This step helps in focusing the model on just the facial features, which improves recognition performance. Cropping also removes any unnecessary background noise that could negatively impact the model's ability to learn distinguishing features.

### 3. Data Augmentation and Splitting
- **Data Augmentation**: Perform data augmentation to increase the variety of training images. Techniques such as rotation, flipping, zooming, and shifting are used to make the model more robust to variations in facial orientation, lighting, and expression. Data augmentation helps simulate different real-world conditions, ensuring that the model can generalize well to unseen data.

- **Data Splitting**: Split the augmented dataset into training and validation sets. Typically, 80% of the data is used for training, and 20% is used for validation. This split helps in evaluating the model's performance on unseen data and provides a benchmark to prevent overfitting. Proper splitting ensures that the model does not simply memorize the training data but learns generalizable patterns.

### 4. Transfer Learning
- **Load Pre-trained Model**: Use a pre-trained model, such as VGG16, for feature extraction. Transfer learning allows us to leverage the knowledge from models trained on large datasets (like ImageNet) to extract relevant facial features, reducing the need for extensive training data. This approach significantly speeds up training and improves performance, especially when working with limited data.

- **Add Custom Layers**: Add custom fully connected layers on top of the pre-trained model to adapt it for the specific face recognition task. These layers are trained specifically for the face recognition dataset, allowing the model to learn the unique features of each individual in the dataset.

- **Compile the Model**: Compile the model using an appropriate loss function, such as categorical cross-entropy, and an optimizer like Adam. This step defines how the model will be updated during training to minimize the error. The loss function measures the difference between the predicted and actual labels, while the optimizer adjusts the model weights to reduce this difference.

### 5. Model Training and Validation
- **Train the Model**: Train the model using the training dataset. Monitor the training and validation accuracy to ensure the model is learning effectively and not overfitting. Training involves adjusting the weights of the model to minimize the error between predictions and actual labels. During training, it is important to keep track of metrics such as accuracy and loss to evaluate how well the model is learning.

<img width="321" alt="image" src="https://github.com/user-attachments/assets/d850dd74-a016-442f-b8ef-672cfb220b78">

<img width="329" alt="image" src="https://github.com/user-attachments/assets/2ba58722-fe4a-4bc5-9a65-be3a0fd8939b">

### 6. Live Face Recognition using Webcam
- **Real-time Recognition**: Implement a system that uses a live webcam feed to detect and recognize faces in real-time. The model first detects faces in each frame, then predicts the identity of each detected face. This requires efficient processing to achieve near real-time performance, as delays in recognition can affect user experience.

- **Face Detection and Recognition Flow**: Each frame from the webcam is processed through the face detector, and the detected faces are passed through the trained model to predict their class. The results are displayed on the video feed, with bounding boxes and labels indicating the recognized individuals. This requires the model to be optimized for speed, ensuring that the recognition process can handle continuous input from the webcam.

### 7. Testing and Fine-Tuning
- **Testing**: Evaluate the model on different inputs, particularly those that were not part of the training or validation sets. This step is crucial for verifying the model's generalizability and robustness to new data. Testing helps identify any biases in the model and ensures that it can accurately recognize faces in different environments and conditions.

- **Fine-Tuning**: Adjust the model's hyperparameters, such as the learning rate, batch size, and the number of dense layers, to optimize its accuracy. Fine-tuning may also involve training the model for a few additional epochs with a lower learning rate to further refine the learned features. This step is essential for squeezing out the last bit of performance from the model, ensuring that it is as accurate as possible.

### 8. Save the Trained Model
- **Save Model**: Save the trained model to a file for future use. This allows you to load the model later without having to retrain it, making it easier to deploy the model for real-time recognition. Saving the model also allows you to share it with others or use it for additional projects without having to go through the entire training process again.

### 9. Running the Model
- **Run Real-Time Recognition**: Use the saved model to perform real-time face recognition with a webcam. This requires setting up the environment with a compatible webcam and ensuring that all dependencies are properly installed. Running the model in real-time allows you to see the system in action, recognizing faces as they appear in front of the camera.

## Dependencies
- **Python 3.x**: The programming language used for implementing the system. Python provides a wide range of libraries and frameworks that are well-suited for deep learning and computer vision tasks.
- **TensorFlow**: The deep learning framework used for building and training the model. TensorFlow provides powerful tools for creating neural networks and optimizing their performance.
- **OpenCV**: A library used for computer vision tasks, including face detection and video capture. OpenCV is widely used for real-time image processing and is crucial for implementing the face detection component of the system.
- **NumPy**: A library used for numerical operations, particularly for handling image arrays. NumPy provides efficient data structures for storing and manipulating images, which is essential for preprocessing and feature extraction.
- **Matplotlib**: A library used for visualizing training history, including accuracy and loss curves. Visualizing the training process helps in understanding how well the model is learning and identifying any issues with overfitting or underfitting.
- **Kaggle CLI**: A command-line tool to download datasets directly from Kaggle. The Kaggle CLI simplifies the process of downloading large datasets and integrating them into your workflow.

## Instructions
- **Install Dependencies**: Ensure all necessary dependencies are installed before running the scripts. This includes TensorFlow, OpenCV, and other Python packages. Use a virtual environment to manage dependencies and avoid conflicts between different projects.
- **Download and Extract Dataset**: Download the dataset from Kaggle and extract it to the appropriate directory. Ensure that the dataset is organized properly to facilitate seamless training and evaluation.
- **Prepare Cropped Faces**: Run the preprocessing script to crop faces from the original images and organize them into the cropped faces directory. This step is crucial for focusing the training data on relevant features, improving the accuracy of the model.
- **Train the Model**: Train the model using the provided training and validation data. Monitor the training process to ensure the model is learning effectively. Use tools like TensorBoard to visualize training metrics and identify any issues that may arise.
- **Run Real-Time Recognition**: Use the webcam setup to run real-time face recognition. Make sure the webcam is properly connected and supported by OpenCV. Test the system in different lighting conditions to ensure that it performs well in a variety of environments.

## Notes
- **Webcam Requirement**: Real-time recognition requires a webcam connected to the local system. Ensure drivers are installed and compatible with OpenCV. If running on a laptop, the built-in webcam should work, but external webcams may require additional configuration.
- **Kaggle Environment**: Kaggle notebooks do not support webcam-based live recognition due to the cloud environment's limitations. Use a local Python environment for live recognition. Kaggle is best suited for training and validation, while real-time recognition should be performed locally.
- **Model Performance**: The accuracy of the recognition system depends on the quality and diversity of the training dataset. Better results can be achieved by using high-quality images with diverse lighting and facial expressions. It is also important to include a variety of angles and expressions to ensure the model can recognize individuals under different conditions.
- **Model Limitations**: The model's performance may be limited by factors such as occlusions (e.g., hats or glasses) or extreme variations in lighting. Addressing these limitations may require additional data collection or model adjustments.
- **Ethical Considerations**: Face recognition systems can have significant privacy implications. Ensure that the model is used ethically and that individuals are aware of how their data is being used. Obtain consent before collecting or using personal data for training or recognition purposes.



