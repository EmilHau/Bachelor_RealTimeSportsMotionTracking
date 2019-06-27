To execute the test of both models, the following excution order is followed:

1. Run generateTrainingData.py to generate the data augmented dataset
2. Run ballDetectionDataRead.py to read in the dataset
3. Run CNNBallDetectionModel.py to train the model
4. Run ballDetectionFrameByFrame.py to see result. Here you tap 'p' to get next frame.
Every drawn red rectangle in the result frame, is a ball detection.
