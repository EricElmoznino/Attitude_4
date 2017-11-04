# Attitude_4
Phase 4 of developing an AI that can determine attitude (orientation and position) from a sequence of images.

File Descriptions:
- Attitude_4.py:
Contains the principal class for model instantiation, training, and inference. All the session logic is in here. Training images must be music in .jpg format and labels in .txt. The model takes two images as input (reference and current) and passes them together through a deep convolutional net, predicting the pitch, yaw, roll, and position of the current image with regards to the reference one.
- Helpers.py:
Convenience functions and classes.
- main.py:
An example of training the model and predicting some attitude's with it.
