# MNIST Handwritten Digits and a Modern LeNet-5

This project implements the LeNet-5 architecture from scratch in Python, using only `numpy`.
This version of LeNet-5 has the same basic structure as the original, but some details were changed to reflect more modern sensibilities (e.g., ReLU activation).
This modern version manages to crack 99.3% accuracy against the [MNIST dataset](http://yann.lecun.com/exdb/mnist/); the original version reported an accuracy of 99.05%.

In the [Jupyter notebook](MNIST%20Handwritten%20Digits%20and%20a%20Modern%20LeNet-5.ipynb), we also try some classical ML models and a few simpler deep learning architectures before using LeNet-5.

This project also contains code to load the MNIST dataset, do some basic preprocessing, and score predictions.
It also provides some useful visualizations to show the overall performance of a model and sample both correct and incorrect predictions.

In the real world, a deep learning framework such as Keras would be used instead.
However, some experts recommend implementing deep learning yourself to understand how it works&mdash;and how frameworks like Keras work.
