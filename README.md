# Food Vision

- This is a milestone project in Learning Convolutional Neural Networks. It aims at classifying different foods into their specific labels.
- It is based on computer vision concepts.

## Author

- [Dan Njuguna](mailto:njorogedan020j@gmail.com), Data Scientist | Machine Learning Enthusiast.

## Project Scope

- The project utilises TensorFlow module to create an artificial neural network that learns different foods and their labels.
- The dataset is fully images of various foods that has is used to train the neural network.

### Prerequisites

- Good understanding of TensorFlow, and Machine Learning concepts.
- Install TensorFlow.

```bash
pip3 install tensorflow # The complete tensorflow framework
pip3 install tf-nightly # For a lighter version of tensorflow
```

- Download the dataset [Food Vision](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)

- Now, let's role and let the computer see👀!

## Observations

- It is observable that when the weights are kept constant, the accuracy is generally low as observed when we run one epoch with the base model weighst constant.
- Allowing change in the base model weights increases accuracy of the model.
- After, running five epochs for the model; the accuracy steadily impoves which proves that running more epochs for the model will increase its accuracy:

```Python
# fine tune the model with very low learning rate
history = model.fit(
    train_generator, 
    steps_per_epoch = train_generator.n // 32, 
    epochs = 5, 
    callbacks = [checkpointer, earlystopping]
)
```

- Further, assess the accuracy of my model, I evaluate the model, and the accuracy generally improves.

```Python
# Evaluate the performance of the model
evaluate = model.evaluate(
    test_generator, 
    steps = test_generator.n // 32, 
    verbose = 1
)

print('Accuracy Test : {}'.format(evaluate[1]))
```

## Conclusion

- The convolution neural network predicts the food class with relatively good accuracy at **75.32126307487488%**.
