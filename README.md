# Food Vision

- This is a milestone project in Learning Convolutional Neural Networks. It aims at classifying different foods into their specific labels.
- It is based on computer vision concepts.

## Author

- [Dan Njuguna](mailto:njorogedan020j@gmail.com), Data Scientist | Machine Learning Enthusiast.

## Project Scope

- In this project, we will make use of **Transfer Learning**, where we will essentially use pretrained models on this dataset from similar already solved problems.


### Why Transfer Learning?

- It will reduce training time for our model.
- Improve on performance since pretrained models learn powerful features from vast datasets which can significantly improve the performance of the model.
- **ImageNet** model will be used to perform the training of our food classification project.

## Workflow

1. Examine and understand data: Data Exploration
2. Build an input pipeline: To load data directly from local disk.
3. Build the model: model creation.
4. Train the model: Training the network.
5. Test the model: To check if the model works without overfitting or underfitting or error.
6. Improve the model and repeatedly redo all the steps.

### Prerequisites

- Good understanding of TensorFlow, and Machine Learning concepts.
- Install TensorFlow:

```bash
pip3 install tensorflow # The complete tensorflow framework
pip3 install tf-nightly # For a lighter version of tensorflow
```

- Download the dataset [Food Vision](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)
- Now, let's role and let the computer see👀!

## Observations

- It is observable that when the weights are kept constant, the accuracy is generally low as observed when we run one epoch with the base model weight constant.
- Allowing change in the base model weights increases accuracy of the model.
- After, running ten epochs for the model; the accuracy steadily impoves which proves that running more epochs for the model will increase its accuracy:

```Python
# fine tune the model with very low learning rate
history = model.fit(
    train_generator, 
    steps_per_epoch = train_generator.n // 32, 
    epochs = 10, 
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

print(f'Accuracy Test : {evaluate[1]}')
```

## Conclusion

- The convolution neural network, with transfer learning, predicts the test data food class with relatively high accuracy at **84.22%**.
- Saving the best accuracy for each training iteration increased the accuracy of the model.
- Changing the weights of the network as it learns improves its accuracy.
- The comparatively low learning rate, **0.01**, improves accuracy in the model.
