import numpy as np
from datasets import get_mnist
from network import *

def one_hot_encode(scalar, num_classes):
    return np.eye(num_classes)[scalar]


class MNIST_Model(BaseModel):
    def __init__(self):
        hidden_units = 256
        self.layers = [
            DenseLayer(28 * 28, hidden_units),
            GeLU(),
            DenseLayer(hidden_units, hidden_units),
            GeLU(),
            DenseLayer(hidden_units, hidden_units),
            GeLU(),
            DenseLayer(hidden_units, hidden_units),
            GeLU(),
            DenseLayer(hidden_units, 10),
        ]


if __name__ == '__main__':
    model = MNIST_Model()


    X_train, y_train, X_test, y_test = get_mnist()

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)


    train_data = Dataset(X_train, y_train)
    test_data = Dataset(X_test, y_test, shuffle=False)

    # Define the loss and optimization functions
    loss_fn = CategoricalCrossentropy()
    optimizer = Adam(lr=0.0005)

    # Train the model for a specified number of epochs
    num_epochs = 20
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_data:
            # Forward pass
            y_pred = model.forward(X_batch)
            loss = np.mean(loss_fn.forward(y_pred, y_batch))
            
            # Backward pass
            grad = loss_fn.backward(y_pred, y_batch)
            model.backward(grad)
            
            # Update the weights
            optimizer.step(model.layers)
            
        # Print the loss every 10 epochs
        if (epoch + 1) % 1 == 0:
            y_test_pred = model.forward(X_test)
            test_loss = np.mean(loss_fn.forward(y_test_pred, y_test))
            test_accuracy = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_test_pred, axis=-1))

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Evaluate the model on the test data
    y_test_pred = model.forward(X_test)
    test_loss = np.mean(loss_fn.forward(y_test_pred, y_test))
    print(f'Test Loss: {test_loss}')

    # Compute the accuracy
    test_accuracy = np.mean(np.argmax(y_test, axis=-1) == np.argmax(y_test_pred, axis=-1))
    print(f'Test Accuracy: {test_accuracy}')

    model.save(f"MNIST_Model_Acc_{round(test_accuracy*100, 2)}")