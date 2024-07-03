# MLP Implementation on Iris Dataset with Python and VHDL

This project involves implementing a Multi-Layer Perceptron (MLP) for the Iris dataset using both Python and VHDL. The Iris dataset contains 150 samples, each with four features (sepal length, sepal width, petal length, and petal width), and is used to classify samples into one of three species.

## Project Description

### Python Implementation

1. **Data Preparation**:
    - The dataset was split into training and testing sets.
    - Various network configurations were tested to find the optimal architecture:
        - (3, 4, 4)
        - (3, 3, 3, 4)
        - (3, 5, 4)

2. **Training and Evaluation**:
    - Libraries used: Pytorch/Keras/TensorFlow/Scikit-learn.
    - Activation function: ReLU.
    - Optimization algorithm: Stochastic Gradient Descent (SGD).
    - The model was trained for 1000 epochs with a learning rate of 0.01.
    - Accuracy achieved:
        - Training set: approximately 99%
        - Test set: 100%

### VHDL Implementation

1. **Architecture Design**:
    - The architecture of the MLP was designed in VHDL.
    - Forward propagation and backward propagation were implemented to compute the weighted sums, activations, and gradients.
    - The VHDL implementation used floating-point numbers and is not synthesizable.

2. **Forward Propagation**:
    - Compute the weighted sum for each neuron in the hidden layer 1:
        ```text
        z11 = w11 * x1 + w21 * x2 + w31 * x3 + w41 * x4 + b1
        z12 = w12 * x1 + w22 * x2 + w32 * x3 + w42 * x4 + b2
        z13 = w13 * x1 + w23 * x2 + w33 * x3 + w43 * x4 + b3
        z14 = w14 * x1 + w24 * x2 + w34 * x3 + w44 * x4 + b4
        ```
    - Apply ReLU activation function:
        ```text
        h11 = ReLU(z11)
        h12 = ReLU(z12)
        h13 = ReLU(z13)
        h14 = ReLU(z14)
        ```
    - Compute the weighted sum for each neuron in the hidden layer 2:
        ```text
        z21 = w51 * h11 + w61 * h12 + w71 * h13 + w81 * h14 + b5
        z22 = w52 * h11 + w62 * h12 + w72 * h13 + w82 * h14 + b6
        z23 = w53 * h11 + w63 * h12 + w73 * h13 + w83 * h14 + b7
        z24 = w54 * h11 + w64 * h12 + w74 * h13 + w84 * h14 + b8
        ```
    - Apply ReLU activation function:
        ```text
        h21 = ReLU(z21)
        h22 = ReLU(z22)
        h23 = ReLU(z23)
        h24 = ReLU(z24)
        ```
    - Compute the weighted sum for each neuron in the output layer:
        ```text
        z31 = w91 * h21 + w101 * h22 + w111 * h23 + w121 * h24 + b9
        z32 = w92 * h21 + w102 * h22 + w112 * h23 + w122 * h24 + b10
        z33 = w93 * h21 + w103 * h22 + w113 * h23 + w123 * h24 + b11
        ```
    - Apply ReLU activation function:
        ```text
        y1 = ReLU(z31)
        y2 = ReLU(z32)
        y3 = ReLU(z33)
        ```

3. **Backward Propagation**:
    - Compute the gradients for the output layer:
        ```text
        output_error = (y1 - t1, y2 - t2, y3 - t3)
        output_delta = hidden_layer_2.T * output_error
        output_bias_delta = sum(output_error)
        ```
    - Compute the error for the hidden layer 2 neurons:
        ```text
        hidden_error_2 = (output_error * w51) * (h21 > 0) + (output_error * w52) * (h22 > 0) + (output_error * w53) * (h23 > 0) + (output_error * w54) * (h24 > 0)
        ```
    - Compute the gradients for the hidden layer 2 weights:
        ```text
        hidden_delta_2 = hidden_layer_1.T * hidden_error_2
        hidden_bias_delta_2 = sum(hidden_error_2)
        ```
    - Compute the error for the hidden layer 1 neurons:
        ```text
        hidden_error_1 = ((hidden_error_2 * w11) * (h11 > 0) + (hidden_error_2 * w12) * (h12 > 0) + (hidden_error_2 * w13) * (h13 > 0) + (hidden_error_2 * w14) * (h14 > 0)) * (w21 > 0) + ((hidden_error_2 * w21) * (h11 > 0) + (hidden_error_2 * w22) * (h12 > 0) + (hidden_error_2 * w23) * (h13 > 0) + (hidden_error_2 * w24) * (h14 > 0)) * (w22 > 0)
        ```
    - Compute the gradients for the hidden layer 1 weights:
        ```text
        hidden_delta_1 = X.T * hidden_error_1
        hidden_bias_delta_1 = sum(hidden_error_1)
        ```

4. **Update Weights and Biases**:
    - Update the output layer weights and biases:
        ```text
        w91 -= learning_rate * output_delta[0]
        w92 -= learning_rate * output_delta[1]
        w93 -= learning_rate * output_delta[2]
        b9 -= learning_rate * output_bias_delta
        ```
    - Update the hidden layer 2 weights and biases:
        ```text
        w51 -= learning_rate * hidden_delta_2[0]
        w52 -= learning_rate * hidden_delta_2[1]
        w53 -= learning_rate * hidden_delta_2[2]
        b5 -= learning_rate * hidden_bias_delta_2
        ```
    - Update the hidden layer 1 weights and biases:
        ```text
        w11 -= learning_rate * hidden_delta_1[0]
        w12 -= learning_rate * hidden_delta_1[1]
        w13 -= learning_rate * hidden_delta_1[2]
        b1 -= learning_rate * hidden_bias_delta_1
        ```

### Results

- The Python implementation achieved:
    - Training set accuracy: approximately 99%
    - Test set accuracy: 100%
- The VHDL implementation achieved an accuracy of approximately 66%.


## Conclusion

This project demonstrates the implementation of a Multi-Layer Perceptron for the Iris dataset using both Python and VHDL. The Python implementation provides a high accuracy baseline, while the VHDL implementation offers insight into hardware design for neural networks, albeit with lower accuracy and non-synthesizable floating-point arithmetic.

