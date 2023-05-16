import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------
# General Constants
# ---------------
INPUT_FILENAME = "../datasets/student_result_data.txt"
BLUE_CIRCLE_PLT = 'bo'
RED_CIRCLE_PLT = 'ro'
GRAPH_PADDING = 2.0

# ---------------
# Modeling Constants
# ---------------
NUM_EPOCHS = 100
LEARNING_RATE = 0.001


def load_input_file(filename, cols):
    """
    Loads input data from csv.

    Returns (features, labels)
    """
    num_features = len(cols) - 1
    df = pd.read_csv(filename, sep=",", index_col=False)
    df.columns = cols
    data = np.array(df, dtype=float)

    features, labels = data[:, :num_features], data[:, -1]

    num_examples, num_features = features.shape[0], features.shape[1]
    # Add const feature (1)
    features = np.hstack((np.ones((num_examples, 1)), features))
    # Reshape to (NUM_EXAMPLES, 1)
    labels = np.reshape(labels, (num_examples, 1))

    return features, labels


def plot_inputs(features, labels, feature_names):
    """
    Plots inputs for dataset.
    """
    num_examples = features.shape[0]
    plt.xlabel('Score of Test 1')
    plt.ylabel('Score of Test 2')
    for i in range(num_examples):
        test_1_score, test_2_score = features[i]
        if labels[i] == 1:
            plt.plot(test_1_score, test_2_score, 'gX')
        else:
            plt.plot(test_1_score, test_2_score, 'mD')
    plt.show()


def sigmoid_activation(z):
    """
    Implements sigmoid activation function
    """
    return 1/(1 + np.exp(-z))


def log_loss(features, labels, theta):
    """
    Implements Binary cross-entropy (Log Loss) cost function
    """
    # x.shape: (NUM_EXAMPLES, NUM_FEATURES)
    # theta.shape: (NUM_FEATURES, 1)
    # linear_combo.shape: (NUM_EXAMPLES, 1)
    linear_combo = np.matmul(features, theta)

    predictions = sigmoid_activation(linear_combo)

    num_examples = labels.shape[0]
    ones = np.ones((num_examples, 1))

    left_term = np.matmul(labels.T, np.log(predictions))
    right_term = np.matmul(
        (ones - labels).T,
        np.log(ones - predictions)
    )
    return (-(left_term + right_term) / num_examples)


def gradient_descent(features, labels, theta, learning_rate=0.1, num_epochs=10):
    # features.shape: (NUM_EX, NUM_FEATURES)
    num_examples = features.shape[0]
    J_all = []
    thetas = []

    for _ in range(num_epochs):
        linear_combo = np.matmul(features, theta)
        probabilities = sigmoid_activation(linear_combo)

        cost_derivative = (1/num_examples) * \
            np.matmul(features.T, probabilities - labels)

        theta = theta - (learning_rate)*cost_derivative
        thetas.append(theta)

        loss = log_loss(features, labels, theta)[0][0]
        J_all.append(loss)

    return theta, thetas, J_all


def plot_costs_over_epochs(J_all):
    """
    Plots epoch vs Log Loss
    """
    n_epochs = [i for i in range(len(J_all))]
    jplot = np.array(J_all)
    n_epochs = np.array(n_epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(n_epochs, J_all, 'm', linewidth="5")
    plt.show()


def inference(weights, features):
    """
    Returns (Binary Prediction, Admittal Probability) given two test scores
    """
    features = [1] + features
    linear_combo = np.matmul(features, theta)
    prob_of_admittal = float(sigmoid_activation(linear_combo))
    if prob_of_admittal >= 0.5:
        return True
    else:
        return False


# ---------------
# Read input file
# ---------------
raw_features, labels = load_input_file(
    INPUT_FILENAME, ["Test 1 Score", "Test 2 Score", "Accepted/Rejected"]
)

NUM_EXAMPLES = raw_features.shape[0]
# This includes constant (1) feature
NUM_FEATURES = raw_features.shape[1]

# ---------------
# Plot input data
# ---------------
plot_inputs(raw_features[:, 1:], labels, [
    "Test 1 Score", "Test 2 Score"]
)

# ---------------
# Initialize Params
# ---------------
theta = np.zeros((raw_features.shape[1], 1))

# --------------
# Perform GD
# --------------
theta, all_thetas, J_all = gradient_descent(
    raw_features, labels, theta, LEARNING_RATE, NUM_EPOCHS
)
print(f"Final weights: {theta}")

# -------------
# Plot Cost function over time
# -------------
plot_costs_over_epochs(J_all)

# -------------
# Use for inference
# -------------
test_score_a, test_score_b = 0.60, 0.62
features = [test_score_a, test_score_b]
positive_inference = inference(theta, features)
print(
    f"Prediction for {features}: {'admitted' if positive_inference else 'not admitted'}")
