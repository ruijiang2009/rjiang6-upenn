import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        print("__init__")
        super(RNN, self).__init__()  # Calling the parent class (nn.Module) initializer

        self.hidden_size = hidden_size  # Define the size of the hidden state

        # Linear layer taking concatenated input and hidden state to the next hidden state
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Linear layer to map hidden state to output
        # A hidden layer in a neural network is between the input and output layers and captures patterns
        # in the data by applying weights and activation functions.
        self.h2o = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output (useful for classification tasks)
        # The softmax function converts a vector of values into a probability distribution, 
        # often used in multi-class classification to assign probabilities to different classes.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Concatenate the input and hidden tensors along dimension 1
        # print("input is: ", input)
        # print("hidden is: ", hidden)
        # print("input is on:", input.device)
        # print("hidden is on:", hidden.device)
        if input.device != hidden.device: # added by ruijiang
            # print("input is on:", input.device)
            # print("hidden is on:", hidden.device)
            input = input.to(hidden.device)

        combined = torch.cat((input, hidden), 1)
        # Pass the concatenated tensor through the i2h layer to get the next hidden state
        hidden = self.i2h(combined)
        # Pass the hidden state through the h2o layer to get the raw output
        output = self.h2o(hidden)
        # Apply softmax to the raw output
        output = self.softmax(output)
        # Return the final output and the new hidden state
        return output, hidden

    def initHidden(self):
        # Initializes hidden state with zeros
        return torch.zeros(1, self.hidden_size)

import random

def random_training_pair(X, y, seed = None): # seed is required for penngrader only.
    '''
    Pseudocode:
        1. Initialize a random generator with given seed.
        2. Generate a random index 'ind' between 0 and (number of rows in X) - 1.
        3. Fetch 'category' from y and 'line' from X using the random index 'ind'.
        4. Convert 'category' to a tensor and move it to the specified device.
        5. Convert 'line' to a tensor by calling the function lineToTensor.
        6. Return 'category', 'line', 'category_tensor', and 'line_tensor'.

    Input:
        training data:
            X: features
            y: labels
            seed: needed for randomness

    Returns:
        A tuple of 4 items:
            category: output label(category) as an integer,
            line: input line (here by word) as a string,
            category_tensor: the category as a tensor. Ex) category = 1 => category_tensor = tensor([1]),
                            Tip: make sure to send your tensor to GPU!
            line_tensor: line as a tensor. Tip: use lineToTensor()!
    '''
    # 1. Initialize a random generator with given seed.
    if seed is not None:
        random.seed(seed)

    # 2. Generate a random index 'ind' between 0 and (number of rows in X) - 1.
    ind = random.randint(0, len(X)-1)

    # 3. Fetch 'category' from y and 'line' from X using the random index 'ind'.
    category = int(y[ind])
    line = str(X[ind])

    # 4. Convert 'category' to a tensor and move it to the specified device.
    category_tensor = torch.tensor([category], device=device)

    # 5. Convert 'line' to a tensor by calling the function lineToTensor.
    line_tensor = lineToTensor(line).to(device)

    # 6. Return 'category', 'line', 'category_tensor', and 'line_tensor'.
    return category, line, category_tensor, line_tensor

def trainOneEpoch(model, criterion, optimizer, X, y):
    '''
    Define a function to train the model for one epoch called trainOneEpoch.

    Do the following steps:

    1. Reset any accumulated gradients in the model to zero.
    2. Initialize a hidden state for the model using its initHidden method.
    3. Randomly select a training pair (a category and a line, along with their tensor representations) using the random_training_pair function on X and y.
    4. Loop over each tensor (character) in the line_tensor:
    a. For each tensor, pass it and the current hidden state into the model to get the predicted output and the next hidden state.
    5. Once the entire line_tensor is processed, compute the loss by comparing the model's final output to the true category_tensor using the provided criterion.
    6. Propagate the error backward through the model to compute the gradients.
    7. Update the model's parameters using the optimizer's step method.
    8. Return the model's output, the computed loss as a single value, and the original line and category from the random training pair.

    Inputs:
        - model: the neural network model we want to train
        - criterion: the loss function to calculate the training error
        - optimizer: the optimization algorithm to adjust model parameters
        - X: the input data
        - y: the corresponding labels

    Returns:
        - output: the model's final output (prediction)
        - output_loss: the computed loss as a single value
        - line: the randomly choosen line from random_training_pair()
        - category: the randomly choosen category from random_training_pair()
    '''
    # Zeroing the gradients to clear up the accumulated history
    model.zero_grad()
    # Initializing the hidden state for the model
    hidden = model.initHidden().to(device)
    # TODO: implement step 3, 4
    category, line, category_tensor, line_tensor = random_training_pair(X, y)
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)


    # Calculating the loss between the model's output and the actual target (category_tensor)
    loss = criterion(output, category_tensor)
    # Backward pass: compute the gradient of the loss with respect to model parameters
    loss.backward()
    # Updating the model parameters based on the calculated gradients
    optimizer.step()
    # Extracting the value of the loss as a Python number
    output_loss = loss.data.item()
    return output, output_loss, line, category

# TEST: DO NOT CHANGE
test_model = RNN(input_size=len(all_letters), hidden_size=10).to(device)
test_model.train()
before = list(test_model.parameters())[-1].clone()
output, loss, line, category = trainOneEpoch(test_model, nn.NLLLoss(),
              torch.optim.SGD(test_model.parameters(), lr=0.2),
              word_train_data[0], word_train_data[1])
after = list(test_model.parameters())[-1].clone()

assert not np.array_equal(before.detach().cpu().numpy(), after.detach().cpu().numpy())

def predict(model, X, y = None, loss_func = None):
    '''
    Make predictions on the input data X using the given model.
    Optionally calculate the average loss using true labels y and loss function loss_func.

    Inputs:
        model: trained model
        X: a list of words
        y: a list of categories (optional)
        loss_func: a loss function (optional)
    Returns:
        predictions: as a NumPy array if y and loss_func are None, else the average loss.
    '''
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        # Initialize lists to store predictions and individual losses
        pred = []
        val_loss = []
        # Loop over each sample in the input data X
        for ind in range(X.shape[0]):
            # Initialize hidden state
            hidden = model.initHidden().to(device)
            # Convert the current input sample to a tensor
            val = lineToTensor(X[ind])
            # Loop over each element in the input tensor and get the model's output
            for i in range(val.size()[0]):
                output, hidden = model(val[i], hidden)
            # Move the output tensor back to CPU and extract data (log probabilities)
            log_probabilities = output.cpu().data
            # Calculate the prediction by comparing the log probabilities
            log_prob0, log_prob1 = log_probabilities[0]
            pred.append(int(log_prob0 < log_prob1))
            # If true labels and a loss function are provided, calculate the loss for the current sample
            if y is not None and loss_func is not None:
                category_tensor = torch.tensor([int(y[ind])]).to(device)
                val_loss.append(loss_func(output, category_tensor).data.item())

    # If true labels and a loss function were provided, return the average loss
    if y is not None and loss_func is not None:
        return sum(val_loss) / len(val_loss)

    # Otherwise, return the predictions as a NumPy array
    return np.array(pred)

def run(train_data, val_data, hidden_size, n_epochs, learning_rate, loss_func, print_every, plot_every, model_name):
    X, y = train_data
    X_val, y_val = val_data
    model = RNN(input_size=len(all_letters), hidden_size=hidden_size)
    model = model.to(device)
    current_loss = 0
    train_losses = []
    val_losses = []

    for epoch in range(0, n_epochs):
        output, loss, line, category = trainOneEpoch(model,
                    criterion = loss_func,
                    optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate),
                    X=X,
                    y=y)
        current_loss += loss

        # print intermediate reports
        if epoch % print_every == 0:
            log_probabilities = output.cpu().data
            log_prob0, log_prob1 = log_probabilities[0]
            prediction = int(log_prob0 < log_prob1)
            correct = 'correct' if prediction == category else 'incorrect (True:%s)' % category
            print('Epoch %d (%d%%)  Loss: %.4f, Word: %s, Prediction: %s | %s' % (epoch, epoch / n_epochs * 100, loss, line, prediction, correct))

        if epoch % plot_every == 0:
            # Training Loss
            train_losses.append(current_loss/plot_every)
            current_loss= 0

            # Validation Loss
            val_losses.append(predict(model, X_val, y_val, loss_func))

    torch.save(model.state_dict(), model_name)
    return train_losses, val_losses

# Let's do simple vs complex word classification!
# 1 is complex, 0 is simple
# Don't worry about the hyperparameters, we will take a look at them later.
# training will take around 5 minutes
word_train_losses, word_val_losses = run(train_data = word_train_data,
                              val_data = word_val_data,
                              hidden_size = 50,
                              n_epochs = 50000,
                              learning_rate = 0.005,
                              loss_func = nn.NLLLoss(),
                              print_every = 5000,
                              plot_every = 250,
                              model_name = "./word_RNN"
                            )