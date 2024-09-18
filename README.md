# Deep Learning
Deep Learning Course with François Fleuret (Spring 2020)

Define a list with all the objects composing our neural network (successive linear layers with activation functions in between them).

FORWARD PASS

- Go through the list from start to end calling the "forward" method of each object. We feed it the input of the layer and get in the return the output of the layer, that we feed as the input for the next object, and so on.
- The local grad is also computed and stored in each object.

BACKWARD PASS

- Shuffle the training set.
- Randomly pick a sample from the training set and compute its MSE (= cost function).
- Feed it as an input to the "backward method" of the penultimate object to compute dC/dw (weights) and dC/db (parameters). Update the weights and biases accordingly using the defined learning rate. Feed the compiled gradient (computed before updating the weights) to the previous level. Continue to go through the list from end to start.

Questions
- How to define the learning rate? -> Do we change it when we get closer to the minimum?
- Put a condition (if cost function < epsilon), to stop the training phase of our neural network.
