class PerceptronMain:
    def __init__(self, layer_sizes, activation_function, optimizer_function, weight_decay= 0.0, add_bias = True):
        self.layer_sizes = layer_sizes
        self.activation_function = TorchActivations.activation(activation_function)
        self.activation_derivative = TorchActivations.derivative(activation_function)
        self.optimizer_function = optimizer_function
        self.add_bias = add_bias
        self.weight_decay = weight_decay
        #self.optimizer_params = {}
        self.initialize_weights()
        if self.add_bias:
            self.layer_sizes[0] += 1

    def initialize_weights(self, dtype=torch.float64):
        self.weights = [torch.randn(n, m, dtype=dtype) for n, m in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    def forward(self, X):
        self.a_values = [X]
        for w in self.weights:
            self.a_values.append(self.activation_function(self.a_values[-1] @ w))
        return self.a_values[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        gradients = [torch.zeros_like(w) for w in self.weights]

        if y.dim() == 1:
            y = y.view(-1, 1)

        delta = (self.a_values[-1] - y) * self.activation_derivative(self.a_values[-2] @ self.weights[-1])
        gradients[-1] = self.a_values[-2].t() @ delta + self.weight_decay * self.weights[-1]

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].t()) * self.activation_derivative(self.a_values[i] @ self.weights[i])
            gradients[i] = self.a_values[i].t() @ delta + self.weight_decay * self.weights[i]

        return gradients

    def optimize(self, gradients, learning_rate):
        #self.weights, updated_params = self.optimizer_function(self.weights, gradients, learning_rate, self.weight_decay, **self.optimizer_params)
        #self.optimizer_params.update(updated_params)
        self.weights = self.optimizer_function(self.weights, gradients, learning_rate, self.weight_decay)

    def fit(self, X, y, epochs, batch_size, learning_rate, epoch_step=100, optimizer_parameters=None):
       # if optimizer_parameters is None:
            #optimizer_parameters = {}
        step = epoch_step
        current_epochs = epochs
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        self.initialize_weights(dtype=X.dtype)

        if self.add_bias:
            # Add a column of 1s to the input data
            X = torch.cat((X, torch.ones((X.shape[0], 1))), dim=1)
        
        while current_epochs > 0:
            print(f"Trying {current_epochs} epochs.")
            
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    for epoch in range(current_epochs):
                        for i in range(0, X.shape[0], batch_size):
                            X_batch = X[i:min(i + batch_size, X.shape[0])]
                            y_batch = y[i:min(i + batch_size, y.shape[0])]
                            self.forward(X_batch)
                            gradients = self.backward(X_batch, y_batch, learning_rate)
                            
                            self.optimize(gradients, learning_rate)
                            
                            # Update weights with the optimizer function and provided parameters
                            #self.weights, *optimizer_params_values = self.optimizer_function(
                                #self.weights, gradients, learning_rate, self.weight_decay, **optimizer_parameters
                            #)
                            
                            # Update optimizer_parameters with the new values returned by the optimizer function
                            #for key, value in zip(optimizer_parameters.keys(), optimizer_params_values):
                                #optimizer_parameters[key] = value

                        if w:
                            raise RuntimeWarning("Overflow encountered during training.")

                print(f"Training successful with {current_epochs} epochs.")
                break

            except RuntimeWarning:
                print(f"Warning encountered with {current_epochs} epochs. Reducing the number of epochs.")
                current_epochs -= step

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        # Add a column of 1s to the input data
        X = torch.cat((X, torch.ones((X.shape[0], 1))), dim=1)

        # Ensure that the input tensor X has the same data type as the weights
        X = X.to(self.weights[0].dtype)

        for w in self.weights[:-1]:
            X = self.activation_function(X @ w)
        return X @ self.weights[-1]
    
class Optimizers:
    @staticmethod
    def sgd_optimizer(weights, gradients, learning_rate, weight_decay, momentum=None, velocity=None, eps=None):
        new_weights = [w - learning_rate * (g + weight_decay * w) for w, g in zip(weights, gradients)]
        return new_weights

    #def sgd_optimizer(weights, gradients, learning_rate, weight_decay, momentum=0.0, velocity=None, **kwargs):
        #velocity = [torch.zeros_like(w) for w in weights]
        #new_velocity = [momentum * v + learning_rate * (g + weight_decay * w) for v, w, g in zip(velocity, weights, gradients)]
        #new_weights = [w - v for w, v in zip(weights, new_velocity)]
        #return new_weights, new_velocity

    @staticmethod
    def adagrad_optimizer(weights, gradients, learning_rate, weight_decay, squared_gradients, eps=1e-8, **kwargs):
        new_squared_gradients = [sg + g ** 2 for sg, g in zip(squared_gradients, gradients)]
        new_weights = [w - learning_rate / (torch.sqrt(sg) + eps) * (g + weight_decay * w) for w, sg, g in zip(weights, new_squared_gradients, gradients)]
        return new_weights, new_squared_gradients


class TorchActivations:
    activations = {
        'sigmoid': lambda x: 1 / (1 + torch.exp(-x)),
        'relu': lambda x: torch.max(torch.zeros_like(x), x),
        'relu_squared': lambda x: torch.pow(torch.max(torch.zeros_like(x), x), 2),
        'linear': lambda x: x
    }
    
    derivatives = {
        'sigmoid': lambda x: sigmoid(x) * (1 - sigmoid(x)),
        'relu': lambda x: (x > 0).float(),
        'relu_squared': lambda x: 2 * torch.max(torch.zeros_like(x), x),
        'linear': lambda x: torch.ones_like(x)
    }
    
    @staticmethod
    def activation(activation_name):
        return TorchActivations.activations.get(activation_name, None)
    
    @staticmethod
    def derivative(activation_name):
        return TorchActivations.derivatives.get(activation_name, None)

