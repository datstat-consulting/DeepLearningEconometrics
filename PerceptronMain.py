class PerceptronMain:
    def __init__(self, layer_sizes, activation_function, optimizer_function, weight_decay= 0.0, add_bias = True):
        self.layer_sizes = layer_sizes
        self.activation_function = TorchActivations.activation(activation_function)
        self.activation_derivative = TorchActivations.derivative(activation_function)
        self.optimizer_function = optimizer_function
        self.add_bias = add_bias
        self.weight_decay = weight_decay
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
        self.weights = self.optimizer_function(self.weights, gradients, learning_rate, self.weight_decay)

    def fit(self, X, y, epochs, batch_size, learning_rate, epoch_step = 100):
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
