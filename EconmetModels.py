import torch

# Single Layer Perceptron ARIMA
class ArimaSlp(PerceptronMain):
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        super().__init__(
            layer_sizes=[p + q, 1],
            activation_function="linear",
            optimizer_function=Optimizers.sgd_optimizer,
            weight_decay=0.0,
            add_bias=True
        )

    def fit(self, y, epochs, batch_size, learning_rate, epoch_step = 100):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)

        y_d = torch.diff(y, n=self.d) if self.d > 0 else y

        X_ar = torch.zeros((len(y_d) - self.p, self.p), dtype=torch.float64)
        for t in range(self.p, len(y_d)):
            X_ar[t - self.p] = y_d[t - self.p:t]

        ar_coeffs = WorkhorseFunctions.ols_estimator_torch(X_ar, y_d[self.p:]).view(-1, 1)
        residuals = y_d[self.p:] - X_ar.mm(ar_coeffs).view(-1)

        X_ma = torch.zeros((len(residuals) - self.q + 1, self.q), dtype=torch.float64)
        for t in range(self.q - 1, len(residuals)):
            X_ma[t - self.q + 1] = residuals[t - self.q + 1:t + 1]

        ma_coeffs = WorkhorseFunctions.ols_estimator_torch(X_ma, residuals[self.q - 1:]).squeeze()
        X = torch.cat((X_ar[:len(X_ma)], X_ma), dim=1)

        super().fit(X, y_d[self.p + self.q - 1:], epochs, batch_size, learning_rate, epoch_step)

    def predict_next_period(self, y, horizon):
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)

        y_d = torch.diff(y, n=self.d) if self.d > 0 else y
        predictions = []

        for _ in range(horizon):
            X_ar = y_d[-self.p:].view(1, -1)
            X_ma = y_d[-self.q:].view(1, -1)

            X = torch.cat((X_ar, X_ma), dim=1)
            y_next_d = super().predict(X).item()
            y_next = y_next_d + y[-1] if self.d > 0 else y_next_d

            predictions.append(y_next)
            y_d = torch.cat((y_d, torch.tensor([y_next_d], dtype=torch.float64).unsqueeze(0)), dim=0)
            y = torch.cat((y, torch.tensor([y_next], dtype=torch.float64).unsqueeze(0)), dim=0)

        return torch.tensor(predictions)

# Deep Instrumental Variable 
class DeepIv:
    def __init__(self, first_stage_layer_sizes, second_stage_layer_sizes, first_activation, second_activation, optimizer_function, add_bias = True):
        self.first_stage_network = PerceptronMain(layer_sizes=first_stage_layer_sizes, activation_function=first_activation, optimizer_function=optimizer_function, add_bias = add_bias)
        self.second_stage_network = PerceptronMain(layer_sizes=second_stage_layer_sizes, activation_function=second_activation, optimizer_function=optimizer_function, add_bias = add_bias)

    def fit(self, X, Z, y, epochs, batch_size, learning_rate, epoch_step = 100):
        # Fit the first-stage network using Z as input and X as output
        self.first_stage_network.fit(Z, X, epochs, batch_size, learning_rate, epoch_step = epoch_step)

        # Estimate the instrument variable
        estimated_IV = self.first_stage_network.predict(Z)

        # Fit the second-stage network using the estimated instrument variable and y
        self.second_stage_network.fit(estimated_IV, y, epochs, batch_size, learning_rate, epoch_step = 100)

    def predict(self, X):
        # Estimate the instrument variable
        #estimated_IV = self.first_stage_network.predict(Z)

        # Predict the outcome using the estimated instrument variable
        return self.second_stage_network.predict(X)

# Vector Autoencoding Nonlinear Autoregression
class Vanar:
    def __init__(self, n_lags, hidden_layer_sizes, n_components, autoencoder_activ = "linear", forecaster_activ = "linear", autoen_optim = Optimizers.sgd_optimizer, fore_optim = Optimizers.sgd_optimizer):
        self.n_lags = n_lags
        self.autoencoder = PerceptronMain(
            layer_sizes=[n_lags, n_components, n_lags],
            activation_function=autoencoder_activ,
            optimizer_function=autoen_optim,
        )
        self.forecaster = PerceptronMain(
            layer_sizes=[n_lags] + hidden_layer_sizes + [1],
            activation_function=forecaster_activ,
            optimizer_function=fore_optim,
        )
        
    def initialize_forecaster_weights(self, X, y):
        beta_hat = WorkhorseFunctions.ols_estimator_torch(X, y)
        self.forecaster.weights[0].data = beta_hat.t()

    def fit(self, data, epochs, batch_size, learning_rate, validation_split=0.2, epoch_step = 100):
        X, y = WorkhorseFunctions.create_input_output_pairs(data, self.n_lags)
        n_validation = int(validation_split * X.shape[0])
        X_train, y_train = X[:-n_validation], y[:-n_validation]
        X_val, y_val = X[-n_validation:], y[-n_validation:]

        # Train the autoencoder
        self.autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, epoch_step=epoch_step)

        # Encode the input data
        X_train_encoded = self.autoencoder.predict(X_train)[:, :self.n_lags]
        X_val_encoded = self.autoencoder.predict(X_val)[:, :self.n_lags]
        
        # Initialize VANAR weights with OLS
        self.initialize_forecaster_weights(X_train_encoded, y_train)

        # Train the forecaster
        self.forecaster.fit(X_train_encoded, y_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, epoch_step=epoch_step)

        # Validate the model
        y_val_pred = self.forecaster.predict(X_val_encoded)
        mse = torch.mean((y_val_pred - y_val) ** 2)
        print("Validation MSE:", mse.item())

    def predict_next_period(self, data, horizon):
        predictions = []

        for _ in range(horizon):
            X, _ = WorkhorseFunctions.create_input_output_pairs(data, self.n_lags)
            X_encoded = self.autoencoder.predict(X)[:, :self.n_lags]
            y_next = self.forecaster.predict(X_encoded[-1].unsqueeze(0)).item()
            predictions.append(y_next)
            data = torch.cat((data, torch.tensor([y_next])), dim=0)

        return torch.tensor(predictions)

