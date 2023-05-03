# EconmetPerceptron
Standard econometric models use Maximum Likelihood Estimation or situational equivalents (ie OLS) for calculations. `EconmetPerceptron` uses its own implementation of a Perceptron for estimates robust to distributional properties. Notably,
- Linear Regression models are no longer fragile to heteroskedasticity or a non-zero error term, and is more robust to multicollinearity (but endogeneity remains a problem).
- Logistic Regression models are more robust to multicollinearity.
- Dynamic panel models no longer need more individuals than observations (unlike Arellano-Bond GMM estimator).
- ARIMA is less fragile to non-stationarity (this implementation uses a single layer perceptron).

Nonlinear models are also supported, as well as those with multiple hidden layers. These have the benefit of further ignoring distributional properties. For instance, time-series models may completely ignore non-stationarity. These specific models are implemented:
- Vector Autoencoding Nonlinear Autoregression (VANAR)
- Deep Instrumental Variables (Deep IV)
- more to come

# Preliminaries

A good rule of thumb for hidden layers:
- The number of hidden nodes in each layer should be somewhere between the size of the input and output layer, potentially the mean.
- The number of hidden nodes need exceed twice the number of input nodes, as you are already overfitting at this point.

# Examples

## Generalized Linear Models

The `PerceptronMain` class implements a general perceptron which handles GLMs.

```
nn = PerceptronMain(layer_sizes=[1, 1], # replace output layer with number of classes if categorical regression model. 
                   activation_function="linear", 
                   activation_derivative="linear", 
                   optimizer_function=Optimizers.sgd_optimizer,
                   weight_decay = 0.0,
                   add_bias = True # for intercept term in GLMs
                   )

# Train the single layer perceptron with independent variables X and deepndent variable y
nn.fit(X, y, epochs=1000, batch_size=32, learning_rate=0.0001, epoch_step = 100)

# Make predictions using the trained single layer perceptron
predictions = nn.predict(X)
```

## Deep Instrumental Variables

The `DeepIv` class implements a two-stage artificial neural network estimation.

```
model = DeepIv(first_stage_layer_sizes=[n_instruments, 10, n_features],
               second_stage_layer_sizes=[n_features, 10, n_classes],
               first_activation="relu",
               second_activation="relu",
               optimizer_function=Optimizers.sgd_optimizer)

# Train the DeepIV model
epochs = 1000
batch_size = 32
learning_rate = 0.001
model.fit(X, Z, y, epochs, batch_size, learning_rate, epoch_step = 100)
```

## VANAR
The `Vanar` class is suitable for both univariate and multivariate datasets. 
```
vanar = Vanar(n_lags=5, hidden_layer_sizes=[10], n_components=3, autoencoder_activ="relu", forecaster_activ="relu", autoen_optim = Optimizers.sgd_optimizer, fore_optim = Optimizers.sgd_optimizer)

vanar.fit(endog, epochs=1000, batch_size=32, learning_rate=0.001)
y_pred_vanar = vanar.predict_next_period(data, horizon=5)
print("VANAR predictions:", y_pred_vanar)
```
To remove autoencoding, simply set `n_components` to be the same as `n_lags`, and set `autoencoder_activ="linear"`.
