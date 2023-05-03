# EconmetPerceptron
Standard econometric models use Maximum Likelihood Estimation or situational equivalents (ie OLS) for calculations. `EconmetPerceptron` uses its own implementation of a Perceptron for estimates robust to distributional properties. Notably,
- Linear Regression models are no longer be fragile to heteroskedasticity or a non-zero error term, and is more robust to multicollinearity (but endogeneity remains a problem).
- Logistic Regression models no 
- Dynamic panel models no longer need more individuals than observations (unlike Arellano-Bond GMM estimator).
- ARIMA is less fragile to non-stationarity (this implementation uses a single layer perceptron).

Nonlinear models are also supported, as wel as those with multiple hidden layers. These have the benefit of further ignoring distributional properties. For instance, time-series models may completely ignore non-stationarity. These specific models are implemented:
- Vector Autoencoding Nonlinear Autoregression (VANAR)
- Deep Instrumental Variables (Deep IV)
- more to come.

# Preliminaries

A good rule of thumb for hidden layers:
- The number of hidden nodes in each layer should be somewhere between the size of the input and output layer, potentially the mean.
- The number of hidden nodes need exceed twice the number of input nodes, as you are already overfitting at this point.

# Examples

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
model.fit(NewIndep, Z, NewEndog, epochs, batch_size, learning_rate, epoch_step = 100)
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
