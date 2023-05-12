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

## Installation
You can install the package using pip:
```
pip install git+https://github.com/datstat-consulting/EconmetPerceptron
```
# Preliminaries

A good rule of thumb for hidden layers:
- The number of hidden nodes in each layer should be somewhere between the size of the input and output layer, potentially the mean.
- The number of hidden nodes need not exceed twice the number of input nodes, as you are already overfitting at this point.

# Examples

## Generalized Linear Models

The `PerceptronMain` class implements a general perceptron which handles GLMs. 
```
nn = PerceptronMain(layer_sizes=[1, 1], 
                   activation_function="linear", 
                   optimizer_function = Optimizers.sgd_optimizer,
                   weight_decay = 0.0,
                   add_bias = True)

# Train the single layer perceptron with independent variables X and deepndent variable y
nn.fit(X1, y, 
       epochs=1000, 
       batch_size=32, 
       learning_rate=0.0001, 
       epoch_step=100)

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

# Fit on new independent variable
model.predict(NewX)
```

## Deep GMM

The `DeepGmm` class implements a two-stage artificial neural network estimation. Unlike the `DeepIv` class, the `DeepGmm` class uses a GMM loss function for the second estimation stage.
```
model = DeepGmm(first_stage_layer_sizes=[n_instruments, 10, n_features],
               second_stage_layer_sizes=[n_features, 10, 1],
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

## Shapley Value
The Shapley Value of a model shows how each independent variable contributes to output prediction. This is a useful alternative to p-values for interpreting Machine Learning models. It may be computationally efficient to use only one observation to demonstrate how each independent variable contributes to prediction. In this case, we use the very first observation.
```
torch.manual_seed(375)
# Create an instance of the PerceptronShap class
shap_explainer = PerceptronShap(nn, num_samples=1000)

# Select an instance from your dataset
instance_index = 0
instance = X[instance_index]

# Compute the SHAP values and expected value for the selected instance
num_features = X.shape[1]
shap_values, expected_value = shap_explainer.compute_shap_values_single(instance, num_features)

# Plot the SHAP values using either Plotly or Matplotlib
feature_names = [f'Feature {i+1}' for i in range(num_features)]
shap_explainer.plot_shap_values(shap_values, feature_names, expected_value, is_plotly=True)
```
When feasible, especially for mere GLMs, we can use a random sample, or even the entire dataset.
```
num_instances = 10
random_indices = torch.randint(0, len(X), (num_instances,))
random_sample = X[random_indices]
shap_values_list, expected_value_list = shap_explainer.compute_shap_values(X, num_features) # entire data set.

# Plot the aggregated SHAP values using either Plotly or Matplotlib
shap_explainer.plot_aggregated_shap_values(shap_values_list, feature_names, expected_value_list, is_plotly=True)
```
The `PerceptronShap` class will be configured to support more models later on.

# References
- Bennett, A., Kallus, N., & Schnabel, T. (2019). Deep generalized method of moments for instrumental variable analysis. Advances in neural information processing systems, 32.
- Cabanilla, K. I., & Go, K. T. (2019). Forecasting, Causality, and Impulse Response with Neural Vector Autoregressions. arXiv preprint arXiv:1903.09395.
- Hartford, J., Lewis, G., Leyton-Brown, K., & Taddy, M. (2017, July). Deep IV: A flexible approach for counterfactual prediction. In International Conference on Machine Learning (pp. 1414-1423). PMLR.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.
