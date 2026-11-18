# DeepLearningEconometrics

Fit PyTorch perceptrons for econometric research, learn state estimates from observed signals, and experiment with neural derivative pricing. The package includes a configurable perceptron, two-stage instrumental-variable models, nonlinear autoregression, and tools for exploring fitted predictions. Its signal-processing and pricing modules provide a supervised Kalman-style filter, a Black–Scholes PDE solver, and terminal distribution transformations.

## Installation

~~~bash
python -m pip install git+https://github.com/datstat-consulting/DeepLearningEconometrics.git
~~~

Run `python -m pip install .` from a checkout. Python 3.10 or later is required.

## Perceptron regression

A perceptron with one input and one linear output fits a single-variable regression. Add hidden layers to model nonlinear relations. With hidden layers, the selected activation applies inside the network and the final output remains linear.

~~~python
import torch
from DeepLearningEconometrics import LossFuncs, Optimizers, PerceptronMain

torch.manual_seed(42)
X = torch.linspace(0, 3, 64)[:, None]
y = 2 * X + 1 + 0.1 * torch.randn_like(X)

nn = PerceptronMain(
    layer_sizes=[1, 1],
    activation_function="linear",
    optimizer_function=Optimizers.adam,
    add_bias=True,
)
nn.fit(X, y, epochs=200, batch_size=32, learning_rate=0.05, loss=LossFuncs.MseLoss)
predictions = nn.predict(X)
~~~

Optimizers include `sgd_optimizer` with momentum, `adagrad_optimizer`, `adam`, `rmsprop`, and `adadelta`. Hidden layers can use `linear`, `relu`, `relu_squared`, `sigmoid` or its alias `logistic`, `tanh`, and `softmax`. Training defaults to `MseLoss`. `RmseLoss` and `TofallisLoss` are also available. The Tofallis loss works with positive targets and predictions.

Changing the optimizer or adding hidden layers changes the fitted prediction function. Endogeneity, model identification, and sampling uncertainty still depend on the data and the research design.

## Two-stage instrumental variables

`DeepIv` fits a first network from instruments `Z` to regressors `X_iv`, then trains a second network to predict outcomes using the fitted regressors. `DeepGmm` uses the same two-stage structure and computes a residual-moment criterion. Pass feature rows to `predict`. To predict from instruments, first obtain the fitted features from `first_stage_network`.

~~~python
from DeepLearningEconometrics import DeepGmm, DeepIv

Z = torch.randn(128, 1)
disturbance = torch.randn_like(Z)
X_iv = Z + 0.4 * disturbance
outcome = 2 * X_iv + 0.3 * disturbance

settings = dict(
    first_stage_layer_sizes=[1, 8, 1],
    second_stage_layer_sizes=[1, 8, 1],
    first_activation="tanh",
    second_activation="tanh",
    optimizer_function=Optimizers.adam,
)
iv = DeepIv(**settings)
iv.fit(X_iv, Z, outcome, epochs=120, batch_size=32, learning_rate=0.01)
fitted_outcome = iv.predict(iv.first_stage_network.predict(Z))

gmm = DeepGmm(**settings)
gmm.fit(X_iv, Z, outcome, epochs=120, batch_size=32, learning_rate=0.01, regularize=True)
gmm_prediction = gmm.predict(X_iv)
~~~

## Autoregressive models

`ArimaSlp` is the earlier single-layer ARIMA experiment. `Vanar` reconstructs lag inputs with an autoencoder and fits a nonlinear forecaster. This example uses a univariate series and a chronological validation portion during fitting.

~~~python
from DeepLearningEconometrics import Vanar

time = torch.linspace(0, 10, 150)
series = torch.sin(time) + 0.1 * torch.randn_like(time)
vanar = Vanar(
    n_lags=5, n_variables=1, hidden_layer_sizes=[8], n_components=3,
    autoencoder_activ="tanh", forecaster_activ="tanh",
    autoen_optim=Optimizers.adam, fore_optim=Optimizers.adam,
)
vanar.fit(
    series[:, None], auto_epochs=40, fore_epochs=40,
    batch_size=32, learning_rate=0.01,
)
forecast = vanar.predict_next_period(series, horizon=5)
indices = vanar.nonlinear_granger_causality(
    epochs=40, batch_size=32, learning_rate=0.01, activation_function="tanh"
)
~~~

The Granger routine compares prediction errors when excluding a lag representation. Treat these indices and the associated `granger_causality_p_values` helper as exploratory predictive diagnostics.

## Causal inference experiments

`CausalDAG` records proposed edges in a directed graph. `CausalInference` contains experimental code for Mahalanobis-distance matching with a perceptron outcome model and a random-common-cause refutation. The current matching path depends on a covariance helper missing from the package. The example below initializes the graph for an identification exercise.

~~~python
from DeepLearningEconometrics import CausalDAG

graph = CausalDAG()
graph.add_edge("treatment", "outcome")
graph.add_edge("covariate", "treatment")
graph.add_edge("covariate", "outcome")
graph.show_edges()
~~~

## Neural filtering

`KalmanNet` learns a state transition, an observation map, and a recurrent gain from aligned observations and labelled states. At inference time the gain network carries its hidden state across the observation sequence. Initialize the sequence with a state estimate for the period immediately before its first observation.

~~~python
from DeepLearningEconometrics import KalmanNet, butterworth_filter

time = torch.arange(64) / 100
states = torch.stack((torch.sin(4 * torch.pi * time), torch.cos(4 * torch.pi * time)), dim=1)
observations = states[:, :1] + 0.05 * torch.randn(64, 1)
model = KalmanNet([2, 8, 2], [2, 8, 1])
model.fit(states, observations, epochs=40, batch_size=16, learning_rate=0.003)
estimates = model(observations[1:], states[0])
smoothed_signal = butterworth_filter(observations[:, 0], cutoff=5, sample_rate=100)
~~~

The standalone Butterworth function uses zero-phase filtering by default. Set `zero_phase=False` for a forward-only filter.

For an individual observation, call the transition and update methods directly. Each standalone update initializes a fresh recurrent state. Use the sequence interface above when the gain needs the preceding observations.

~~~python
prior = model.predict_next_state(states[0])
one_step = model.update_state(prior, observations[1])
~~~


## European Black–Scholes prices

`BlackScholesPINN` trains a physics-informed neural network on the Black–Scholes PDE residual. Its output enforces the terminal payoff and the boundary conditions on a bounded spot-price domain. Compare the trained approximation with the analytic formula at the same time remaining.

~~~python
from DeepLearningEconometrics import BlackScholesPINN

torch.manual_seed(5)
pricing_model = BlackScholesPINN(strike=100, maturity=1, volatility=0.2, rate=0.03)
losses = pricing_model.fit(steps=300, seed=5)
approximation = pricing_model(100, 0.5)
benchmark = pricing_model.exact_price(100, 0.5)
~~~

## Terminal distribution transport

A `DombiKappa` map transforms CDF probabilities and sends the first anchor probability to the second. `TerminalTransport` uses this map to evaluate the transformed law. For discrete log-return scenarios, `calibrate_esscher` chooses exponential tilting weights to match the requested expected gross return.

~~~python
from scipy.stats import norm
from DeepLearningEconometrics import DombiKappa, TerminalTransport, calibrate_esscher

physical = norm(loc=0.05, scale=0.2)
transformed = TerminalTransport(physical, DombiKappa(0.5, 0.4))
median = transformed.ppf(0.5)
theta, weights = calibrate_esscher([-0.2, 0.0, 0.15], gross_target=1.0)
~~~

These operations describe a terminal law. A dynamic pricing measure also requires assumptions about the price process between observation times.

## Prediction attribution

`PerceptronShap` measures how predictions change when it fixes one feature across random draws near an observation. Use `compute_shap_values_single` to inspect one row or `compute_shap_values` for several. The associated plot methods accept Matplotlib or Plotly.

~~~python
from DeepLearningEconometrics import PerceptronShap

explainer = PerceptronShap(nn, num_samples=1000)
values, base_value = explainer.compute_shap_values_single(X[0], X.shape[1])
batch_values, batch_base = explainer.compute_shap_values(X[:10], X.shape[1])
~~~

The reported values summarize prediction sensitivity to the chosen perturbation distribution. They depend on the fitted network and the sampled inputs.

## References

- Bennett, A., Kallus, N., & Schnabel, T. (2019). Deep generalized method of moments for instrumental variable analysis. Advances in neural information processing systems, 32.
- Cabanilla, K. I., & Go, K. T. (2019). Forecasting, Causality, and Impulse Response with Neural Vector Autoregressions. arXiv preprint arXiv:1903.09395.
- Hartford, J., Lewis, G., Leyton-Brown, K., & Taddy, M. (2017, July). Deep IV: A flexible approach for counterfactual prediction. In International Conference on Machine Learning (pp. 1414-1423). PMLR.
- King, G., & Nielsen, R. (2019). Why propensity scores should not be used for matching. Political analysis, 27(4), 435-454.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.
- Revach, G., Shlezinger, N., Ni, X., Escoriza, A. L., Van Sloun, R. J., & Eldar, Y. C. (2022). KalmanNet: Neural network aided Kalman filtering for partially known dynamics. IEEE Transactions on Signal Processing, 70, 1532-1547.
- Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for causal inference. arXiv preprint arXiv:2011.04216.
- Shevitski, B., Watkins, Y., Man, N., & Girard, M. (2021). Digital signal processing using deep neural networks. arXiv preprint arXiv:2109.10404.
- Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. B. (2021). Neural granger causality. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8), 4267-4279.
- Tofallis, C. (2015). A better measure of relative prediction accuracy for model selection and model estimation. Journal of the Operational Research Society, 66, 1352-1362.
