# EconmetPerceptron
Standard econometric models use Maximum Likelihood Estimation or situational equivalents (ie OLS) for calculations. `EconmetPerceptron` uses its own implementation of a Perceptron for estimates robust to distribution properties. Notably,
- Linear Regression models are no longer be fragile to heteroskedasticity or a non-zero error term, and is more robust to multicollinearity (but endogeneity remains a problem).
- Logistic Regression models no 
- Dynamic panel models no longer need more individuals than observations (unlike Arellano-Bond GMM estimator).

Nonlinear models are also supported. These have the benefit of further ignoring distributional properties.
- Vector Autoencoding Nonlinear Autoregression (VANAR) is implemented.
- Deep Instrumental Variables approach (Deep IV) is implemented.
- more features to come.
