import torch

class WorkhorseFunctions:

    # OLS estimator
    def ols_estimator_torch(X, y):
        y = y.view(-1, 1)  # Reshape y to have an extra dimension
        XtX = X.t().mm(X)
        Xty = X.t().mm(y)
        beta_hat = torch.linalg.solve(XtX, Xty)
        return beta_hat
    
    # VAR helper function
    def create_input_output_pairs(data, n_lags):
        X, y = [], []
        for i in range(n_lags, len(data)):
            X.append(data[i - n_lags:i])
            y.append(data[i])
        return torch.tensor(X), torch.tensor(y)

class TimeSeriesWorkhorse:
    # Initialize AR and MA parameters using OLS estimation
    def initialize_params_torch(y, p, q):
        # AR part
        X_ar = torch.zeros((len(y) - p, p), dtype=torch.float64)
        for t in range(p, len(y)):
            for i in range(p):
                X_ar[t - p, i] = y[t - i - 1]
        ar_coeffs = WorkhorseFunctions.ols_estimator_torch(X_ar, y[p:])

        # Compute the residuals
        residuals = y[p:] - X_ar.mm(ar_coeffs).view(-1)

        # MA part
        X_ma = torch.zeros((len(residuals) - q, q), dtype=torch.float64)
        for t in range(q, len(residuals)):
            for j in range(q):
                X_ma[t - q, j] = residuals[t - j - 1]
        ma_coeffs = WorkhorseFunctionsols_estimator_torch(X_ma, residuals[q:])

        return ar_coeffs.view(-1), ma_coeffs.view(-1)

    # Compute the negative log-likelihood and gradients for the ARIMA model
    def negative_log_likelihood_torch(params, y, p, d, q):
        ar_params = params[:p]
        ma_params = params[p:p + q]
        intercept = params[-1]

        y_hat = torch.zeros_like(y)
        for t in range(p, len(y)):
            ar_term = sum(ar_params[i] * y[t - i - 1] for i in range(p))
            ma_term = sum(ma_params[j] * (y[t - j - 1] - intercept) for j in range(q))
            y_hat[t] = intercept + ar_term + ma_term

        residuals = y - y_hat
        sigma2 = torch.sum(residuals ** 2) / len(residuals)
        log_likelihood = -0.5 * (len(residuals) * torch.log(2 * torch.acos(torch.zeros(1)).item() * 2 * sigma2) + torch.sum(residuals ** 2) / sigma2)

        gradients = torch.zeros_like(params)
        for t in range(p, len(y)):
            for i in range(p):
                gradients[i] += residuals[t] * y[t - i - 1]
            for j in range(q):
                gradients[p + j] += residuals[t] *(residuals[t - j - 1] - intercept)
                gradients[-1] += residuals[t]
                gradients /= sigma2

        return -log_likelihood, -gradients

    def arima_estimator_torch(y, p, d, q, learning_rate=0.01, n_iterations=500):
        if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float64)
        if d > 0:
            y = torch.diff(y, n=d)
        # Initialize the AR, MA, and intercept parameters using OLS
        ar_coeffs, ma_coeffs = initialize_params_torch(y, p, q)
        params = torch.cat((ar_coeffs, ma_coeffs, torch.zeros(1, dtype=torch.float64)), dim=0)

        # Optimize the negative log-likelihood using custom SGD
        for i in range(n_iterations):
            neg_loglik, neg_grads = negative_log_likelihood_torch(params, y, p, d, q)
            params -= learning_rate * neg_grads

        ar_coeffs = params[:p]
        ma_coeffs = params[p:p + q]
        intercept = params[-1]
        return ar_coeffs, ma_coeffs, intercept
