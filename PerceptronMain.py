"""
Разработанный Адриелу Ванг от ДанСтат Консульти́рования
"""

import torch
class PerceptronMain:
    def __init__(self, layer_sizes, activation_function, optimizer_function,
                 weight_decay=0.0, add_bias=True):
        self.layer_sizes = list(layer_sizes)
        self.activation_function = TorchActivations.activation(activation_function)
        self.activation_derivative = TorchActivations.derivative(activation_function)
        if self.activation_function is None or len(self.layer_sizes) < 2 or min(self.layer_sizes) < 1:
            raise ValueError("Provide positive layer sizes and a supported activation.")
        self.optimizer_function = optimizer_function
        self.add_bias = add_bias
        self.weight_decay = weight_decay
        self.layer_sizes[0] += int(add_bias)
        self.initialize_weights()

    def initialize_weights(self, dtype=torch.float64, device=None):
        self.weights = [
            torch.randn(n, m, dtype=dtype, device=device) / n ** 0.5
            for n, m in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        self.velocity = self.squared_gradients = self.m = self.v = None
        self._optimizer_step = 0

    def _input(self, X):
        weights = self.weights[0]
        X = torch.as_tensor(X, dtype=weights.dtype, device=weights.device)
        if self.add_bias and X.shape[-1] == weights.shape[0] - 1:
            X = torch.cat((X, torch.ones_like(X[..., :1])), dim=-1)
        return X

    def forward(self, X):
        self.a_values = [self._input(X)]
        for w in self.weights[:-1]:
            self.a_values.append(self.activation_function(self.a_values[-1] @ w))
        self.a_values.append(self.a_values[-1] @ self.weights[-1])
        return self.a_values[-1]

    def backward(self, X, y, learning_rate, loss=None):
        with torch.enable_grad():
            original = self.weights
            self.weights = [w.detach().requires_grad_(True) for w in original]
            try:
                prediction = self.forward(X)
                target = torch.as_tensor(y, device=prediction.device, dtype=prediction.dtype)
                if target.ndim + 1 == prediction.ndim and prediction.shape[-1] == 1:
                    target = target.unsqueeze(-1)
                if prediction.shape != target.shape:
                    raise ValueError("Target dimensions must match the predictions.")
                objective = (loss or LossFuncs.MseLoss)(self.a_values, target)
                if not torch.isfinite(objective):
                    raise ValueError("Training loss is nonfinite.")
                return torch.autograd.grad(objective, self.weights)
            finally:
                self.weights = original
                self.a_values = [a.detach() for a in self.a_values]

    def optimize(self, gradients, learning_rate, momentum):
        if self.optimizer_function != Optimizers.adagrad_optimizer:
            gradients = [g + self.weight_decay * w for g, w in zip(gradients, self.weights)]
            decay = 0.0
        else:
            decay = self.weight_decay
        result = self.optimizer_function(
            self.weights, gradients, learning_rate, decay,
            momentum=momentum, velocity=self.velocity, squared_gradients=self.squared_gradients,
            m=self.m, v=self.v, t=self._optimizer_step,
        )
        if len(result) == 2:
            self.weights, state = result
            if self.optimizer_function == Optimizers.adagrad_optimizer:
                self.squared_gradients = state
            else:
                self.velocity = state
        else:
            self.weights, self.velocity, self.squared_gradients, self.m, self.v = result
        self._optimizer_step += 1

    def fit(self, X, y, epochs, batch_size, learning_rate, momentum=0, epoch_step=100,
            loss=None):
        X, y = torch.as_tensor(X), torch.as_tensor(y)
        if X.ndim != 2 or len(X) == 0 or len(X) != len(y) or batch_size <= 0:
            raise ValueError("Provide aligned training rows and a positive batch size.")
        dtype = X.dtype if X.is_floating_point() else torch.get_default_dtype()
        self.initialize_weights(dtype, X.device)
        for _ in range(epochs):
            for start in range(0, len(X), batch_size):
                batch = slice(start, start + batch_size)
                gradients = self.backward(X[batch], y[batch], learning_rate, loss)
                self.optimize(gradients, learning_rate, momentum)
        return self

    def predict(self, X):
        return self.forward(X)


class Optimizers:
    @staticmethod
    def sgd_optimizer(weights, gradients, learning_rate, weight_decay, momentum=0.0, velocity=None, **kwargs):
        if velocity is None:
            velocity = [torch.zeros_like(w) for w in weights]

        # update the velocity
        velocity = [momentum * v + (1 - momentum) * g for v, g in zip(velocity, gradients)]
        
        # update the weights
        new_weights = [w - learning_rate * v for w, v in zip(weights, velocity)]

        return new_weights, velocity

    @staticmethod
    def adagrad_optimizer(weights, gradients, learning_rate, weight_decay, squared_gradients=None, eps=1e-8, **kwargs):
        if squared_gradients is None:
            squared_gradients = [torch.zeros_like(w) for w in weights]

        # update the squared gradients
        new_squared_gradients = [sg + g ** 2 for sg, g in zip(squared_gradients, gradients)]
        
        # update the weights
        new_weights = [w - learning_rate / (torch.sqrt(sg) + eps) * (g + weight_decay * w) for w, sg, g in zip(weights, new_squared_gradients, gradients)]

        return new_weights, new_squared_gradients

    @staticmethod
    def adadelta(weights, gradients, learning_rate, weight_decay, eps=1e-8, rho=0.9, velocity=None, squared_gradients=None, **kwargs):
        if squared_gradients is None:
            squared_gradients = [torch.zeros_like(w) for w in weights]
        if velocity is None:
            velocity = [torch.zeros_like(w) for w in weights]

        # update squared_gradients and velocity
        new_squared_gradients = [rho * sg + (1 - rho) * g ** 2 for sg, g in zip(squared_gradients, gradients)]
        delta = [torch.sqrt((v + eps) / (sg + eps)) * g for v, g, sg in zip(velocity, gradients, new_squared_gradients)]
        new_velocity = [rho * v + (1 - rho) * d ** 2 for v, d in zip(velocity, delta)]

        # update weights
        new_weights = [w - d - learning_rate * weight_decay * w for w, d in zip(weights, delta)]

        return new_weights, new_velocity, new_squared_gradients, None, None

    @staticmethod
    def rmsprop(weights, gradients, learning_rate, weight_decay, eps=1e-8, rho=0.9, squared_gradients=None, **kwargs):
        if squared_gradients is None:
            squared_gradients = [torch.zeros_like(w) for w in weights]

        # update squared_gradients
        new_squared_gradients = [rho * sg + (1 - rho) * g ** 2 for sg, g in zip(squared_gradients, gradients)]

        # update weights
        new_weights = [w - learning_rate / (torch.sqrt(sg) + eps) * g - learning_rate * weight_decay * w for w, g, sg in zip(weights, gradients, new_squared_gradients)]

        return new_weights, None, new_squared_gradients, None, None

    @staticmethod
    def adam(weights, gradients, learning_rate, weight_decay, eps=1e-8, beta1=0.9, beta2=0.999, m=None, v=None, t=0, **kwargs):
        if m is None:
            m = [torch.zeros_like(w) for w in weights]
        if v is None:
            v = [torch.zeros_like(w) for w in weights]

        # update m and v
        new_m = [beta1 * mt + (1 - beta1) * g for mt, g in zip(m, gradients)]
        new_v = [beta2 * vt + (1 - beta2) * g ** 2 for vt, g in zip(v, gradients)]

        # compute bias corrected estimates
        m_hat = [mt / (1 - beta1 ** (t + 1)) for mt in new_m]
        v_hat = [vt / (1 - beta2 ** (t + 1)) for vt in new_v]

        # update weights
        new_weights = [w - learning_rate * m_h / (torch.sqrt(v_h) + eps) - learning_rate * weight_decay * w for w, m_h, v_h in zip(weights, m_hat, v_hat)]

        return new_weights, None, None, new_m, new_v

class TorchActivations:
    activations = {
        'sigmoid': lambda x: 1 / (1 + torch.exp(-x)),
        'tanh': lambda x: torch.tanh(x),
        'relu': lambda x: torch.max(torch.zeros_like(x), x),
        'relu_squared': lambda x: torch.pow(torch.max(torch.zeros_like(x), x), 2),
        'linear': lambda x: x,
        'softmax': lambda x: torch.softmax(x, dim=-1),
        'logistic': lambda x: 1 / (1 + torch.exp(-x))  # Logistic is the same as sigmoid
    }
    
    derivatives = {
        'sigmoid': lambda x: TorchActivations.activations['sigmoid'](x) * (1 - TorchActivations.activations['sigmoid'](x)),
        'tanh': lambda x: 1 - torch.pow(TorchActivations.activations['tanh'](x), 2),
        'relu': lambda x: (x > 0).float(),
        'relu_squared': lambda x: 2 * torch.max(torch.zeros_like(x), x),
        'linear': lambda x: torch.ones_like(x),
        'softmax': lambda x: TorchActivations.activations['softmax'](x) * (1 - TorchActivations.activations['softmax'](x)),
        'logistic': lambda x: TorchActivations.activations['logistic'](x) * (1 - TorchActivations.activations['logistic'](x))  # Logistic is the same as sigmoid
    }
    
    @staticmethod
    def activation(activation_name):
        return TorchActivations.activations.get(activation_name, None)
    
    @staticmethod
    def derivative(activation_name):
        return TorchActivations.derivatives.get(activation_name, None)

class LossFuncs:
    """
    Proposed loss function by Chris Tofallis
    """
    @staticmethod
    def TofallisLoss(a, y):
        if torch.any(a[-1] <= 0) or torch.any(y <= 0):
            raise ValueError("The log-accuracy loss requires positive forecasts and targets.")
        return (a[-1].log() - y.log()).square().mean()

    """
    Root Mean Square Error
    """
    @staticmethod
    def RmseLoss(a, y):
        return torch.sqrt(torch.mean((a[-1] - y)**2))

    """
    Mean Square Error
    """
    @staticmethod
    def MseLoss(a, y):
        return torch.mean((a[-1] - y)**2)
