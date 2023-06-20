import torch.nn as nn

class KalmanNet:
    def __init__(self, layer_sizes_f, layer_sizes_h, activation_function, optimizer_function, f_add_bias = True, h_add_bias = True):
        self.layer_sizes_f = layer_sizes_f
        self.layer_sizes_h = layer_sizes_h
        self.activation_function = activation_function
        self.optimizer_function = optimizer_function
        # state transition model
        self.model_f = PerceptronMain(self.layer_sizes_f, self.activation_function, self.optimizer_function, add_bias=f_add_bias)
        # observation model
        self.model_h = PerceptronMain(self.layer_sizes_h, self.activation_function, self.optimizer_function, add_bias=h_add_bias)
        # RNN for learning the Kalman Gain
        self.rnn = nn.RNN(input_size=1, hidden_size=1, batch_first=True)

    def fit(self, X, y, epochs, batch_size, learning_rate):
        """
        Fit the state transition and observation models.
        """
        self.model_f.fit(X, y, epochs, batch_size, learning_rate)
        self.model_h.fit(X, y, epochs, batch_size, learning_rate)

    def predict_next_state(self, current_state, add_bias = True):
        """
        Predict the next state given the current state.
        Uses the state transition model f.
        """
        if add_bias == True:
            current_state = torch.cat((current_state, torch.tensor([1.0])))
        next_state_pred = self.model_f.forward(current_state)
        return next_state_pred

    def update_state(self, next_state_pred, observation, add_bias = True):
        """
        Update the predicted next state with the new observation to
        compute the corrected next state estimate.
        Uses the Kalman Gain to correct the prediction.
        """
        if add_bias:
            next_state_pred = torch.cat((next_state_pred, torch.tensor([1.0])))
        
        # Compute the innovation term
        innovation = observation - self.model_h.forward(next_state_pred)
        
        # Get Kalman Gain from the RNN
        innovation_reshaped = innovation.reshape(1, 1, -1) # reshape to be suitable for RNN input
        _, kalman_gain = self.rnn(innovation_reshaped)
        kalman_gain = kalman_gain.squeeze() # remove unnecessary dimensions

        # Compute the updated state estimate
        next_state_estimate = next_state_pred + kalman_gain * innovation

        return next_state_estimate
