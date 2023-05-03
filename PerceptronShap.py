import torch
import itertools
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class PerceptronShap:
    def __init__(self, perceptron, num_samples=1000):
        self.perceptron = perceptron
        self.num_samples = num_samples

    def generate_samples(self, mean, covariance_matrix):
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix)
        samples = dist.sample(sample_shape=(self.num_samples,))
        return samples

    def compute_shap_values_single(self, instance, num_features):
        instance = instance.reshape(1, -1)
        shap_values = torch.zeros(num_features)

        expected_value = torch.mean(self.perceptron.predict(instance))

        mean = torch.mean(instance, dim=0)
        covariance_matrix = torch.eye(num_features) * 0.1
        random_instances = self.generate_samples(mean, covariance_matrix)

        for i in range(num_features):
            without_feature = random_instances.clone()
            with_feature = random_instances.clone()
            with_feature[:, i] = instance[:, i]

            marginal_contribution = self.perceptron.predict(with_feature) - self.perceptron.predict(without_feature)

            shap_values[i] = torch.mean(marginal_contribution)

        return shap_values, expected_value

    def plot_shap_values(self, shap_values, feature_names, expected_value, is_plotly=False):
        shap_values = shap_values.detach().numpy()
        expected_value = expected_value.item()

        if is_plotly:
            fig = go.Figure(go.Bar(y=feature_names, x=shap_values, orientation='h'))
            fig.update_layout(title=f"SHAP Values (Base value: {expected_value:.2f})")
            fig.show()
        else:
            plt.barh(feature_names, shap_values)
            plt.title(f"SHAP Values (Base value: {expected_value:.2f})")
            plt.show()

    def compute_shap_values(self, instances, num_features):
        shap_values_list = []
        expected_value_list = []

        for instance in instances:
            shap_values, expected_value = self.compute_shap_values_single(instance.reshape(1, -1), num_features)
            shap_values_list.append(shap_values)
            expected_value_list.append(expected_value)

        return shap_values_list, expected_value_list

    def plot_aggregated_shap_values(self, shap_values_list, feature_names, expected_value_list, is_plotly=False):
        aggregated_shap_values = torch.mean(torch.stack(shap_values_list), axis=0)
        mean_expected_value = torch.mean(torch.stack(expected_value_list))

        aggregated_shap_values = aggregated_shap_values.detach().numpy()
        mean_expected_value = mean_expected_value.item()

        if is_plotly:
            fig = go.Figure(go.Bar(y=feature_names, x=aggregated_shap_values.tolist(), orientation='h'))
            fig.update_layout(title=f"Aggregated SHAP Values (Mean base value: {mean_expected_value:.2f})")
            fig.show()
        else:
            plt.barh(feature_names, aggregated_shap_values)
            plt.title(f"Aggregated SHAP Values (Mean base value: {mean_expected_value:.2f})")
            plt.show()

