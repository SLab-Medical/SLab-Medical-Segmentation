"""
Real-time plotting utilities for training metrics
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict


class MetricsPlotter:
    """Real-time plotter for training metrics"""

    def __init__(self, save_dir, metrics=['loss', 'dice'], window_size=100):
        """
        Args:
            save_dir: Directory to save plots
            metrics: List of metric names to plot
            window_size: Moving average window size
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics = metrics
        self.window_size = window_size

        # Store all values
        self.history = defaultdict(list)
        self.iterations = []

    def update(self, iteration, **kwargs):
        """
        Update metrics with new values

        Args:
            iteration: Current training iteration
            **kwargs: Metric name and value pairs (e.g., loss=0.5, dice=0.9)
        """
        self.iterations.append(iteration)

        for metric_name, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.history[metric_name].append(value)
            else:
                # Handle tensor values
                self.history[metric_name].append(float(value))

    def moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return data

        cumsum = np.cumsum(np.insert(data, 0, 0))
        ma = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        # Pad the beginning
        return np.concatenate([data[:window_size-1], ma])

    def plot(self, show_ma=True):
        """
        Generate and save plots for all metrics

        Args:
            show_ma: Whether to show moving average line
        """
        if len(self.iterations) == 0:
            return

        # Create subplots for each metric
        n_metrics = len(self.history)
        if n_metrics == 0:
            return

        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        for idx, (metric_name, values) in enumerate(self.history.items()):
            ax = axes[idx]

            # Plot raw values
            ax.plot(self.iterations, values, alpha=0.3, label=f'{metric_name} (raw)', linewidth=0.5)

            # Plot moving average
            if show_ma and len(values) >= self.window_size:
                ma_values = self.moving_average(values, self.window_size)
                ax.plot(self.iterations, ma_values, label=f'{metric_name} (MA-{self.window_size})', linewidth=2)

            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()} vs Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add current value text
            if len(values) > 0:
                current_val = values[-1]
                ax.text(0.02, 0.98, f'Current: {current_val:.4f}',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.save_dir, 'metrics.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def save_csv(self):
        """Save metrics history to CSV file"""
        import csv

        csv_path = os.path.join(self.save_dir, 'metrics.csv')

        # Prepare data
        headers = ['iteration'] + list(self.history.keys())

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i, iteration in enumerate(self.iterations):
                row = [iteration]
                for metric_name in self.history.keys():
                    if i < len(self.history[metric_name]):
                        row.append(self.history[metric_name][i])
                    else:
                        row.append('')
                writer.writerow(row)
