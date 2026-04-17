"""
Analyze temperature loss landscape data from training runs

This script loads temperature-loss data saved during training and performs additional
analyses to help validate the research approach:

1. Plots the full loss landscape with highlighted minimum
2. Creates a zoomed view around the minimum to check smoothness
3. Calculates derivatives to check for stability
4. Compares multiple epochs to show how the optimal temperature evolves
5. Creates a 3D visualization (temperature vs. epoch vs. loss)

Usage:
    python analyze_temperature_loss.py --data_dir /path/to/temperature_plots

Author: Research Team
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import glob
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze temperature loss landscape data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing saved temperature loss data (.npz files)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results (defaults to data_dir/analysis)')
    parser.add_argument('--epochs', type=str, default=None,
                        help='Specific epochs to analyze, comma-separated (e.g., "0,10,50,100")')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply smoothing to the loss curves')
    parser.add_argument('--derivative_order', type=int, default=2,
                        help='Order of derivative to calculate (1 for first derivative, 2 for second)')
    
    return parser.parse_args()


def load_data_files(data_dir, epochs=None):
    """Load all temperature loss data files from the directory"""
    npz_files = glob.glob(os.path.join(data_dir, 'temp_loss_data_epoch*.npz'))
    
    # Extract epoch numbers from filenames
    epoch_numbers = [int(f.split('epoch')[-1].split('.')[0]) for f in npz_files]
    
    # Sort by epoch number
    sorted_indices = np.argsort(epoch_numbers)
    npz_files = [npz_files[i] for i in sorted_indices]
    epoch_numbers = [epoch_numbers[i] for i in sorted_indices]
    
    # Filter by specified epochs if provided
    if epochs is not None:
        epoch_list = [int(e) for e in epochs.split(',')]
        filtered_files = []
        filtered_epochs = []
        for f, e in zip(npz_files, epoch_numbers):
            if e in epoch_list:
                filtered_files.append(f)
                filtered_epochs.append(e)
        npz_files = filtered_files
        epoch_numbers = filtered_epochs
    
    # Load the data
    data_by_epoch = {}
    for file, epoch in zip(npz_files, epoch_numbers):
        data = np.load(file)
        data_by_epoch[epoch] = {
            'temperatures': data['temperatures'],
            'losses': data['losses'],
            'optimal_temp': data['optimal_temp']
        }
    
    return data_by_epoch


def smooth_curve(x, y, smoothing_factor=0.3):
    """Apply spline smoothing to the curve"""
    # Number of points in new interpolated curve
    n_points = len(y)
    
    # Create a B-spline representation of the curve
    spl = make_interp_spline(x, y, k=3)
    
    # Generate new x values
    x_smooth = np.linspace(x.min(), x.max(), n_points)
    
    # Generate new y values
    y_smooth = spl(x_smooth)
    
    return x_smooth, y_smooth


def savgol_smooth(y, window_length=51, poly_order=3):
    """Apply Savitzky-Golay filter to smooth the curve"""
    # Make sure window_length is odd and not larger than the data
    if window_length >= len(y):
        window_length = min(len(y) - 1, 51)
    if window_length % 2 == 0:
        window_length += 1
        
    return savgol_filter(y, window_length, poly_order)


def calculate_derivative(x, y, order=1):
    """Calculate numerical derivative of the curve"""
    if order == 1:
        # First derivative
        dx = np.diff(x)
        dy = np.diff(y)
        derivative = dy / dx
        # Center the x values for the derivative
        x_derivative = (x[:-1] + x[1:]) / 2
        return x_derivative, derivative
    elif order == 2:
        # First get the first derivative
        x_d1, y_d1 = calculate_derivative(x, y, order=1)
        # Then get the second derivative
        x_d2, y_d2 = calculate_derivative(x_d1, y_d1, order=1)
        return x_d2, y_d2
    else:
        raise ValueError(f"Derivative order {order} not supported")


def analyze_single_epoch(data, epoch, output_dir, apply_smoothing=False, derivative_order=2):
    """Analyze and visualize the loss landscape for a single epoch"""
    temperatures = data['temperatures']
    losses = data['losses']
    optimal_temp = data['optimal_temp']
    
    # Find the index of the optimal temperature
    opt_idx = np.argmin(losses)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original data plot
    axs[0, 0].plot(temperatures, losses, 'b-', alpha=0.7, label='Original Data')
    axs[0, 0].scatter(optimal_temp, losses[opt_idx], color='red', s=100, zorder=5)
    axs[0, 0].set_xlabel('Temperature')
    axs[0, 0].set_ylabel('KL Divergence Loss')
    axs[0, 0].set_title(f'Temperature vs. Loss Landscape (Epoch {epoch})')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Add annotation for the optimal temperature
    axs[0, 0].annotate(f'Optimal T={optimal_temp:.6f}\nLoss={losses[opt_idx]:.6f}',
                xy=(optimal_temp, losses[opt_idx]),
                xytext=(optimal_temp + 0.1, losses[opt_idx] + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10)
    
    # Smoothed data if requested
    if apply_smoothing:
        # Apply Savitzky-Golay smoothing
        smooth_losses = savgol_smooth(losses)
        axs[0, 0].plot(temperatures, smooth_losses, 'g-', alpha=0.7, label='Smoothed Data')
        
        # Find the smoothed optimal temperature
        smooth_opt_idx = np.argmin(smooth_losses)
        smooth_opt_temp = temperatures[smooth_opt_idx]
        axs[0, 0].scatter(smooth_opt_temp, smooth_losses[smooth_opt_idx], color='green', s=100, zorder=5)
        axs[0, 0].annotate(f'Smoothed Optimal T={smooth_opt_temp:.6f}',
                    xy=(smooth_opt_temp, smooth_losses[smooth_opt_idx]),
                    xytext=(smooth_opt_temp + 0.1, smooth_losses[smooth_opt_idx] - 0.1),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
        
        # Use the smoothed data for the remaining analysis
        working_losses = smooth_losses
    else:
        working_losses = losses
    
    axs[0, 0].legend()
    
    # Zoomed view around the minimum
    zoom_range = max(int(len(temperatures) * 0.1), 20)
    zoom_start = max(0, opt_idx - zoom_range)
    zoom_end = min(len(temperatures), opt_idx + zoom_range)
    
    zoom_temps = temperatures[zoom_start:zoom_end]
    zoom_losses = working_losses[zoom_start:zoom_end]
    
    axs[0, 1].plot(zoom_temps, zoom_losses, 'g-')
    axs[0, 1].scatter(optimal_temp, working_losses[opt_idx], color='red', s=100, zorder=5)
    axs[0, 1].set_xlabel('Temperature')
    axs[0, 1].set_ylabel('KL Divergence Loss')
    axs[0, 1].set_title(f'Temperature vs. Loss Landscape (Zoomed)')
    axs[0, 1].grid(True, alpha=0.3)
    
    # First derivative
    x_d1, y_d1 = calculate_derivative(temperatures, working_losses, order=1)
    axs[1, 0].plot(x_d1, y_d1, 'r-')
    axs[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[1, 0].axvline(x=optimal_temp, color='k', linestyle='--', alpha=0.3)
    axs[1, 0].set_xlabel('Temperature')
    axs[1, 0].set_ylabel('First Derivative')
    axs[1, 0].set_title('First Derivative of Loss Landscape')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Second derivative or higher
    if derivative_order >= 2:
        x_d2, y_d2 = calculate_derivative(temperatures, working_losses, order=2)
        axs[1, 1].plot(x_d2, y_d2, 'purple')
        axs[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axs[1, 1].axvline(x=optimal_temp, color='k', linestyle='--', alpha=0.3)
        axs[1, 1].set_xlabel('Temperature')
        axs[1, 1].set_ylabel('Second Derivative')
        axs[1, 1].set_title('Second Derivative of Loss Landscape')
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'analysis_epoch{epoch}.png'), dpi=300)
    plt.close(fig)
    
    # Create a dedicated high-resolution zoomed plot focusing on the local minimum
    plt.figure(figsize=(10, 8))
    
    # Create a more narrow zoom range for a closer look at the minimum
    narrow_zoom_range = max(int(len(temperatures) * 0.05), 10)
    narrow_zoom_start = max(0, opt_idx - narrow_zoom_range)
    narrow_zoom_end = min(len(temperatures), opt_idx + narrow_zoom_range)
    
    narrow_zoom_temps = temperatures[narrow_zoom_start:narrow_zoom_end]
    narrow_zoom_losses = working_losses[narrow_zoom_start:narrow_zoom_end]
    
    # Create a higher resolution interpolation for smoother visualization
    if len(narrow_zoom_temps) > 3:  # Need at least 4 points for cubic interpolation
        try:
            x_new = np.linspace(min(narrow_zoom_temps), max(narrow_zoom_temps), 500)
            spl = make_interp_spline(narrow_zoom_temps, narrow_zoom_losses, k=3)
            y_new = spl(x_new)
            
            plt.plot(x_new, y_new, 'b-', linewidth=2.5, label='Interpolated Data')
            plt.scatter(narrow_zoom_temps, narrow_zoom_losses, color='gray', s=30, alpha=0.6, label='Original Data Points')
        except Exception as e:
            # Fallback if interpolation fails
            plt.plot(narrow_zoom_temps, narrow_zoom_losses, 'b-', linewidth=2.5)
            print(f"Interpolation failed for epoch {epoch}: {e}")
    else:
        # Not enough points for interpolation
        plt.plot(narrow_zoom_temps, narrow_zoom_losses, 'b-', linewidth=2.5)
    
    plt.scatter(optimal_temp, working_losses[opt_idx], color='red', s=150, zorder=5, label='Optimal Temperature')
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('KL Divergence Loss', fontsize=14)
    plt.title(f'High-Resolution View of Local Minimum (Epoch {epoch})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotation for the optimal temperature
    plt.annotate(f'Optimal T={optimal_temp:.6f}\nLoss={working_losses[opt_idx]:.6f}',
                xy=(optimal_temp, working_losses[opt_idx]),
                xytext=(optimal_temp, max(narrow_zoom_losses) - (max(narrow_zoom_losses) - min(narrow_zoom_losses))*0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'local_minimum_zoom_epoch{epoch}.pdf'), dpi=300)
    plt.close()
    
    # Return key metrics
    return {
        'epoch': epoch,
        'optimal_temp': optimal_temp,
        'min_loss': losses[opt_idx],
        'derivative_at_optimal': np.interp(optimal_temp, x_d1, y_d1) if len(x_d1) > 0 else None,
    }


def compare_epochs(data_by_epoch, output_dir, apply_smoothing=False):
    """Compare the loss landscapes across multiple epochs"""
    epochs = sorted(data_by_epoch.keys())
    
    # Plot all epochs on the same graph
    plt.figure(figsize=(15, 8))
    
    # Track optimal temperatures
    optimal_temps = []
    min_losses = []
    
    for epoch in epochs:
        data = data_by_epoch[epoch]
        temperatures = data['temperatures']
        losses = data['losses']
        optimal_temp = data['optimal_temp']
        
        # Smooth if requested
        if apply_smoothing:
            losses = savgol_smooth(losses)
        
        # Plot with alpha based on how many epochs there are
        alpha = 0.7 if len(epochs) < 5 else max(0.3, 1.0 / len(epochs))
        plt.plot(temperatures, losses, alpha=alpha, label=f'Epoch {epoch}')
        
        # Mark the minimum
        min_idx = np.argmin(losses)
        plt.scatter(optimal_temp, losses[min_idx], s=50)
        
        optimal_temps.append(optimal_temp)
        min_losses.append(losses[min_idx])
    
    plt.xlabel('Temperature')
    plt.ylabel('KL Divergence Loss')
    plt.title('Temperature vs. Loss Landscape Across Epochs')
    plt.grid(True, alpha=0.3)
    
    # Add legend only if there aren't too many epochs
    if len(epochs) <= 10:
        plt.legend()
    
    plt.tight_layout()
    plt.ylim(0, 5)
    plt.savefig(os.path.join(output_dir, 'epochs_comparison.pdf'), dpi=300)
    plt.close()
    
    # Plot the trend of optimal temperature across epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, optimal_temps, 'bo-', label='Optimal Temperature')
    plt.xlabel('Epoch')
    plt.ylabel('Optimal Temperature')
    plt.title('Optimal Temperature Trend Across Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_temp_trend.pdf'), dpi=300)
    plt.close()
    
    # Plot the trend of minimum loss across epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, min_losses, 'ro-', label='Minimum Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Minimum Loss')
    plt.title('Minimum Loss Trend Across Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'min_loss_trend.pdf'), dpi=300)
    plt.close()


def create_3d_visualization(data_by_epoch, output_dir):
    """Create a 3D surface plot showing temperature, epoch, and loss"""
    epochs = sorted(data_by_epoch.keys())
    
    # Determine common temperature range
    all_temps = []
    for epoch in epochs:
        all_temps.append(data_by_epoch[epoch]['temperatures'])
    
    # Create 3D surface data
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    
    # Create a color scale based on the number of epochs
    colors = [f'rgb({int(255 * (1 - i/len(epochs)))}, {int(255 * i/len(epochs))}, 150)' 
              for i in range(len(epochs))]
    
    # Add a surface for each epoch
    for i, epoch in enumerate(epochs):
        data = data_by_epoch[epoch]
        temps = data['temperatures']
        losses = data['losses']
        
        # Create a line trace for each epoch
        fig.add_trace(
            go.Scatter3d(
                x=temps,
                y=np.full_like(temps, epoch),
                z=losses,
                mode='lines',
                line=dict(color=colors[i], width=4),
                name=f'Epoch {epoch}'
            )
        )
        
        # Add marker for the minimum
        min_idx = np.argmin(losses)
        fig.add_trace(
            go.Scatter3d(
                x=[temps[min_idx]],
                y=[epoch],
                z=[losses[min_idx]],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                ),
                name=f'Min Epoch {epoch}'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='3D Visualization of Temperature Loss Landscape Across Epochs',
        scene=dict(
            xaxis_title='Temperature',
            yaxis_title='Epoch',
            zaxis_title='Loss',
            xaxis=dict(range=[0, max(d['temperatures'][-1] for d in data_by_epoch.values())]),
            yaxis=dict(range=[min(epochs)-1, max(epochs)+1]),
        ),
        width=1000,
        height=800,
    )
    
    # Save the 3D visualization
    fig.write_html(os.path.join(output_dir, '3d_visualization.html'))


def main():
    args = parse_args()

    # Use the correct directory name with the "_mr50" suffix
    args.data_dir = '/home/tempuse/project/LPLDv2/relabel_and_validate/save/temperature_plots_mr50'
    # every 50 epochs from 1 to 300 inclusively
    args.epochs = '1,100,200,300'
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data files
    data_by_epoch = load_data_files(args.data_dir, args.epochs)
    
    if not data_by_epoch:
        print(f"No temperature loss data files found in {args.data_dir}")
        return
    
    print(f"Found data for {len(data_by_epoch)} epochs: {sorted(data_by_epoch.keys())}")
    
    # Analyze each epoch individually
    metrics_by_epoch = {}
    for epoch, data in data_by_epoch.items():
        print(f"Analyzing epoch {epoch}...")
        metrics = analyze_single_epoch(
            data, 
            epoch, 
            args.output_dir, 
            apply_smoothing=args.smooth,
            derivative_order=args.derivative_order
        )
        metrics_by_epoch[epoch] = metrics
    
    # Compare epochs
    if len(data_by_epoch) > 1:
        print("Comparing epochs...")
        compare_epochs(data_by_epoch, args.output_dir, apply_smoothing=args.smooth)
        
        # Create 3D visualization
        print("Creating 3D visualization...")
        create_3d_visualization(data_by_epoch, args.output_dir)
    
    # Save metrics summary
    with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Epoch\tOptimal Temp\tMin Loss\tDerivative at Optimal\n")
        for epoch in sorted(metrics_by_epoch.keys()):
            m = metrics_by_epoch[epoch]
            if m['derivative_at_optimal'] is not None:
                derivative_str = f"{m['derivative_at_optimal']:.6f}"
            else:
                derivative_str = 'N/A'
            f.write(f"{epoch}\t{m['optimal_temp']:.6f}\t{m['min_loss']:.6f}\t{derivative_str}\n")
    
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
