#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal script to read soft labels from batch_xxx.tar files in a directory.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


def get_epoch_dirs(base_dir):
    """
    Get all epoch directories in the base directory.
    
    Args:
        base_dir (str): Base directory containing epoch_xxx folders
        
    Returns:
        list: List of epoch directories sorted by epoch number
    """
    epoch_dirs = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('epoch_'):
            epoch_dirs.append(item)
    
    # Sort by epoch number
    epoch_dirs.sort(key=lambda x: int(x.split('_')[1]))
    return epoch_dirs


def get_batch_files(epoch_dir):
    """
    Get all batch files in an epoch directory.
    
    Args:
        epoch_dir (str): Path to epoch directory
        
    Returns:
        list: List of batch files sorted by batch number
    """
    batch_files = []
    for item in os.listdir(epoch_dir):
        if item.endswith('.tar') and item.startswith('batch_'):
            batch_files.append(item)
    
    # Sort by batch number
    batch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return batch_files


def compute_batch_statistics(soft_label):
    """
    Compute statistics for a batch of soft labels.
    
    Args:
        soft_label (torch.Tensor): Soft labels tensor
        
    Returns:
        dict: Dictionary containing statistics
    """
    stats = {}
    
    # Convert to numpy for easier computation if needed
    if isinstance(soft_label, torch.Tensor):
        # Move to CPU if on GPU
        if soft_label.is_cuda:
            soft_label_np = soft_label.detach().cpu().numpy()
        else:
            soft_label_np = soft_label.detach().numpy()
    else:
        soft_label_np = np.array(soft_label)
    
    # Basic statistics
    stats['shape'] = list(soft_label_np.shape)
    stats['min'] = float(np.min(soft_label_np))
    stats['max'] = float(np.max(soft_label_np))
    stats['mean'] = float(np.mean(soft_label_np))
    stats['std'] = float(np.std(soft_label_np))
    
    # Per-sample statistics
    max_values = np.max(soft_label_np, axis=1)
    stats['max_per_sample_mean'] = float(np.mean(max_values))
    stats['max_per_sample_min'] = float(np.min(max_values))
    stats['max_per_sample_max'] = float(np.max(max_values))
    stats['max_per_sample_std'] = float(np.std(max_values))
    
    # Compute top-1 class distribution
    top_classes = np.argmax(soft_label_np, axis=1)
    unique_classes, counts = np.unique(top_classes, return_counts=True)
    stats['top1_class_count'] = len(unique_classes)
    stats['top1_class_most_common'] = int(unique_classes[np.argmax(counts)])
    stats['top1_class_most_common_count'] = int(np.max(counts))
    
    # Skip entropy calculation for now as it's causing numerical issues
    # We'll focus on other statistics that are more reliable
    stats['entropy_mean'] = 0.0
    stats['entropy_min'] = 0.0
    stats['entropy_max'] = 0.0
    stats['entropy_std'] = 0.0
    
    return stats


def read_soft_labels(base_dir, epochs=None, max_batches=None, verbose=True, batch_processor=None):
    """
    Read soft labels from batch files in the specified epochs.
    
    Args:
        base_dir (str): Base directory containing epoch_xxx folders
        epochs (list, optional): List of epoch numbers to process. If None, process all epochs.
        max_batches (int, optional): Maximum number of batches to process per epoch. If None, process all batches.
        verbose (bool, optional): Whether to show progress bars
        batch_processor (callable, optional): Function to process each batch of soft labels.
            If provided, this function will be called for each batch with arguments:
            (epoch_num, batch_idx, soft_label, batch_path, batch_statistics, absolute_batch_idx)
        
    Returns:
        dict: Dictionary with absolute batch indices as keys and statistics as values
    """
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory {base_dir} does not exist")
    
    # Get all epoch directories
    all_epoch_dirs = get_epoch_dirs(base_dir)
    
    # Filter epochs if specified
    if epochs is not None:
        epoch_dirs = []
        for epoch_dir in all_epoch_dirs:
            epoch_num = int(epoch_dir.split('_')[1])
            if epoch_num in epochs:
                epoch_dirs.append(epoch_dir)
    else:
        epoch_dirs = all_epoch_dirs
    
    if verbose:
        print(f"Found {len(epoch_dirs)} epoch directories")
    
    # use absolute batch index as the key
    batch_statistics = {}
    absolute_batch_idx = 0

    # Process each epoch directory
    epoch_iter = tqdm(epoch_dirs, desc="Processing epochs") if verbose else epoch_dirs
    for epoch_dir_name in epoch_iter:
        epoch_num = int(epoch_dir_name.split('_')[1])
        epoch_path = os.path.join(base_dir, epoch_dir_name)
        
        # Get all batch files in this epoch
        batch_files = get_batch_files(epoch_path)
        
        if max_batches is not None:
            batch_files = batch_files[:max_batches]
        
        # Process each batch file
        batch_iter = tqdm(batch_files, desc=f"Processing batches for epoch {epoch_num}", leave=False) if verbose else batch_files
        for batch_idx, batch_file in enumerate(batch_iter):
            batch_path = os.path.join(epoch_path, batch_file)
            
            try:
                # Load the batch file
                config = torch.load(batch_path, weights_only=False)
                
                # Extract soft labels (based on the code in utils_fkd.py, soft labels are at index 5)
                # [coords, flip_status, mix_index, mix_lam, mix_bbox, soft_label]
                if len(config) >= 6:
                    soft_label = config[5]  # Get the soft labels

                    # soft label should be normalized first using softmax
                    soft_label = F.softmax(soft_label, dim=1)
                    
                    # Compute statistics for this batch
                    stats = compute_batch_statistics(soft_label)
                    
                    # Store statistics with absolute batch index as key
                    batch_statistics[absolute_batch_idx] = {
                        'epoch': epoch_num,
                        'batch_idx': batch_idx,
                        'batch_path': batch_path,
                        'stats': stats
                    }
                    
                    if batch_processor is not None:
                        # Process this batch with the provided function
                        batch_processor(epoch_num, batch_idx, soft_label, batch_path, batch_statistics, absolute_batch_idx)
                    else:
                        # Just print some basic info about the soft label
                        if verbose and batch_idx % 10 == 0:
                            print(f"\nEpoch {epoch_num}, Batch {batch_idx}:")
                            if isinstance(soft_label, torch.Tensor):
                                print(f"  Shape: {soft_label.shape}")
                                print(f"  Type: {soft_label.dtype}")
                                print(f"  Max value: {stats['max']:.4f}")
                                print(f"  Min value: {stats['min']:.4f}")
                                print(f"  Mean entropy: {stats['entropy_mean']:.4f}")
                else:
                    if verbose:
                        print(f"Warning: Batch file {batch_file} does not contain soft labels")
            except Exception as e:
                if verbose:
                    print(f"Error processing {batch_path}: {str(e)}")
            
            absolute_batch_idx += 1
    
    return batch_statistics


def rank_batches(batch_statistics, metric='max_per_sample_mean', top_n=10, ascending=False):
    """
    Rank batches based on a specific metric.
    
    Args:
        batch_statistics (dict): Dictionary with batch statistics
        metric (str): Metric to rank by (must be a key in the stats dictionary)
        top_n (int): Number of top batches to return
        ascending (bool): Whether to sort in ascending order
        
    Returns:
        list: List of (batch_idx, value) tuples sorted by the metric
    """
    # Extract the specified metric from each batch
    metric_values = []
    for batch_idx, batch_data in batch_statistics.items():
        if 'stats' in batch_data and metric in batch_data['stats']:
            metric_values.append((batch_idx, batch_data['stats'][metric]))
    
    # Sort by the metric
    metric_values.sort(key=lambda x: x[1], reverse=not ascending)
    
    # Return the top N
    return metric_values[:top_n]


def analyze_batch_statistics(batch_statistics, output_prefix=None):
    """
    Analyze batch statistics and generate plots.
    
    Args:
        batch_statistics (dict): Dictionary with batch statistics
        output_prefix (str, optional): Prefix for output files
    """
    print(f"Analyzing statistics for {len(batch_statistics)} batches.")
    
    # Example 1: Rank batches by different metrics
    metrics = [
        'max_per_sample_mean',  # Average max confidence
        'top1_class_count',     # Number of unique top-1 classes
        'max',                  # Maximum value in the batch
        'max_per_sample_std'    # Standard deviation of max confidences
    ]
    
    for metric in metrics:
        print(f"\nTop 5 batches by {metric}:")
        ranked = rank_batches(batch_statistics, metric=metric, top_n=5)
        for i, (batch_idx, value) in enumerate(ranked):
            batch_data = batch_statistics[batch_idx]
            print(f"{i+1}. Epoch {batch_data['epoch']}, Batch {batch_data['batch_idx']}: {value:.4f}")
    
    # # Example 2: Plot distribution of a metric across all batches
    # if output_prefix is not None:
    #     metric_to_plot = 'max_per_sample_mean'
    #     values = [batch_data['stats'][metric_to_plot] for batch_data in batch_statistics.values()]
        
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(values, bins=20)
    #     plt.title(f"Distribution of {metric_to_plot} across batches")
    #     plt.xlabel(metric_to_plot)
    #     plt.ylabel("Count")
    #     plot_file = f"{output_prefix}_{metric_to_plot}_distribution.png"
    #     plt.savefig(plot_file)
    #     print(f"\nSaved distribution plot to {plot_file}")
    
    # Example 3: Find batches with the most diverse class distribution
    diverse_batches = rank_batches(batch_statistics, metric='top1_class_count', top_n=10)
    print("\nBatches with most diverse class distribution:")
    for i, (batch_idx, value) in enumerate(diverse_batches):
        batch_data = batch_statistics[batch_idx]
        print(f"{i+1}. Epoch {batch_data['epoch']}, Batch {batch_data['batch_idx']}: {int(value)} unique classes")
    
    # Example 4: Find batches with the highest confidence
    confident_batches = rank_batches(batch_statistics, metric='max_per_sample_mean', top_n=10)
    print("\nBatches with highest average confidence:")
    for i, (batch_idx, value) in enumerate(confident_batches):
        batch_data = batch_statistics[batch_idx]
        print(f"{i+1}. Epoch {batch_data['epoch']}, Batch {batch_data['batch_idx']}: {value:.4f}")


def my_processor(epoch_num, batch_idx, soft_label, batch_path, batch_statistics, absolute_batch_idx):
    """
    Custom processor function to handle soft labels.
    
    Args:
        epoch_num (int): Current epoch number
        batch_idx (int): Current batch index
        soft_label (tensor): Soft labels for this batch
        batch_path (str): Path to the batch file
        batch_statistics (dict): Dictionary containing statistics for all batches
        absolute_batch_idx (int): Absolute batch index
    """
    # Only process every 10th batch to avoid too much output
    if batch_idx % 10 == 0:
        print(f"\nProcessing epoch {epoch_num}, batch {batch_idx}")
        
        if isinstance(soft_label, torch.Tensor):
            print(f"  Shape: {soft_label.shape}")
            
            # Print some statistics if available
            if batch_statistics and absolute_batch_idx in batch_statistics:
                stats = batch_statistics[absolute_batch_idx]['stats']
                print(f"  Value range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Mean value: {stats['mean']:.4f}")
                print(f"  Max confidence per sample (mean): {stats['max_per_sample_mean']:.4f}")
                print(f"  Max confidence per sample (range): [{stats['max_per_sample_min']:.4f}, {stats['max_per_sample_max']:.4f}]")
                print(f"  Unique top-1 classes: {stats['top1_class_count']} (out of {stats['shape'][1]})")
                print(f"  Most common top-1 class: {stats['top1_class_most_common']} (count: {stats['top1_class_most_common_count']})")


def main():
    parser = argparse.ArgumentParser(description='Read soft labels from batch files')
    parser.add_argument('--dir', type=str, 
                        help='Directory containing epoch_xxx folders with batch_xxx.tar files')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                        help='Specific epochs to process (e.g., --epochs 0 1 2). If not specified, process all epochs.')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to process per epoch')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress information')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save the batch statistics (as a PyTorch file). If not specified, will save to label_stats.pt in the input directory.')
    parser.add_argument('--rank-by', type=str, default=None,
                        help='Rank batches by this metric (e.g., max_per_sample_mean, top1_class_count)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top batches to show when ranking')
    parser.add_argument('--ascending', action='store_true',
                        help='Sort in ascending order when ranking (default is descending)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze the statistics and generate plots')
    parser.add_argument('--load', type=str, default=None,
                        help='Load statistics from a file instead of processing batches')
    
    args = parser.parse_args()

    # Either load statistics from a file or process batches
    if args.load:
        if not os.path.exists(args.load):
            print(f"Error: Statistics file {args.load} not found.")
            return
        
        print(f"Loading statistics from {args.load}")
        batch_statistics = torch.load(args.load)
        print(f"Loaded statistics for {len(batch_statistics)} batches.")
    else:
        # Make sure directory is provided if not loading from file
        if not args.dir:
            parser.error("--dir is required when not using --load")
            
        # Read and process soft labels
        batch_statistics = read_soft_labels(
            args.dir, 
            epochs=args.epochs, 
            max_batches=args.max_batches,
            verbose=not args.quiet, 
            batch_processor=my_processor
        )
    
    # Rank batches if requested
    if args.rank_by:
        print(f"\nRanking batches by {args.rank_by} ({'ascending' if args.ascending else 'descending'}):")
        ranked_batches = rank_batches(
            batch_statistics, 
            metric=args.rank_by, 
            top_n=args.top_n,
            ascending=args.ascending
        )
        
        print(f"\nTop {len(ranked_batches)} batches:")
        for i, (batch_idx, value) in enumerate(ranked_batches):
            batch_data = batch_statistics[batch_idx]
            print(f"{i+1}. Epoch {batch_data['epoch']}, Batch {batch_data['batch_idx']}: {args.rank_by} = {value:.4f}")
    
    # Analyze statistics if requested
    if args.analyze:
        output_prefix = os.path.splitext(args.output)[0] if args.output else "batch_stats"
        analyze_batch_statistics(batch_statistics, output_prefix)
    
    # Save batch statistics if output file is specified or use default
    if args.dir and not args.load:
        output_file = args.output
        if output_file is None:
            # Use default output file in the input directory
            output_file = os.path.join(args.dir, "label_stats.pt")
        
        print(f"Saving batch statistics to {output_file}")
        torch.save(batch_statistics, output_file)
        print("Done!")


if __name__ == '__main__':
    main()

    """
    Example:
    python read_soft_labels.py --dir /path/to/syn_label_LPQLD/FKD_cutmix_fp16_LPLD_in1k_rn18_4k_ipc10 --max-batches 10000 --quiet
    """