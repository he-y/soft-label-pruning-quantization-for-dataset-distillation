import torch
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
import os
import json

class EpochSampler(RandomSampler):
    def __init__(self, data_source, generator=None, initial_epoch=0):
        super().__init__(data_source, generator=generator)
        self.epoch = initial_epoch
        self.indices_epoch = {}

    def set_epoch(self, epoch):
        """Set the current epoch to sample the data, for epoch tracking."""
        self.epoch = epoch

    def __iter__(self):
        # If the epoch's indices are already stored, use them;
        # otherwise, use the parent class's iterator to generate and store them.
        if self.epoch in self.indices_epoch:
            indices = self.indices_epoch[self.epoch]
        else:
            indices = list(super().__iter__())  # Generate indices using RandomSampler's mechanism
            self.indices_epoch[self.epoch] = indices
        return iter(indices)

class EpochBatchSampler(RandomSampler):
    """
    Allow the sampler to be specified for a certain epoch or batch
    """
    def __init__(self, data_source, generator=None, initial_epoch=0, use_batch=False):
        super().__init__(data_source, generator=generator)
        self.epoch = initial_epoch
        self.indices_epoch = {}
        # batch related
        self.batch_num = -1
        self.batch_num_per_epoch = -1
        self.batch_size = -1
        self.batch_idx = -1 # use to set the batch index
        self.batch_list = None  # recieve a list of batch indices
                                # to ensemble an epoch
        self.indices_batch = None

    def use_batch(self, batch_size):
        self.batch_size = batch_size
        self._create_batch_indices(batch_size)
        # set sampler length to 1
    
    def set_batch(self, batch_idx):
        self.batch_idx = batch_idx
        self.batch_list = None
        
    def set_batch_list(self, batch_list):
        """Receive a list of batch indices to ensemble an epoch"""
        self.batch_list = batch_list
        self.batch_idx = -1
    
    def get_batch_list_img_mapping(self):
        return {self.indices_batch[batch_idx][0]: batch_idx for batch_idx in self.batch_list}

    def _create_batch_indices(self, batch_size):
        # create the indices for the batch
        self.indices_batch = {}
        batch_index = 0
        for epoch in range(max(self.indices_epoch.keys()) + 1):
            epoch_indices = self.indices_epoch[epoch]

            for i in range(0, len(epoch_indices), self.batch_size):
                self.indices_batch.update({batch_index: epoch_indices[i:i + self.batch_size]})
                batch_index += 1  # Increment batch index for the next batch
            
        self.batch_num = self.indices_batch.keys().__len__()

    def set_epoch(self, epoch):
        """Set the current epoch to sample the data, for epoch tracking."""
        self.epoch = epoch

    def __iter__(self):
        # If the epoch's indices are already stored, use them;
        # otherwise, use the parent class's iterator to generate and store them.
        if self.indices_batch is not None:
            if self.batch_list is not None:
                """ensemble the batches"""
                indices = [self.indices_batch[batch_idx] for batch_idx in self.batch_list]
                # flatten the list
                indices = [item for sublist in indices for item in sublist]
            else:
                """use the batch indices if it is set"""
                indices = self.indices_batch[self.batch_idx]
        elif self.epoch in self.indices_epoch:
            indices = self.indices_epoch[self.epoch]
        else:
            indices = list(super().__iter__())  # Generate indices using RandomSampler's mechanism
            self.indices_epoch[self.epoch] = indices
        return iter(indices)
    
class BatchSampler(RandomSampler):
    """
    A sampler that works directly with batch indices instead of epochs.
    This is designed for the flat batch structure where all batches are stored
    in the root directory with batch_{}.tar format.
    """
    def __init__(self, data_source, generator=None, batch_indices_path=None):
        super().__init__(data_source, generator=generator)
        self.batch_indices = {}
        self.batch_idx = -1  # current batch index
        self.batch_list = None  # list of batch indices to ensemble
        
        # load batch indices if provided
        if batch_indices_path and os.path.exists(batch_indices_path):
            self._load_batch_indices(batch_indices_path)
    
    def _load_batch_indices(self, batch_indices_path):
        """load the batch indices from the json file"""
        import json
        with open(batch_indices_path, 'r') as f:
            self.batch_indices = json.load(f)
    
    def set_batch(self, batch_idx):
        """set the batch index to sample from"""
        self.batch_idx = batch_idx
        self.batch_list = None
    
    def set_batch_list(self, batch_list):
        """set a list of batch indices to ensemble"""
        self.batch_list = batch_list
        self.batch_idx = -1
    
    def __iter__(self):
        if self.batch_list is not None:
            # ensemble multiple batches
            indices = []
            for batch_idx in self.batch_list:
                batch_key = str(batch_idx)
                if batch_key in self.batch_indices:
                    indices.extend(self.batch_indices[batch_key])
            return iter(indices)
        elif self.batch_idx >= 0:
            # use a single batch
            batch_key = str(self.batch_idx)
            if batch_key in self.batch_indices:
                return iter(self.batch_indices[batch_key])
            else:
                # if batch index not found, return empty iterator
                return iter([])
        else:
            # fallback to random sampling
            return super().__iter__()

def rank_batches_by_metric(stats_path, metric='max_per_sample_mean', top_n=None, order='descending'):
    """
    Rank batches based on a specific metric from the label_stats.pt file.
    
    Args:
        stats_path (str): Path to the label_stats.pt file
        metric (str): Metric to rank by. Must be one of the keys in the stats dictionary.
            Examples: 'max_per_sample_mean', 'top1_class_count', 'max', 'entropy_mean', etc.
        top_n (int, optional): Number of top batches to return. If None, return all batches.
        order (str, optional): How to sort the batches. Options:
            - 'ascending': Sort from lowest to highest metric value
            - 'descending': Sort from highest to lowest metric value (default)
            - 'middle': Select batches from the middle of the sorted list
        
    Returns:
        list: List of batch indices sorted by the specified metric
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at {stats_path}")
    
    # Load the statistics
    batch_statistics = torch.load(stats_path)
    
    # Create a list of (batch_idx, metric_value) tuples
    batch_metrics = []
    for batch_idx, batch_data in batch_statistics.items():
        # Check if the metric is directly in the batch data
        if metric in batch_data:
            metric_value = batch_data[metric]
        # Check if the metric is in the stats dictionary
        elif 'stats' in batch_data and metric in batch_data['stats']:
            metric_value = batch_data['stats'][metric]
        else:
            raise ValueError(f"Metric '{metric}' not found in batch statistics")
        
        batch_metrics.append((batch_idx, metric_value))
    
    # Sort by the metric value
    if order == 'ascending':
        batch_metrics.sort(key=lambda x: x[1])
    elif order == 'descending':
        batch_metrics.sort(key=lambda x: x[1], reverse=True)
    elif order == 'middle':
        # Sort in ascending order first
        batch_metrics.sort(key=lambda x: x[1])
        
        # If top_n is specified, select the middle portion
        if top_n is not None:
            total_batches = len(batch_metrics)
            start_idx = (total_batches - top_n) // 2
            batch_metrics = batch_metrics[start_idx:start_idx + top_n]
    else:
        raise ValueError(f"Unknown order: {order}. Must be 'ascending', 'descending', or 'middle'")
    
    # Return the top N batch indices if specified and we're not using 'middle' order
    # (for 'middle' order, we've already selected the middle portion above)
    if top_n is not None and order != 'middle':
        batch_metrics = batch_metrics[:top_n]
    
    # Return just the batch indices
    return [batch_idx for batch_idx, _ in batch_metrics]

def select_batches_by_strategy(fkd_path, strategy='random', metric=None, top_n=None, order='descending', percentage=None):
    """
    Select batches based on a specific strategy.
    
    Args:
        fkd_path (str): Path to the FKD directory containing the label_stats.pt file
        strategy (str): Strategy to use for selecting batches. 
            Options: 'random', 'metric_ranking'
        metric (str, optional): Metric to rank by if strategy is 'metric_ranking'.
            Examples: 'max_per_sample_mean', 'top1_class_count', 'max', 'entropy_mean', etc.
        top_n (int, optional): Number of top batches to select. If None, use percentage.
        order (str, optional): How to sort batches for metric_ranking. Options:
            - 'ascending': Sort from lowest to highest metric value
            - 'descending': Sort from highest to lowest metric value (default)
            - 'middle': Select batches from the middle of the sorted list
        percentage (float, optional): Percentage of batches to select (0.0 to 1.0). Used if top_n is None.
        
    Returns:
        list: List of selected batch indices
    """
    stats_path = os.path.join(fkd_path, 'label_stats.pt')
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at {stats_path}")
    
    # Load the statistics to get the total number of batches
    batch_statistics = torch.load(stats_path)
    total_batches = len(batch_statistics)
    
    # Determine how many batches to select
    if top_n is None and percentage is not None:
        top_n = int(total_batches * percentage)
    elif top_n is None:
        top_n = total_batches  # Select all batches if neither top_n nor percentage is specified
    
    # Select batches based on the strategy
    if strategy == 'random':
        # Randomly select batch indices
        all_batch_indices = list(batch_statistics.keys())
        np.random.shuffle(all_batch_indices)
        selected_batches = all_batch_indices[:top_n]
    
    elif strategy == 'metric_ranking':
        if metric is None:
            raise ValueError("Metric must be specified for metric_ranking strategy")
        
        # Rank batches by the specified metric
        selected_batches = rank_batches_by_metric(
            stats_path, 
            metric=metric, 
            top_n=top_n, 
            order=order
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return selected_batches