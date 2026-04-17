# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""


class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            if hasattr(self.dataset, "G_VBSM") and self.dataset.G_VBSM:
                pass # G_VBSM: uses self-decoding in the training script
            elif hasattr(self.dataset, "use_batch") and self.dataset.use_batch:
                mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config_by_batch_idx(possibly_batched_index[0])
            elif hasattr(self.dataset, "lpldv2") and self.dataset.lpldv2:
                if hasattr(self.dataset, "label_quantization") and self.dataset.label_quantization:
                    if self.dataset.label_quantization[:3] == 'MRS': # soft label pruning + label quantization 
                        mix_index, mix_lam, mix_bbox, soft_label, indices, sample_min = self.dataset.load_batch_config(str(possibly_batched_index[:3]))
                    elif self.dataset.label_quantization[:2] == 'MR': # soft label pruning + label quantization 
                        mix_index, mix_lam, mix_bbox, soft_label, indices = self.dataset.load_batch_config(str(possibly_batched_index[:3]))
                else: # soft label pruning only
                    mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(str(possibly_batched_index[:3]))
            else:
                mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])

        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]

        if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
            # NOTE: mix_index, mix_lam, mix_bbox can be None
            mix_index_cpu = mix_index.cpu() if mix_index is not None else None
            if hasattr(self.dataset, "label_quantization") and self.dataset.label_quantization:
                if self.dataset.label_quantization[:3] == 'MRS':
                    return self.collate_fn(data), mix_index_cpu, mix_lam, mix_bbox, soft_label.cpu(), indices.cpu(), sample_min.cpu()
                else:   # MR
                    return self.collate_fn(data), mix_index_cpu, mix_lam, mix_bbox, soft_label.cpu(), indices.cpu()
            else:
                return self.collate_fn(data), mix_index_cpu, mix_lam, mix_bbox, soft_label.cpu()
        else:
            return self.collate_fn(data)
