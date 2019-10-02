'''
Multithread pytorch dataloaders.
Usefull for due to improper work of standard DataLoader in a multiprocess (num_workers > 0) mode.
See issue https://github.com/pytorch/pytorch/issues/12831 etc.

Based on https://github.com/lopuhin/kaggle-imet-2019/blob/f1ec0827149a8218430a6884acf49c27ba6fcb1f/imet/utils.py#L35-L56
'''
import torch
from multiprocessing.pool import ThreadPool


class BatchThreadingDataLoader(torch.utils.data.DataLoader):
    '''
    Prepares each batch in a separate thread, including collate, so no processing is done in the caller thread.
    Requires map-style dataset (i.e. dataset with indexing) and automatic batching.
    Ignores worker_init_fn parameter.
    '''
    def __iter__(self):
        sample_iter = iter(self.batch_sampler)
        if self.num_workers == 0:
            for indices in sample_iter:
                yield self._get_batch(indices)
        else:
            prefetch = self.num_workers
            with ThreadPool(processes=self.num_workers) as pool:
                futures = []
                for indices in sample_iter:
                    futures.append(pool.apply_async(self._get_batch, args=(indices,)))
                    if len(futures) > prefetch:
                        yield futures.pop(0).get()
                for batch_futures in futures:
                    yield batch_futures.get()

    def _get_item(self, i):
        return self.dataset[i]

    def _get_batch(self, indices):
        return self.collate_fn([self._get_item(i) for i in indices])


class ThreadingDataLoader(torch.utils.data.DataLoader):
    '''
    Original dataset from https://github.com/lopuhin/kaggle-imet-2019
    Prepares each dataset item in a separate thread, but collation is processed in the caller thread.
    Requires map-style dataset (i.e. dataset with indexing) and automatic batching.
    Ignores worker_init_fn parameter.
    '''
    def __iter__(self):
        sample_iter = iter(self.batch_sampler)
        if self.num_workers == 0:
            for indices in sample_iter:
                yield self.collate_fn([self._get_item(i) for i in indices])
        else:
            prefetch = 1
            with ThreadPool(processes=self.num_workers) as pool:
                futures = []
                for indices in sample_iter:
                    futures.append([pool.apply_async(self._get_item, args=(i,))
                                    for i in indices])
                    if len(futures) > prefetch:
                        yield self.collate_fn([f.get() for f in futures.pop(0)])
                for batch_futures in futures:
                    yield self.collate_fn([f.get() for f in batch_futures])

    def _get_item(self, i):
        return self.dataset[i]
