class CachedDataSet:
    '''
    provides caching dataset items in memory
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = [None] * len(dataset)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            self.cache[index] = self.dataset[index]
        return self.cache[index]