import jittor as jt
from jittor.dataset import Dataset

class TrainDataset(Dataset):
    
    def __init__(self, dataset):
        super().__init__(dataset)
        
        print(f'\nInitializing TrainDataset: {dataset}')
        self.collate_fn = None
        self.set_attrs(total_len=len(dataset))
        self.rows = dataset
    
    def __getitem__(self, index):
        return self.rows[index]

    def collate_batch(self, batch):
        return self.collate_fn(batch)