from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch



class CustomDataSet(Dataset):
    """Load data under folders"""
    def __init__(self, args, train=True):
        self.main_dir = args.main_dir 
        self.task = args.task 

        if train:
            data_path = self.main_dir + self.task + "/data_train.npy"
            label_path = self.main_dir + self.task + "/label_train.npy"
        else:
            data_path = self.main_dir + self.task + "/data_test.npy"
            label_path = self.main_dir + self.task + "/label_test.npy"
        
        self.data = torch.from_numpy(np.load(data_path))
        self.label = torch.from_numpy(np.load(label_path)).to(torch.long) # in cls task, (N,), in seg task, (N, 10000), N is the number of objects
        

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def get_data_loader(args, train=True):
    """
    Creates training and test data loaders
    """
    dataset = CustomDataSet(args=args, train=train)
    dloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers)
    

    return dloader