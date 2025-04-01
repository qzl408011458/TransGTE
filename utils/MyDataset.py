from torch.utils.data.dataset import Dataset
import torch

class Trip_parDataset(Dataset):
    def __init__(self, x, if_yg=False):
        # list[[], [], []...]
        self.gtrj = x['gtrj']
        self.trj = x['trj']

        self.w = x['w']
        self.h = x['h']
        self.dest = x['dest']
        self.dest_g = x['dest_g']
        self.if_yg = if_yg


    def __len__(self):
        return len(self.dest)

    def __getitem__(self, i):
        X = (self.gtrj[i], self.trj[i], self.w[i], self.h[i])
        Y = self.dest[i]

        if self.if_yg:
            Y = (self.dest[i], self.dest_g[i])

        return X, Y
