import random
import numpy as np
import torch
import torch.utils.data as data_utils


torch.set_default_tensor_type(torch.DoubleTensor)


class NILMDataloader():
    def __init__(self, args, dataset, bert=False):
        self.args = args
        self.mask_prob = args.mask_prob
        self.batch_size = args.batch_size

        if bert:
            self.train_dataset, self.val_dataset = dataset.get_bert_datasets(mask_prob=self.mask_prob)
        else:
            self.train_dataset, self.val_dataset = dataset.get_datasets()

    @classmethod
    def code(cls):
        return 'dataloader'

    def get_dataloaders(self):
        train_loader = self._get_loader(self.train_dataset)
        val_loader = self._get_loader(self.val_dataset)
        return train_loader, val_loader

    def _get_loader(self, dataset):
        dataloader = data_utils.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class NILMDataset(data_utils.Dataset):
    def __init__(self, x, y, status, window_size=480, stride=30):
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min(
            (len(self.x), index * self.stride + self.window_size))
        x = self.padding_seqs(self.x[start_index: end_index])
        y = self.padding_seqs(self.y[start_index: end_index])
        status = self.padding_seqs(self.status[start_index: end_index])
        return torch.tensor(x), torch.tensor(y), torch.tensor(status)

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array


class BERTDataset(data_utils.Dataset):
    def __init__(self, x, y, status, window_size=480, stride=30, mask_prob=0.2):
        self.x = x
        self.y = y
        self.status = status
        self.window_size = window_size
        self.stride = stride
        self.mask_prob = mask_prob
        self.columns = y.shape[1]

    def __len__(self):
        return int(np.ceil((len(self.x) - self.window_size) / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min(
            (len(self.x), index * self.stride + self.window_size))
        x = self.padding_seqs(self.x[start_index: end_index])
        y = self.padding_seqs(self.y[start_index: end_index])
        status = self.padding_seqs(self.status[start_index: end_index])

        tokens = []
        labels = []
        on_offs = []
        for i in range(len(x)):
            prob = random.random()
            if prob < self.mask_prob:
                prob = random.random()
                if prob < 0.8:
                    tokens.append(-1)
                elif prob < 0.9:
                    tokens.append(np.random.normal())
                else:
                    tokens.append(x[i])

                labels.append(y[i])
                on_offs.append(status[i])
            else:
                tokens.append(x[i])
                temp = np.array([-1] * self.columns)
                labels.append(temp)
                on_offs.append(temp)
        
        return torch.tensor(tokens), torch.tensor(labels), torch.tensor(on_offs)

    def padding_seqs(self, in_array):
        if len(in_array) == self.window_size:
            return in_array
        try:
            out_array = np.zeros((self.window_size, in_array.shape[1]))
        except:
            out_array = np.zeros(self.window_size)

        length = len(in_array)
        out_array[:length] = in_array
        return out_array
