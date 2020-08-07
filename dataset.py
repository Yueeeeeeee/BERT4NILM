from abc import *
from config import *
from dataloader import *

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch.utils.data as data_utils


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args, stats=None):
        self.house_indicies = args.house_indicies
        self.appliance_names = args.appliance_names
        self.normalize = args.normalize
        self.sampling = args.sampling
        self.cutoff = [args.cutoff[i]
                       for i in ['aggregate'] + self.appliance_names]

        self.threshold = [args.threshold[i] for i in self.appliance_names]
        self.min_on = [args.min_on[i] for i in self.appliance_names]
        self.min_off = [args.min_off[i] for i in self.appliance_names]

        self.val_size = args.validation_size
        self.window_size = args.window_size
        self.window_stride = args.window_stride

        self.x, self.y = self.load_data()
        self.status = self.compute_status(self.y)
        print('Appliance:', self.appliance_names)
        print('Sum of ons:', np.sum(self.status, axis=0))
        print('Total length:', self.status.shape[0])

        if stats is None:
            self.x_mean = np.mean(self.x, axis=0)
            self.x_std = np.std(self.x, axis=0)
        else: 
            self.x_mean, self.x_std = stats

        self.x = (self.x - self.x_mean) / self.x_std

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def load_data(self):
        pass

    def get_data(self):
        return self.x, self.y, self.status

    def get_original_data(self):
        x_org = self.x * self.x_std + self.x_mean
        return x_org, self.y, self.status

    def get_mean_std(self):
        return self.x_mean, self.x_std

    def compute_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        for i in range(columns):
            initial_status = data[:, i] >= self.threshold[i]
            status_diff = np.diff(initial_status)
            events_idx = status_diff.nonzero()

            events_idx = np.array(events_idx).squeeze()
            events_idx += 1

            if initial_status[0]:
                events_idx = np.insert(events_idx, 0, 0)

            if initial_status[-1]:
                events_idx = np.insert(
                    events_idx, events_idx.size, initial_status.size)

            events_idx = events_idx.reshape((-1, 2))
            on_events = events_idx[:, 0].copy()
            off_events = events_idx[:, 1].copy()
            assert len(on_events) == len(off_events)

            if len(on_events) > 0:
                off_duration = on_events[1:] - off_events[:-1]
                off_duration = np.insert(off_duration, 0, 1000)
                on_events = on_events[off_duration > self.min_off[i]]
                off_events = off_events[np.roll(
                    off_duration, -1) > self.min_off[i]]

                on_duration = off_events - on_events
                on_events = on_events[on_duration >= self.min_on[i]]
                off_events = off_events[on_duration >= self.min_on[i]]
                assert len(on_events) == len(off_events)

            temp_status = data[:, i].copy()
            temp_status[:] = 0
            for on, off in zip(on_events, off_events):
                temp_status[on: off] = 1
            status[:, i] = temp_status

        return status

    def get_status(self):
        return self.status

    def get_datasets(self):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = NILMDataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                            self.window_size, self.window_stride)
        return train, val

    def get_bert_datasets(self, mask_prob=0.25):
        val_end = int(self.val_size * len(self.x))
        val = NILMDataset(self.x[:val_end], self.y[:val_end], self.status[:val_end],
                          self.window_size, self.window_size)
        train = BERTDataset(self.x[val_end:], self.y[val_end:], self.status[val_end:],
                            self.window_size, self.window_stride, mask_prob=mask_prob)
        return train, val

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())


class REDD_LF_Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'redd_lf'

    @classmethod
    def _if_data_exists(self):
        folder = Path(RAW_DATASET_ROOT_FOLDER).joinpath(self.code())
        first_file = folder.joinpath('house_1', 'channel_1.dat')
        if first_file.is_file():
            return True
        return False

    def load_data(self):
        for appliance in self.appliance_names:
            assert appliance in ['dishwasher',
                                 'refrigerator', 'microwave', 'washer_dryer']

        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5, 6]

        if not self.cutoff:
            self.cutoff = [6000] * (len(self.appliance_names) + 1)

        if not self._if_data_exists():
            print('Please download, unzip and move data into',
                  self._get_folder_path())
            raise FileNotFoundError

        else:
            directory = self._get_folder_path()

            for house_id in self.house_indicies:
                house_folder = directory.joinpath('house_' + str(house_id))
                house_label = pd.read_csv(house_folder.joinpath(
                    'labels.dat'), sep=' ', header=None)

                main_1 = pd.read_csv(house_folder.joinpath(
                    'channel_1.dat'), sep=' ', header=None)
                main_2 = pd.read_csv(house_folder.joinpath(
                    'channel_2.dat'), sep=' ', header=None)
                house_data = pd.merge(main_1, main_2, how='inner', on=0)
                house_data.iloc[:, 1] = house_data.iloc[:,
                                                        1] + house_data.iloc[:, 2]
                house_data = house_data.iloc[:, 0: 2]

                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)

                for appliance in self.appliance_names:
                    data_found = False
                    for i in range(len(appliance_list)):
                        if appliance_list[i] == appliance:
                            app_index_dict[appliance].append(i + 1)
                            data_found = True

                    if not data_found:
                        app_index_dict[appliance].append(-1)

                if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                    self.house_indicies.remove(house_id)
                    continue

                for appliance in self.appliance_names:
                    if app_index_dict[appliance][0] == -1:
                        temp_values = house_data.copy().iloc[:, 1]
                        temp_values[:] = 0
                        temp_data = house_data.copy().iloc[:, :2]
                        temp_data.iloc[:, 1] = temp_values
                    else:
                        temp_data = pd.read_csv(house_folder.joinpath(
                            'channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)

                    if len(app_index_dict[appliance]) > 1:
                        for idx in app_index_dict[appliance][1:]:
                            temp_data_ = pd.read_csv(house_folder.joinpath(
                                'channel_' + str(idx) + '.dat'), sep=' ', header=None)
                            temp_data = pd.merge(
                                temp_data, temp_data_, how='inner', on=0)
                            temp_data.iloc[:, 1] = temp_data.iloc[:,
                                                                  1] + temp_data.iloc[:, 2]
                            temp_data = temp_data.iloc[:, 0: 2]

                    house_data = pd.merge(
                        house_data, temp_data, how='inner', on=0)

                house_data.iloc[:, 0] = pd.to_datetime(
                    house_data.iloc[:, 0], unit='s')
                house_data.columns = ['time', 'aggregate'] + \
                    [i for i in self.appliance_names]
                house_data = house_data.set_index('time')
                house_data = house_data.resample(self.sampling).mean().fillna(
                    method='ffill', limit=30)

                if house_id == self.house_indicies[0]:
                    entire_data = house_data
                else:
                    entire_data = entire_data.append(
                        house_data, ignore_index=True)

                entire_data = entire_data.dropna().copy()
                entire_data = entire_data[entire_data['aggregate'] > 0]
                entire_data[entire_data < 5] = 0
                entire_data = entire_data.clip(
                    [0] * len(entire_data.columns), self.cutoff, axis=1)

            return entire_data.values[:, 0], entire_data.values[:, 1:]


class UK_DALE_Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'uk_dale'

    @classmethod
    def _if_data_exists(self):
        folder = Path(RAW_DATASET_ROOT_FOLDER).joinpath(self.code())
        first_file = folder.joinpath('house_1', 'channel_1.dat')
        if first_file.is_file():
            return True
        return False

    def load_data(self):
        for appliance in self.appliance_names:
            assert appliance in ['dishwasher', 'fridge',
                                 'microwave', 'washing_machine', 'kettle']

        for house_id in self.house_indicies:
            assert house_id in [1, 2, 3, 4, 5]

        if not self.cutoff:
            self.cutoff = [6000] * (len(self.appliance_names) + 1)

        if not self._if_data_exists():
            print('Please download, unzip and move data into',
                  self._get_folder_path())
            raise FileNotFoundError

        else:
            directory = self._get_folder_path()

            for house_id in self.house_indicies:
                house_folder = directory.joinpath('house_' + str(house_id))
                house_label = pd.read_csv(house_folder.joinpath(
                    'labels.dat'), sep=' ', header=None)

                house_data = pd.read_csv(house_folder.joinpath(
                    'channel_1.dat'), sep=' ', header=None)
                house_data.iloc[:, 0] = pd.to_datetime(
                    house_data.iloc[:, 0], unit='s')
                house_data.columns = ['time', 'aggregate']
                house_data = house_data.set_index('time')
                house_data = house_data.resample(self.sampling).mean().fillna(
                    method='ffill', limit=30)

                appliance_list = house_label.iloc[:, 1].values
                app_index_dict = defaultdict(list)

                for appliance in self.appliance_names:
                    data_found = False
                    for i in range(len(appliance_list)):
                        if appliance_list[i] == appliance:
                            app_index_dict[appliance].append(i + 1)
                            data_found = True

                    if not data_found:
                        app_index_dict[appliance].append(-1)

                if np.sum(list(app_index_dict.values())) == -len(self.appliance_names):
                    self.house_indicies.remove(house_id)
                    continue

                for appliance in self.appliance_names:
                    if app_index_dict[appliance][0] == -1:
                        house_data.insert(len(house_data.columns), appliance, np.zeros(len(house_data)))
                    else:
                        temp_data = pd.read_csv(house_folder.joinpath(
                            'channel_' + str(app_index_dict[appliance][0]) + '.dat'), sep=' ', header=None)
                        temp_data.iloc[:, 0] = pd.to_datetime(
                            temp_data.iloc[:, 0], unit='s')
                        temp_data.columns = ['time', appliance]
                        temp_data = temp_data.set_index('time')
                        temp_data = temp_data.resample(self.sampling).mean().fillna(
                            method='ffill', limit=30)
                        house_data = pd.merge(
                            house_data, temp_data, how='inner', on='time')

                if house_id == self.house_indicies[0]:
                    entire_data = house_data
                    if len(self.house_indicies) == 1:
                        entire_data = entire_data.reset_index(drop=True)
                else:
                    entire_data = entire_data.append(
                        house_data, ignore_index=True)

            entire_data = entire_data.dropna().copy()
            entire_data = entire_data[entire_data['aggregate'] > 0]
            entire_data[entire_data < 5] = 0
            entire_data = entire_data.clip(
                [0] * len(entire_data.columns), self.cutoff, axis=1)
            
        return entire_data.values[:, 0], entire_data.values[:, 1:]
