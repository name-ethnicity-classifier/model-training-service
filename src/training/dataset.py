import torch
import numpy as np


class NameEthnicityDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=[], class_amount: int=10, augmentation: float=0.0):
        self.dataset = dataset
        self.class_amount = class_amount

        self.augmentation = augmentation
        self.seperate_dataset = self.dataset.copy()
        np.random.shuffle(self.seperate_dataset)

        self.augmentation_list = [0.0 for _ in range(class_amount)]

    def _preprocess_targets(self, int_representation: int, one_hot: bool=True) -> list:
        """
        Create one-hot encoding of the target

        :param int int_representation: class of sample
        :return list: ie. int_representation = 2 -> [0, 0, 1, ..., 0]
        """

        if one_hot:
            one_hot_target = np.zeros((self.class_amount))
            one_hot_target[int_representation] = 1

            return one_hot_target
        else:
            return [int_representation]

    def _name_switch(self, name: list, class_: int, chance: float=0.3) -> list:
        """ Switches first and last name part of the name with a random name of the same nationality
            TODO: Has to be reimplemented
        """

        return name

    def _split_name(self, int_name: list) -> list:
        try:
            str_index_name = "".join([str(e) + " " for e in int_name])
            str_index_name_split = str_index_name.split("27", 1)

            pre_int_name, sur_int_name = str_index_name_split[0], str_index_name_split[1]
            pre_int_name = [int(e) for e in pre_int_name.split() if e.isdigit()]
            sur_int_name = [int(e) for e in sur_int_name.split() if e.isdigit()]

            return pre_int_name, sur_int_name
            
        except:
            # the case, when the name is only one word (no first-/ sur-name)
            return int_name, int_name

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample, target = self.dataset[idx][1], self.dataset[idx][0]

        int_name = [e + 1 for e in sample]
        target = self._preprocess_targets(target, one_hot=False)

        augmentation_chance = self.augmentation
        if augmentation_chance > 0.0:
            int_name = self._name_switch(int_name, target, chance=augmentation_chance)
        
        return torch.Tensor(int_name), torch.Tensor(target).type(torch.LongTensor)

    def __len__(self):       
        return len(self.dataset)

