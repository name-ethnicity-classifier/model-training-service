import numpy as np
import os
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import json
import random
from training.dataset import NameEthnicityDataset
from schemas import ProcessedName
from s3 import S3Handler
from config import Environment, config


torch.manual_seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def custom_collate(batch):
    """
    Adds custom dataloader feature: batch padding for the sample-batch (the batch containing the one-hot-enc. names)

    :param batch: three batches -> non-padded sample-batch, target-batch, non-padded sample-batch (again)
    :return torch.Tensor: padded sample-batch, target-batch, non-padded sample-batch
    """

    sample_batch, target_batch, non_padded_batch = [], [], []
    for sample, target, non_padded_sample in batch:

        sample_batch.append(sample)
        target_batch.append(target)

        non_padded_batch.append(non_padded_sample)

    padded_batch = pad_sequence(sample_batch, batch_first=True)
    padded_to = list(padded_batch.size())[1]
    padded_batch = padded_batch.reshape(len(sample_batch), padded_to, 1)

    return padded_batch, torch.cat(target_batch, dim=0).reshape(len(sample_batch), target_batch[0].size(0)), non_padded_batch


def create_dataloader(dataset: list[ProcessedName], test_size: float=0.01, val_size: float=0.01, batch_size: int=32, class_amount: int=10, augmentation: float=0.0):
    """
    Creates three dataloaders (train, test, validation)

    :param str dataset: preprocessed dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """
    
    test_size = int(np.round(len(dataset)*test_size))
    val_size = int(np.round(len(dataset)*val_size))

    train_set, test_set, validation_set = dataset[(test_size+val_size):], dataset[:test_size], dataset[test_size:(test_size+val_size)]

    train_set = NameEthnicityDataset(dataset=train_set, class_amount=class_amount, augmentation=augmentation)
    test_set = NameEthnicityDataset(dataset=test_set, class_amount=class_amount, augmentation=0.0)
    val_set = NameEthnicityDataset(dataset=validation_set, class_amount=class_amount, augmentation=0.0)

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=int(batch_size),
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate

    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=int(batch_size),
        num_workers=0,
        shuffle=True,
        collate_fn=custom_collate
    )

    return train_dataloader, val_dataloader, test_dataloader


def lr_scheduler(optimizer: torch.optim, current_iteration: int=0, warmup_iterations: int=0, lr_end: float=0.001, decay_rate: float=0.99, decay_intervall: int=100, verbose: bool=False) -> None:
    current_iteration += 1
    current_lr = optimizer.param_groups[0]["lr"]

    if current_iteration <= warmup_iterations:
        optimizer.param_groups[0]["lr"] = (current_iteration * lr_end) / warmup_iterations
        if verbose: print(" WARMUP", optimizer.param_groups[0]["lr"])

    elif current_iteration > warmup_iterations and current_iteration % decay_intervall == 0:
        optimizer.param_groups[0]["lr"] = current_lr * decay_rate
        if verbose: print(" DECAY", optimizer.param_groups[0]["lr"])
    else:
        pass


def load_model_config(model_config_name: str) -> dict:
    model_config_path = f"model-configs/{model_config_name}.json"
    if config.environment == Environment.DEV:
        return load_json(f"dev-data/{model_config_path}")
    
    return S3Handler.get(config.base_data_bucket, model_config_path)


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path: str, content: dict) -> None:
    with open(file_path, "w") as f:
        json.dump(content, f, indent=4)
