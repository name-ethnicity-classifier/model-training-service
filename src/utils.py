import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import pickle
from termcolor import colored
import json
import random
import boto3
from dotenv import load_dotenv
import os
import pickle

from dataset import NameEthnicityDataset

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


def create_dataloader(dataset_path: str, test_size: float=0.01, val_size: float=0.01, batch_size: int=32, class_amount: int=10, augmentation: float=0.0):
    """
    Creates three dataloaders (train, test, validation)

    :param str dataset_path: path to dataset
    :param float test_size/val_size: test-/validation-percentage of dataset
    :param int batch_size: batch-size
    :return torch.Dataloader: train-, test- and val-dataloader
    """

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

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


def show_progress(epochs: int, epoch: int, train_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float, color: bool=False):
    """ 
    Pretty-prints current training stats
    
    :param int epochs: amount of total epochs
    :param int epoch: current epoch
    :param float train_loss/train_accuracy: train-loss, train-accuracy
    :param float val_loss/val_accuracy: validation accuracy/loss
    :return None
    """
    if color:
        epochs = colored(epoch, "cyan", attrs=["bold"]) + colored("/", "cyan", attrs=["bold"]) + colored(epochs, "cyan", attrs=["bold"])
        train_accuracy = colored(round(train_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
        train_loss = colored(round(train_loss, 6), "cyan", attrs=["bold"])
        val_accuracy = colored(round(val_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
        val_loss = colored(round(val_loss, 6), "cyan", attrs=["bold"])
    else:
        epochs = str(epoch) + "/" + str(epochs)
        train_accuracy = str(round(train_accuracy, 4)) + "%"
        train_loss = str(round(train_loss, 6))
        val_accuracy = str(round(val_accuracy, 4)) + "%"
        val_loss = str(round(val_loss, 6))

    print("epoch {} train_loss: {} - train_acc: {} - val_loss: {} - val_acc: {}".format(epochs, train_loss, train_accuracy, val_loss, val_accuracy), "\n")

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


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path: str, content: dict) -> None:
    with open(file_path, "w") as f:
            json.dump(content, f, indent=4)



def create_s3_client():
    load_dotenv()

    return boto3.client("s3",
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_ACCESS_KEY"),
        endpoint_url=f"{os.environ.get('MINIO_HOST')}:{os.environ.get('MINIO_PORT')}"
    )


def s3_upload(bucket_name: str, body: str, object_key: str):
    s3_client = create_s3_client()

    s3_client.put_object(
        Body=body,
        Bucket=bucket_name,
        Key=object_key,
    )
    
