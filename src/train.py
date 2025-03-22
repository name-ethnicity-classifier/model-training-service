
from tqdm import tqdm
import numpy as np
import os
import json
import sklearn.metrics
import shutil

import torch
import torch.utils.data
import torch.nn as nn

from src.model import ConvLSTM as Model
from src.utils import create_dataloader, show_progress, device, lr_scheduler, write_json
import src.experiment_manager as xman


torch.manual_seed(0)
np.random.seed(0)


class TrainSetup:
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.model_name = model_config["model-name"]

        self.log_path = f"./logs/{self.model_name}"
        self.output_path = f"./outputs/{self.model_name}"

        self.model_file = f"{self.log_path}/model.pt"

        # dataset parameters
        self.dataset_folder = f"./datasets/preprocessed_datasets/{model_config['dataset-name']}"
        self.dataset_path = f"{self.dataset_folder}/dataset.pickle"
        self.test_size = model_config["test-size"]

        with open(f"{self.dataset_folder}/nationalities.json", "r") as f: 
            self.classes = json.load(f) 
            self.total_classes = len(self.classes)

        # hyperparameters
        self.epochs = model_config["epochs"]
        self.batch_size = model_config["batch-size"]
        self.hidden_size = model_config["hidden-size"]
        self.rnn_layers = model_config["rnn-layers"]
        self.dropout_chance = model_config["dropout-chance"]
        self.embedding_size = model_config["embedding-size"]
        self.augmentation = model_config["augmentation"]

        # unpack learning-rate parameters (idx 0: current lr, idx 1: decay rate, idx 2: decay intervall in iterations)
        self.lr = model_config["lr-schedule"][0]
        self.lr_decay_rate = model_config["lr-schedule"][1]
        self.lr_decay_intervall = model_config["lr-schedule"][2]

        # unpack cnn parameters (idx 0: amount of layers, idx 1: kernel size, idx 2: list of feature map dimensions)
        self.kernel_size = model_config["cnn-parameters"][0]
        self.cnn_out_dim = model_config["cnn-parameters"][1]

        # dataloaders for train, test and validation
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=self.test_size, val_size=self.test_size, \
                                                                               batch_size=self.batch_size, class_amount=self.total_classes, augmentation=self.augmentation)

        # resume training boolean
        self.continue_ = model_config["resume"]

        # initialize xman experiment manager
        self.xmanager = xman.ExperimentManager(log_path=self.log_path, continue_=self.continue_)
        self.xmanager.init(
            optimizer="Adam", 
            loss_function="NLLLoss", 
            learning_rate=self.lr, 
            custom_parameters=model_config
        )

    def _validate(self, model, dataset):
        validation_dataset = dataset

        criterion = nn.NLLLoss()
        losses = []
        total_targets, total_predictions = [], []

        for names, targets, _ in tqdm(validation_dataset, desc="validating", ncols=100):
            names = names.to(device=device)
            targets = targets.to(device=device)

            predictions = model.eval()(names)
            loss = criterion(predictions, targets.squeeze())
            losses.append(loss.item())

            for i in range(predictions.size()[0]):
                target_index = targets[i].cpu().detach().numpy()[0]

                prediction = predictions[i].cpu().detach().numpy()
                prediction_index = list(prediction).index(max(prediction))

                total_targets.append(target_index)
                total_predictions.append(prediction_index)

        loss = np.mean(losses)

        accuracy = 100 * sklearn.metrics.accuracy_score(total_targets, total_predictions)
        precision_scores = sklearn.metrics.precision_score(total_targets, total_predictions, average=None)
        recall_scores = sklearn.metrics.recall_score(total_targets, total_predictions, average=None)
        f1_scores = sklearn.metrics.f1_score(total_targets, total_predictions, average=None)
    	
        return loss, accuracy, (precision_scores, recall_scores, f1_scores)

    def train(self):
        model = Model(
            class_amount=self.total_classes,
            hidden_size=self.hidden_size,
            layers=self.rnn_layers,
            dropout_chance=self.dropout_chance,
            embedding_size=self.embedding_size,
            kernel_size=self.kernel_size,
            cnn_out_dim=self.cnn_out_dim
        ).to(device=device)

        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        iterations = 0
        for epoch in range(1, (self.epochs + 1)):

            total_train_targets, total_train_predictions = [], []
            epoch_train_loss = []
            for names, targets, _ in tqdm(self.train_set, desc="epoch", ncols=100):
                optimizer.zero_grad()

                names = names.to(device=device)
                targets = targets.to(device=device)
                predictions = model.train()(names)

                loss = criterion(predictions, targets.squeeze())
                loss.backward()

                lr_scheduler(optimizer, iterations, decay_rate=self.lr_decay_rate, decay_intervall=self.lr_decay_intervall)
                optimizer.step()

                epoch_train_loss.append(loss.item())

                validated_predictions = model.eval()(names)
                for i in range(validated_predictions.size()[0]): 
                    total_train_targets.append(targets[i].cpu().detach().numpy()[0])
                    validated_prediction = validated_predictions[i].cpu().detach().numpy()
                    total_train_predictions.append(list(validated_prediction).index(max(validated_prediction)))
                
                iterations += 1

                # lr decay
                if iterations % self.lr_decay_intervall == 0:
                    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * self.lr_decay_rate

            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = 100 * sklearn.metrics.accuracy_score(total_train_targets, total_train_predictions)
            epoch_val_loss, epoch_val_accuracy, _scores = self._validate(model, self.validation_set)

            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy)

    def test(self):
        model = Model(
            class_amount=self.total_classes,
            hidden_size=self.hidden_size,
            layers=self.rnn_layers,
            dropout_chance=0.0,
            embedding_size=self.embedding_size,
            kernel_size=self.kernel_size,
            cnn_out_dim=self.cnn_out_dim
        ).to(device=device)

        model.load_state_dict(torch.load(self.model_file))
        _, accuracy, scores = self._validate(model, self.test_set)
        
        precisions, recalls, f1_scores = scores
        print("\n\ntest accuracy:", accuracy)
        print("precision of every class:", precisions)
        print("recall of every class:", recalls)
        print("f1-score of every class:", f1_scores)

        self.save_model_configuration(accuracy, [precisions, recalls, f1_scores])
     
    def save_model_configuration(self, accuracy: float, scores: list):
        # TODO save to S3

        if os.path.exists(self.output_path):
            print("\nError: The directory '{}' does already exist! Reinitializing.\n".format(self.output_path))
            shutil.rmtree(self.output_path)

        os.mkdir(self.output_path)
        write_json(f"{self.output_path}/results.json", 
            {
                "accuracy": accuracy,
                "precision-scores": scores[0],
                "recall-scores": scores[1],
                "f1-scores": scores[2]
            }
        )
        write_json(f"{self.output_path}/config.json", self.model_config)

        shutil.copyfile(f"{self.log_path}/model.pt", f"{self.output_path}/model.pt")
        shutil.copyfile(f"{self.dataset_folder}/nationalities.json", f"{self.output_path}/nationalities.json")
