import numpy as np
import sklearn.metrics
import torch
import torch.utils.data
import torch.nn as nn
from training.train_logger import Dataset, TrainLogger
from training.model import ConvLSTM as Model
from training.train_utils import create_dataloader, device, lr_scheduler, load_model_config, calculate_metrics
from schemas import ProcessedName, Metrics, Scores
from logger import logger


torch.manual_seed(0)
np.random.seed(0)


torch.set_warn_always(False)

class TrainSetup:
    def __init__(self, model_id: str, base_model_name: str, classes: list[str], dataset: list[ProcessedName]):
        self.model_id = model_id
        self.base_model_name = base_model_name
        self.model_config = load_model_config(base_model_name)

        self.dataset = dataset
        self.test_size = self.model_config["test-size"]
        self.classes = classes
        self.total_classes = len(classes)

        self.epochs = self.model_config["epochs"]
        self.batch_size = self.model_config["batch-size"]
        self.hidden_size = self.model_config["hidden-size"]
        self.rnn_layers = self.model_config["rnn-layers"]
        self.dropout_chance = self.model_config["dropout-chance"]
        self.embedding_size = self.model_config["embedding-size"]
        self.augmentation = self.model_config["augmentation"]
        self.weight_decay = self.model_config["weight-decay"]
        self.lr = self.model_config["lr-schedule"][0]
        self.lr_decay_rate = self.model_config["lr-schedule"][1]
        self.lr_decay_intervall = self.model_config["lr-schedule"][2]

        self.kernel_size = self.model_config["kernel-size"]
        self.cnn_out_dim = self.model_config["cnn-out-dim"]

        self.train_set, self.validation_set, self.test_set = create_dataloader(
            dataset=self.dataset,
            test_size=self.test_size,
            val_size=self.test_size,
            batch_size=self.batch_size,
            class_amount=self.total_classes,
            augmentation=self.augmentation
        )

        self.model = Model(
            class_amount=self.total_classes,
            hidden_size=self.hidden_size,
            layers=self.rnn_layers,
            dropout_chance=self.dropout_chance,
            embedding_size=self.embedding_size,
            kernel_size=self.kernel_size,
            cnn_out_dim=self.cnn_out_dim
        ).to(device=device)

        self.train_logger = TrainLogger()

    def _validate(self, model, dataset):
        validation_dataset = dataset

        criterion = nn.NLLLoss()
        losses = []
        total_targets, total_predictions = [], []

        for names, targets in validation_dataset:
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
        accuracy, f1_scores, precision_scores, recall_scores = calculate_metrics(total_targets, total_predictions)
    	
        return Metrics(round(accuracy, 2), loss, scores=Scores(f1_scores, precision_scores, recall_scores))

    def train(self):
        logger.info(f"Training model with id {self.model_id}.")
        logger.info("")
        
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        iterations = 0
        for epoch in range(1, (self.epochs + 1)):

            total_train_targets, total_train_predictions = [], []
            epoch_train_loss = []
            for names, targets in self.train_set:
                optimizer.zero_grad()

                names = names.to(device=device)
                targets = targets.to(device=device)
                predictions = self.model.train()(names)

                loss = criterion(predictions, targets.squeeze())
                loss.backward()

                lr_scheduler(optimizer, iterations, decay_rate=self.lr_decay_rate, decay_intervall=self.lr_decay_intervall)
                optimizer.step()

                epoch_train_loss.append(loss.item())

                validated_predictions = self.model.eval()(names)
                for i in range(validated_predictions.size()[0]): 
                    total_train_targets.append(targets[i].cpu().detach().numpy()[0])
                    validated_prediction = validated_predictions[i].cpu().detach().numpy()
                    total_train_predictions.append(list(validated_prediction).index(max(validated_prediction)))
                
                iterations += 1

                if iterations % self.lr_decay_intervall == 0:
                    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * self.lr_decay_rate

            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = 100 * sklearn.metrics.accuracy_score(total_train_targets, total_train_predictions)
            epoch_val_metrics = self._validate(self.model, self.validation_set)

            self.train_logger.save_epoch(Metrics(
                accuracy=round(epoch_train_accuracy, 2),
                loss=epoch_train_loss,
                scores=None
            ), Dataset.TRAIN)
            self.train_logger.save_epoch(epoch_val_metrics, Dataset.VALIDATION)
            self.train_logger.log_epoch(epoch)
        
        logger.info("")
        logger.info(f"Training finished.")
        
    def test(self):
        result_metrics = self._validate(self.model, self.test_set)
        self.train_logger.save_test_evaluation(result_metrics)

        logger.info(f"Evaluated model with id {self.model_id}.")

    def get_logs(self) -> dict:
        return self.train_logger.to_dict()

    def get_results(self) -> tuple[Metrics, dict]:
        return self.train_logger.results, self.model.state_dict()
    

