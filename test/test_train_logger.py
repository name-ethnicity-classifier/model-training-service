import logging
import pytest
from training.train_logger import TrainLogger, Metrics, Scores, Dataset


@pytest.fixture
def logger():
    return TrainLogger()


@pytest.mark.it("should initialize logs with the correct structure")
def test_logger_initialization(logger):
    assert logger.train_history.train == []
    assert logger.train_history.validation == []
    assert logger.results is None


@pytest.mark.it("should save training epoch metrics correctly")
def test_save_epoch_train(logger):
    scores = Scores(f1=0.93, precision=0.92, recall=0.91)
    metrics = Metrics(accuracy=0.95, loss=0.1, scores=scores)
    logger.save_epoch(metrics, dataset=Dataset.TRAIN)

    assert len(logger.train_history.train) == 1
    log_entry = logger.train_history.train[0]

    assert log_entry.accuracy == 0.95
    assert log_entry.scores.f1 == 0.93
    assert log_entry.scores.precision == 0.92
    assert log_entry.scores.recall == 0.91
    assert log_entry.loss == 0.1


@pytest.mark.it("should save validation epoch metrics correctly")
def test_save_epoch_validation(logger):
    scores = Scores(f1=0.83, precision=0.82, recall=0.81)
    metrics = Metrics(accuracy=0.85, loss=0.2, scores=scores)
    logger.save_epoch(metrics, dataset=Dataset.VALIDATION)

    assert len(logger.train_history.validation) == 1
    log_entry = logger.train_history.validation[0]

    assert log_entry.accuracy == 0.85
    assert log_entry.scores.f1 == 0.83
    assert log_entry.scores.precision == 0.82
    assert log_entry.scores.recall == 0.81
    assert log_entry.loss == 0.2


@pytest.mark.it("should save test evaluation correctly")
def test_save_test_evaluation(logger):
    scores = Scores(f1=0.98, precision=0.97, recall=0.96)
    metrics = Metrics(accuracy=0.99, scores=scores)
    logger.save_test_evaluation(metrics)

    assert logger.results.accuracy == 0.99
    assert logger.results.scores.f1 == 0.98
    assert logger.results.scores.precision == 0.97
    assert logger.results.scores.recall == 0.96
    assert logger.results.loss is None


@pytest.mark.it("should save multiple epochs and validate list lengths")
def test_save_multiple_epochs(logger):
    epochs = 4
    for _ in range(epochs):
        scores = Scores(f1=0.89, precision=0.88, recall=0.87)
        metrics = Metrics(accuracy=0.9, loss=0.1, scores=scores)
        logger.save_epoch(metrics, Dataset.TRAIN)
        logger.save_epoch(metrics, Dataset.VALIDATION)

    assert len(logger.train_history.train) == epochs
    assert len(logger.train_history.validation) == epochs


@pytest.mark.it("should log a specific epoch correctly")
def test_log_specific_epoch(logger, caplog):
    epochs = 3
    for i in range(epochs):
        scores = Scores(f1=0.1, precision=0.1, recall=0.1)
        logger.save_epoch(Metrics(accuracy=i, loss=i, scores=scores), Dataset.TRAIN)
        logger.save_epoch(Metrics(accuracy=i * 2, loss=i * 2, scores=scores), Dataset.VALIDATION)

    with caplog.at_level(logging.INFO):
        logger.log_epoch(1)

    expected_output = "Epoch: 1, Train Acc: 0, Train Loss: 0, Val Acc: 0, Val Loss: 0"
    assert expected_output in caplog.text


@pytest.mark.it("should serialize logs to valid JSON")
def test_json_serialize(logger):
    scores = Scores(f1=0.89, precision=0.88, recall=0.87)
    metrics = Metrics(accuracy=0.90, loss=0.15, scores=scores)
    logger.save_epoch(metrics, Dataset.TRAIN)

    json_output = logger.to_dict()

    assert isinstance(json_output, dict)
    assert "accuracy" in str(json_output)
    assert "loss" in str(json_output)
