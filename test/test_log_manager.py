import pytest
from log_manager import LogManager, EpochMetrics, BaseMetrics, Dataset


@pytest.fixture
def logger():
    return LogManager("model_123", "resnet50", ["cat", "dog", "rabbit"])


@pytest.mark.it("should initialize logs with the correct structure")
def test_logger_initialization(logger):
    assert logger.logs["model-id"] == "model_123"
    assert logger.logs["base-model"] == "resnet50"
    assert logger.logs["classes"] == ["cat", "dog", "rabbit"]
    assert logger.logs["train-history"]["train"] == []
    assert logger.logs["train-history"]["validation"] == []
    assert logger.logs["results"] is None


@pytest.mark.it("should save training epoch metrics correctly")
def test_save_epoch_train(logger):
    metrics = EpochMetrics(accuracy=0.95, f1=0.93, precision=0.92, recall=0.91, loss=0.1)
    logger.save_epoch(metrics, dataset=Dataset.TRAIN)

    assert len(logger.logs["train-history"]["train"]) == 1
    log_entry = logger.logs["train-history"]["train"][0]

    assert log_entry["accuracy"] == 0.95
    assert log_entry["scores"]["f1"] == 0.93
    assert log_entry["scores"]["precision"] == 0.92
    assert log_entry["scores"]["recall"] == 0.91
    assert log_entry["loss"] == 0.1


@pytest.mark.it("should save validation epoch metrics correctly")
def test_save_epoch_validation(logger):
    metrics = EpochMetrics(accuracy=0.85, f1=0.83, precision=0.82, recall=0.81, loss=0.2)
    logger.save_epoch(metrics, dataset=Dataset.VALIDATION)

    assert len(logger.logs["train-history"]["validation"]) == 1
    log_entry = logger.logs["train-history"]["validation"][0]

    assert log_entry["accuracy"] == 0.85
    assert log_entry["scores"]["f1"] == 0.83
    assert log_entry["scores"]["precision"] == 0.82
    assert log_entry["scores"]["recall"] == 0.81
    assert log_entry["loss"] == 0.2


@pytest.mark.it("should save test evaluation correctly")
def test_save_test_evaluation(logger):
    metrics = BaseMetrics(accuracy=0.99, f1=0.98, precision=0.97, recall=0.96)
    logger.save_test_evaluation(metrics)

    assert logger.logs["results"]["accuracy"] == 0.99
    assert logger.logs["results"]["scores"]["f1"] == 0.98
    assert logger.logs["results"]["scores"]["precision"] == 0.97
    assert logger.logs["results"]["scores"]["recall"] == 0.96
    assert "loss" not in logger.logs["results"]


@pytest.mark.it("should save multiple epochs and validate list lengths")
def test_save_multiple_epochs(logger):
    epochs = 4
    for _ in range(epochs):
        metrics = EpochMetrics(accuracy=0.9, f1=0.89, precision=0.88, recall=0.87, loss=0.1)
        logger.save_epoch(metrics, Dataset.TRAIN)
        logger.save_epoch(metrics, Dataset.VALIDATION)

    assert len(logger.logs["train-history"]["train"]) == epochs
    assert len(logger.logs["train-history"]["validation"]) == epochs


@pytest.mark.it("should log a specific epoch correctly")
def test_log_specific_epoch(logger, capsys):
    for i in range(3):
        logger.save_epoch(EpochMetrics(accuracy=i, f1=0.1, precision=0.1, recall=0.1, loss=i), Dataset.TRAIN)
        logger.save_epoch(EpochMetrics(accuracy=(i * 2), f1=0.1, precision=0.1, recall=0.1, loss=(i * 2)), Dataset.VALIDATION)

    logger.log_epoch(1)  # log the second epoch (index 1)

    captured = capsys.readouterr()
    expected_output = "epoch: 1, train-acc: 1, train-loss: 1, val-acc: 2, val-loss: 2\n"
    assert captured.out == expected_output


@pytest.mark.it("should serialize logs to valid JSON")
def test_json_serialize(logger):
    metrics = EpochMetrics(accuracy=0.90, f1=0.89, precision=0.88, recall=0.87, loss=0.15)
    logger.save_epoch(metrics, Dataset.TRAIN)

    json_output = logger.json_serialize()

    assert isinstance(json_output, str)
    assert "model-id" in json_output
    assert "accuracy" in json_output
    assert "loss" in json_output
