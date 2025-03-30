import io
import json
import os
import pickle
import sqlalchemy
import torch
from db import create_db_connection
from schemas import UntrainedModel
from errors import error_handler
from preprocessing import create_dataset
from s3 import S3Handler
from config import config
from training.train import TrainSetup
from schemas import ProcessedName


def get_untrained_models(db_connection: sqlalchemy.Connection):
    untrained_models_query = sqlalchemy.text("SELECT id, nationalities, is_grouped FROM model WHERE is_trained = FALSE;")
    rows = db_connection.execute(untrained_models_query)

    return rows


def update_trained_model(db_connection: sqlalchemy.Connection, model_id: str, accuracy: float, scores: list[float]):
    update_trained_model_query = sqlalchemy.text(f"INSERT INTO model (accuracy, scores, is_trained) VALUES = ('{accuracy}', '{scores}', true) WHERE id = {model_id};")
    db_connection.execute(update_trained_model_query)


def save_dataset(dataset: list[ProcessedName], model_id: str):
    S3Handler.upload(
        bucket_name=os.getenv("MODEL_S3_BUCKET"),
        body=pickle.dumps(dataset),
        object_key=f"{model_id}/dataset.pickle"
    )


def save_model(train_results: dict, model_state_dict, model_id: str):
    S3Handler.upload(
        bucket_name=os.getenv("MODEL_S3_BUCKET"),
        body=json.dumps(train_results),
        object_key=f"{model_id}/logs.json"
    )

    model_buffer = io.BytesIO()
    torch.save(model_state_dict, model_buffer)
    model_buffer.seek(0)

    S3Handler.upload(
        bucket_name=os.getenv("MODEL_S3_BUCKET"),
        body=model_buffer.getvalue(),
        object_key=f"{model_id}/model.pt"
    )


def get_dataset(untrained_model: UntrainedModel) -> list[ProcessedName]:
    processed_dataset = S3Handler.get(config.model_bucket, f"{untrained_model.id}/dataset.pickle")

    if processed_dataset:
        return pickle.loads(processed_dataset)

    return create_dataset(untrained_model)


def run_model_pipeline(untrained_model: UntrainedModel) -> tuple[float, list[float]]:
    processed_dataset = get_dataset(untrained_model)
    save_dataset(processed_dataset, untrained_model.id)

    train_setup = TrainSetup(
        model_id=untrained_model.id,
        base_model_name=config.base_model,
        classes=untrained_model.classes,
        dataset=processed_dataset
    )
    train_setup.train()
    train_setup.test()
    result_metrics, model_state_dict = train_setup.get_results()

    save_model(train_setup.get_logs(), model_state_dict)

    return result_metrics.accuracy, result_metrics.scores.f1


@error_handler
def main():
    db_connection = create_db_connection()
    
    untrained_model_rows = get_untrained_models(db_connection)

    for row in untrained_model_rows:
        untrained_model = UntrainedModel(id=row[0], classes=row[1], is_grouped=row[2])
        accuracy, f1_scores = run_model_pipeline(untrained_model)
        update_trained_model(db_connection, untrained_model.id, accuracy, f1_scores)


if __name__ == "__main__":
    main()