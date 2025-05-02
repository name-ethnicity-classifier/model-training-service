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
from logger import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


def get_untrained_models(db_connection: sqlalchemy.Connection):
    untrained_models_query = sqlalchemy.text("SELECT id, nationalities, is_grouped FROM model WHERE is_trained = FALSE;")
    rows = db_connection.execute(untrained_models_query).all()
    
    logger.info(f"{len(rows)} untrained model(s) fetched.")

    return rows


def update_trained_model(db_connection: sqlalchemy.Connection, model_id: str, accuracy: float, scores: list[float]):
    update_trained_model_query = sqlalchemy.text("UPDATE model SET accuracy = :accuracy, scores = :scores, is_trained = true WHERE id = :id")
    db_connection.execute(update_trained_model_query, {"accuracy": accuracy, "scores": scores, "id": model_id})
    db_connection.commit()

    logger.info(f"Updated DB entry for trained model with id {model_id}.")


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

    logger.info(f"Model and train results saved to S3.")


def run_model_pipeline(untrained_model: UntrainedModel) -> tuple[float, list[float]]:
    processed_dataset = create_dataset(untrained_model)
    save_dataset(processed_dataset, untrained_model.id)

    logger.info(f"Dataset created with classes {untrained_model.classes}.")

    train_setup = TrainSetup(
        model_id=untrained_model.id,
        base_model_name=config.base_model,
        classes=untrained_model.classes,
        dataset=processed_dataset
    )
    train_setup.train()
    train_setup.test()
    result_metrics, model_state_dict = train_setup.get_results()

    save_model(train_setup.get_logs(), model_state_dict, untrained_model.id)

    return result_metrics.accuracy, result_metrics.scores.f1


scheduler = BlockingScheduler()
@scheduler.scheduled_job(CronTrigger.from_crontab(config.cron_rule))
@error_handler
def main():
    logger.info("Model-training service envoked.")
    db_connection = create_db_connection()
    
    untrained_model_rows = get_untrained_models(db_connection)

    if not untrained_model_rows or len(untrained_model_rows) == 0:
        logger.info(f"No models to train. Exiting.")
        db_connection.close()
        return

    for idx, row in enumerate(untrained_model_rows):
        logger.info("")
        logger.info(f"Running pipeline for model with id {row[0]} {idx + 1}/{len(untrained_model_rows)}.")

        untrained_model = UntrainedModel(id=row[0], classes=row[1], is_grouped=row[2])
        accuracy, f1_scores = run_model_pipeline(untrained_model)
        update_trained_model(db_connection, untrained_model.id, accuracy, f1_scores)

    logger.info("Model-training service exiting.")
    
    db_connection.close()



if __name__ == "__main__":
    scheduler.start()
