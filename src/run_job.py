import pickle
import sqlalchemy
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


def get_dataset(untrained_model: UntrainedModel) -> list[ProcessedName]:
    processed_dataset = S3Handler.get(config.model_bucket, f"{untrained_model.id}/dataset.pickle")

    if processed_dataset:
        return pickle.loads(processed_dataset)

    return create_dataset(untrained_model)


def run_model_pipeline(untrained_model: UntrainedModel):
    processed_dataset = get_dataset(untrained_model)

    train_setup = TrainSetup(
        model_id=untrained_model.id,
        base_model_name=config.base_model,
        classes=untrained_model.classes,
        dataset=processed_dataset
    )
    train_setup.train()
    train_setup.test()
    train_setup.save()

    print(untrained_model.id, untrained_model.classes, untrained_model.is_grouped)

    return 0


@error_handler
def main():
    db_connection = create_db_connection()
    
    rows = get_untrained_models(db_connection)

    for row in rows:
        run_model_pipeline(UntrainedModel(
            row[0],
            row[1],
            row[2]
        ))

    return 0


if __name__ == "__main__":
    main()