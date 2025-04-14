import os
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum


load_dotenv()


class Environment(Enum):
    PROD = "prod"
    DEV = "dev"


@dataclass
class Config:
    model_bucket: str = os.getenv("MODEL_S3_BUCKET")
    base_data_bucket: str = os.getenv("BASE_DATA_S3_BUCKET")
    base_model: str = os.getenv("BASE_MODEL")
    minio_user: str = os.getenv("MINIO_USER")
    minio_password: str = os.getenv("MINIO_PASSWORD")
    minio_host: str = os.getenv("MINIO_HOST")
    minio_port: str = os.getenv("MINIO_PORT")
    environment: str = Environment(os.getenv("ENVIRONMENT"))


config = Config()