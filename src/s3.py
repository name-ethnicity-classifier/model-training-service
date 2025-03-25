import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os


class S3Handler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        load_dotenv()
        self._client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MINIO_SECRET_ACCESS_KEY"),
            endpoint_url=f"{os.environ.get('MINIO_HOST')}:{os.environ.get('MINIO_PORT')}"
        )

    @classmethod
    def instance(cls):
        return cls()

    @classmethod
    def upload(cls, bucket_name: str, body: str, object_key: str):
        cls.instance()._client.put_object(
            Body=body,
            Bucket=bucket_name,
            Key=object_key,
        )

    @classmethod
    def get(cls, bucket_name: str, object_key: str):    
        try:
            response = cls.instance()._client.get_object(Bucket=bucket_name, Key=object_key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    @classmethod
    def check_file_existence(cls, bucket_name: str, object_key: str) -> bool:
        try:
            cls.instance()._client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError:
            return False

