import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import json
import pickle
from config import config


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
            aws_access_key_id=config.minio_user,
            aws_secret_access_key=config.minio_password,
            endpoint_url=f"{config.minio_host}:{config.minio_port}"
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
            body = response["Body"].read()
            
            if object_key.endswith(".json"):
                return json.loads(body)
            elif object_key.endswith(".pickle"):
                return pickle.loads(body)
            return body
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise

    @classmethod
    def check_file_existence(cls, bucket_name: str, object_key: str) -> bool:
        try:
            cls.instance()._client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except Exception:
            return False
        
    @classmethod
    def clear_bucket(cls, bucket_name: str):
        client = cls.instance()._client

        objects_to_delete = client.list_objects_v2(Bucket=bucket_name)

        if "Contents" in objects_to_delete:
            delete_objects = [{"Key": obj["Key"]} for obj in objects_to_delete["Contents"]]
            client.delete_objects(Bucket=bucket_name, Delete={"Objects": delete_objects})


