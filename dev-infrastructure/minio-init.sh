#!/bin/sh

until (mc alias set minio_instance $MINIO_HOST $MINIO_USER $MINIO_PASSWORD); do
  echo "Waiting for MinIO to start..."
  sleep 3
done

echo "Connected to MinIO!"

BUCKETS="models"

for BUCKET in $BUCKETS; do
  if ! mc ls minio_instance/$BUCKET > /dev/null 2>&1; then
    echo "Creating bucket: $BUCKET"
    mc mb minio_instance/$BUCKET
  else
    echo "Bucket $BUCKET already exists."
  fi
done

echo "Bucket creation completed."
