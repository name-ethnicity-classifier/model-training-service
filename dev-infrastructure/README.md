# Development infrastructure
This folder provides the infrastructure and files necessary to run this service.

## Contents

### 1. docker-compose.yaml
This file is being used to start all services:
- seeds the database using the ``db-seed/init.sql`` script (must be the same as the ``backend`` service uses) and also adds some dummy models using the ``db-seed/dummy-models.sql`` script
- starts the AdminerUI to provide a UI for the database
- starts a Minio S3 compatible storage and adds the contents from ``./base-data`` using the ``minio-init.sh`` script (see below)

### 2. db-seed/
Contains the SQL scripts which the database container uses to intialize the tables and add dummy data.

### 3. base-data/
This folder contains all files which are necessary for the service and need to exist in the Minio ``base-data`` bucket before the service starts up. It contains:
- the model-configuration JSON file in a ``model-configs/`` subfolder used for training the model
- a ``nationalities.json`` file which contains all nationalities which are the possible classes the models can be trained on
- a ``raw_dataset.pickle`` file used to create model-specific datasets and train on them
  
All contents of this folder are not included in the Git repository and must be requested by the maintainer.

### 4. minio-init.sh
This script is being used by the ``minio-init`` container which creates the ``models`` and ``base-data`` buckets and copies the contents from ``./base-data/`` into the respective bucket.

### 5. infra.yml
This docker-compose is basically the same as docker-compose.yml but it also includes the model-training-service as well. Can be used to test the model-training-service in a container as it was in production.