# update.sh
source ./config.sh

gcloud compute instances add-metadata $INSTANCE_NAME \
    --zone=us-central1-c \
    --metadata-from-file startup-script=./startup.sh
