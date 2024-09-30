source ./config.sh

gcloud compute instances set-machine-type $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=n1-standard-8
