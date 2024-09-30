# describe_instance.sh
source ./config.sh

gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
