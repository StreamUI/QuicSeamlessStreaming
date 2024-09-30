# ssh_instance.sh
source ./config.sh

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
