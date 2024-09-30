# start_instance.sh
source ./config.sh

gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
