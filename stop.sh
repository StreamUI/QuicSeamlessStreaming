# stop_instance.sh
source ./config.sh

gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
