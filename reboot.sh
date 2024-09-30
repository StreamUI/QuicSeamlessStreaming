# reboot.sh
source ./config.sh

gcloud compute instances reset $INSTANCE_NAME --zone=$ZONE
