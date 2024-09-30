# create_instance.sh
source ./config.sh

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$GPU_MACHINE_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --accelerator type=$GPU_TYPE,count=$GPU_COUNT \
    --maintenance-policy TERMINATE \
    --metadata-from-file startup-script=./startup.sh \
    --restart-on-failure \
    $PREEMPTIBLE
