# copy_files.sh
source ./config.sh

gcloud compute scp --recurse $INSTANCE_NAME:/home/jordan/app/outputs ./app/outputs \
    --zone=$ZONE