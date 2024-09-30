# copy_files.sh
source ./config.sh

gcloud compute scp --recurse ./app \
    $INSTANCE_NAME:/home/jordan/ \
    --zone=$ZONE
