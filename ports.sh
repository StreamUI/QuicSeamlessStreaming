# configure_firewall.sh
source ./config.sh

# Open necessary ports for QUIC (UDP on port 4433)
gcloud compute firewall-rules create allow-quic \
    --allow udp:4433 \
    --target-tags=quic-server \
    --description="Allow incoming QUIC traffic on UDP 4433"

# Add network tags to the instance
gcloud compute instances add-tags $INSTANCE_NAME \
    --tags=quic-server \
    --zone=$ZONE
