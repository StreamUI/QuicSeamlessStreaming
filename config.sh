# config.sh

# Set variables
PROJECT_ID="GCP_PROJECT_ID"
INSTANCE_NAME="GCP_INSTANCE_NAME"
# ZONE="us-central1-b"
ZONE="us-west4-b"
BOOT_DISK_SIZE="200GB"
GPU_MACHINE_TYPE="n1-standard-8"  # Cheapest GPU-supported machine (4 vCPU, 15 GB memory)
GPU_TYPE="nvidia-tesla-t4"        # T4 is a cost-effective GPU
GPU_COUNT="1"

IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# PREEMPTIBLE="--preemptible"
