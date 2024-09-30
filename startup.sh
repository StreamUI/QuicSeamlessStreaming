#!/bin/bash
# Install necessary dependencies
apt-get update
apt-get install -y python3 python3-pip

# Install any additional Python packages required for the server
# pip3 install -r /path/to/requirements.txt

# Start the QUIC Python server
# nohup python3 /home/jordan/app/server.py > /var/log/quic_server.log 2>&1 &
