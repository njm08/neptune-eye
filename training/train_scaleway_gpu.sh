#!/bin/bash
CONTAINER=rg.fr-par.scw.cloud/njm08/neptune-eye:latest-training-amd64

# Connect to Scaleway GPU instance
python3 scaleway_gpu_cli.py start -v

# Execute all setup commands in a single SSH session
python3 scaleway_gpu_cli.py -v exec \
    "cd training" \
    "rm -rf neptune-eye" \
    "git clone https://github.com/njm08/neptune-eye.git" \
    "cd neptune-eye" \
    "docker pull $CONTAINER"


# For interactive Docker session, use ssh command
echo "Opening SSH session for interactive Docker training..."
echo "Run this command on the instance:"
echo "  docker run -it --ipc=host --runtime=nvidia --rm --gpus all -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -v .:/workspace -w /workspace $CONTAINER"
python3 scaleway_gpu_cli.py ssh

# Store the results SOMEWHERE. :)
# Turn off the instance.
# Uncomment when ready: python3 scaleway_gpu_cli.py stop-and-wait -v