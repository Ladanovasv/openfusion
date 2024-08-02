xhost local:root

docker run --rm --gpus all --device=/dev/dri:/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
    -v "$PWD":/workspace \
    -v /mnt/hdd5/Datasets:/datasets:ro \
    --network=host \
    openfusion:latest \
    python visualization_ply.py $@