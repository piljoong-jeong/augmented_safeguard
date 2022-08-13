#!/bin/sh

CMD_BUILD="\
	-t piljoong/neuralrouting:ransac_icp \
	."
docker build $CMD_BUILD

CMD_RUN="-it \
	--gpus all \
	--mount type=bind,src=/mnt/d/RIO10,dst=/opt/dataset \
	--mount type=bind,src=/root/Documents/Projects/NeuralRouting,dst=/opt/relocalizer_codes/NeuralRouting \
        piljoong/neuralrouting:ransac_icp"

docker run $CMD_RUN
