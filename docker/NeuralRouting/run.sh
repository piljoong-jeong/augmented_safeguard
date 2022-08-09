#!/bin/sh

#CMD_BUILD="\
#	-t piljoong/neuralrouting:ransac \
#	."
#docker build $CMD_BUILD

CMD_RUN="-it \
	--gpus all \
	--name neuralrouting_ransac \
	--mount type=bind,src=/mnt/d/RIO10,dst=/opt/dataset \
        piljoong/neuralrouting:ransac"

docker run $CMD_RUN
