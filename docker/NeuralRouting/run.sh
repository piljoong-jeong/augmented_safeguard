#!/bin/sh

#CMD_BUILD="\
#	-t piljoong/neuralrouting:ransac \
#	."
#docker build $CMD_BUILD

CMD_RUN="-it \
	--gpus all \
	--mount type=bind,src=/mnt/d,dst=/opt/dataset \
        piljoong/neuralrouting:ransac"

docker run $CMD_RUN
