NAME_PROJECT=$(basename ${PWD})

COMMANDS_BUILD="\
	-t $NAME_PROJECT -f docker/Dockerfile ."
docker build $COMMANDS_BUILD

COMMANDS_RUN="\
	--name $NAME_PROJECT \
	--rm \
	-it \
	$NAME_PROJECT"
docker run $COMMANDS_RUN