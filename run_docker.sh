NAME_PROJECT=$(basename ${PWD})

COMMANDS_BUILD="\
	-t $NAME_PROJECT -f docker/Dockerfile ."
docker run $COMMANDS_RUN

COMMANDS_RUN="\
	--name $NAME_PROJECT \
	$NAME_PROJECT"