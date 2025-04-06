
docker_compose_path="./dev-infrastructure/docker-compose.yml"


if [ "$1" = "--init" ]; then
    echo "This will reinitialize the entire development database and buckets and remove their data."
    read -p "Are you sure [y/n]? " response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        docker-compose -f $docker_compose_path down -v
    else
        echo "Exiting."
        exit 0
    fi
fi

docker-compose -f $docker_compose_path up
