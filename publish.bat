docker tag cubequery-deploy-worker:latest satapps/cubequery-worker:%1
docker tag cubequery-deploy-server:latest satapps/cubequery-server:%1

docker push satapps/cubequery-server:%1
docker push satapps/cubequery-worker:%1