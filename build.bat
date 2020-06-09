
cd ..\cubequery\
call build.bat

cd ..\cubequery-deployment

docker build --no-cache=true -f .\server.dockerfile -t cubequery-deploy-server .
docker build --no-cache=true -f .\worker.dockerfile -t cubequery-deploy-worker .

echo •