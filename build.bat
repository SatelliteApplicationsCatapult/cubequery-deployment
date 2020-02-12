
cd ..\cubequery\
call build.bat

cd ..\cubequery-deployment

docker build -f .\server.dockerfile -t cubequery-deploy-server .
docker build -f .\worker.dockerfile -t cubequery-deploy-worker .

echo •