#!/bin/bash

cd ../cubequery/
./build.sh

cd ../cubequery-deployment
docker build -f ./server.Dockerfile -t cubequery-deploy-server .
docker build -f ./worker.Dockerfile -t cubequery-deploy-worker .
