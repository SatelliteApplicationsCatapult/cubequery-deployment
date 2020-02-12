# Cube Query Deployment

This is both an example project to show how to deploy the cube query system and a real deployment.

This shows pulling in two library modules (sac_utils and data_cube_utilities) and adding some processors.

## Building and Running

1) Clone this repo.
1) Clone the [cubequery repo](https://github.com/SatelliteApplicationsCatapult/cubequery).
1) Edit the docker-compose.yml file to include the paths to your repos and set environment vars as you require.
1) Edit the build.bat file to include the paths to your repos.
1) Create the containers by running `build.bat`. If you are working on another platform the commands can be borrowed from that script.
1) Run `docker-compose` up to start the environment.

The server should be able to live reload when changes are made, however the worker is not able to do so. That will need manually restarting.
