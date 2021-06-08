# Cube Query Deployment


THIS PROJECT IS NO LONGER REQUIRED OR MAINTAINED. DO NOT USE OR EXPECT IT TO WORK. 

It is only left here as a historical artifact. Go and see the [cubequery](https://github.com/SatelliteApplicationsCatapult/cubequery) project
to find out how things now work.

*********************************************************************


This is both an example project to show how to deploy the [cubequery](https://github.com/SatelliteApplicationsCatapult/cubequery) system and a real deployment.

This shows pulling in two library modules (sac_utils and data_cube_utilities) and adding some processors. 

## Building and Running

1) Clone this repo.
1) Clone the [cubequery repo](https://github.com/SatelliteApplicationsCatapult/cubequery).
1) Edit the docker-compose.yml file to include the paths to your repos and set environment vars as you require.
1) Edit the build.bat file to include the paths to your repos.
1) Create the containers by running `build.bat`. If you are working on another platform the commands can be borrowed from that script.
1) Run `docker-compose` up to start the environment.

The server should be able to live reload when changes are made, however the worker is not able to do so. That will need manually restarting.

For a production environment check out our [helm charts](https://github.com/SatelliteApplicationsCatapult/helm-charts)

## Converting a notebook to a cubequery process

While a note book is great for developing a process. It is not so great when running the process in production. See `processes/ndvi_anomaly.py` for an example

1) Extract everything that can be a parameter to somewhere near the top of the notebook
1) Remove any and all debug and investigation plotting code. It should not be nessary to import matplotlib at any point in the note book
1) Create a new python file
1) Create a class that extends CubeQueryTask
1) The class needs a member called `display_name` with what this process should be called when shown to a user.
1) The class needs a member called `description` with a description for a user of what this process does.
1) Create a parameter list which contains details of each parameter.
    1) Name
    1) Display Name
    1) Data type
    1) Description
1) Call the parameter setup function. `CubeQueryTask.cal_significant_kwargs(parameters)` This is required to setup the parameters in the processing engine
1) Create a method called `generate_product` that takes self, a datacube instance, and the path prefix to be used for any outputs. Along with each of the defined parameters from the earlier list as arguments.
1) In this method implement the process
1) The method needs to return a list of output files that should be kept when the processing is done. These should include the path prefix passed in to the function. It is important to use the path prefix when generating output data so that each task run gets a unique output folder. Otherwise your process may end up overwriting some of the work of another processor if two are running at the same time.
1) Rebuild and re-deploy all the containers.
