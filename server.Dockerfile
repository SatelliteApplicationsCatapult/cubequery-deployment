from cubequery_server:latest

RUN /opt/conda/bin/pip install git+https://github.com/SatelliteApplicationsCatapult/datacube-utilities.git#egg=datacube_utilities

COPY processes/ /processes/processes
COPY datacube.conf /etc/datacube.conf

ENV APP_EXTRA_PATH=/processes/
