from cubequery_server:latest

USER root
/opt/conda/bin/pip install install git+https://github.com/SatelliteApplicationsCatapult/datacube-utilities.git#egg=datacube_utilities
USER celery:celery

COPY processes/ /processes/processes
COPY datacube.conf /etc/datacube.conf

ENV APP_EXTRA_PATH=/processes/
