from cubequery_worker:latest

USER root
RUN /opt/conda/bin/pip install git+https://github.com/SatelliteApplicationsCatapult/datacube-utilities.git#egg=datacube_utilities
USER celery:celery

COPY processes/ /processes/
COPY datacube.conf /etc/datacube.conf
ENV APP_EXTRA_PATH=/processes/
