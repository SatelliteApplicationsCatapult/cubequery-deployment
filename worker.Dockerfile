from cubequery_worker:latest

USER root
RUN /opt/conda/bin/pip uninstall datacube_utilities
RUN /opt/conda/bin/pip install git+https://github.com/emilyselwood/datacube-utilities.git@test-branch#egg=datacube_utilities
RUN mkdir /data && chmod -R 777 /data
USER celery:celery

COPY processes/ /processes/processes
COPY datacube.conf /etc/datacube.conf
ENV APP_EXTRA_PATH=/processes/
