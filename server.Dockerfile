FROM cubequery_server:latest

RUN /opt/conda/bin/pip uninstall datacube_utilities
RUN /opt/conda/bin/pip install git+https://github.com/emilyselwood/datacube-utilities.git@test-branch#egg=datacube_utilities

COPY processes/ /processes/processes
COPY datacube.conf /etc/datacube.conf

RUN mkdir /data && chmod -R 777 /data

ENV APP_EXTRA_PATH=/processes/
