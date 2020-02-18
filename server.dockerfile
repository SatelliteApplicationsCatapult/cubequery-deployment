from cubequery_server:latest

COPY processes/ /processes/processes
COPY sac_utils/ /processes/sac_utils
COPY data_cube_utilities/ /processes/utils_dcal
COPY datacube.conf /etc/datacube.conf

ENV APP_EXTRA_PATH=/processes/
