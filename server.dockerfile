from cubequery_server:latest

COPY processes/ /processes/processes
COPY sac_utils/ /processes/sac_utils
COPY data_cube_utilities/ /processes/utils_dcal

ENV APP_EXTRA_PATH=/processes/
