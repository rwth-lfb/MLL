# FROM jupyterhub/jupyterhub:1.0.0
ARG BASE_IMAGE=registry.git.rwth-aachen.de/jupyter/profiles/rwth-courses:2020-ss.1
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1
USER ${NB_USER}

COPY requirements.txt /srv/requirements.txt
RUN pip install -r /srv/requirements.txt
