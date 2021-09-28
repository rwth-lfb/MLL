ARG BASE_IMAGE=registry.git.rwth-aachen.de/jupyter/profiles/rwth-courses:latest
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1 ffmpeg 
USER ${NB_USER}

COPY requirements.txt /srv/requirements.txt
RUN pip install -r /srv/requirements.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
