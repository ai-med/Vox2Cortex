FROM continuumio/miniconda3

RUN \
  conda update -n base -c defaults conda -y --quiet && \
  conda install conda-pack -y && \
  conda clean --all

COPY t1_10_pyt3d0_6_1_2.yml /

SHELL ["/bin/bash", "-c"]
RUN \
  conda env create --file /t1_10_pyt3d0_6_1_2.yml && \
  conda clean --all && \
  rm -f /t1_10_pyt3d0_6_1_2.yml

ENV DEBIAN_FRONTEND=noninteractive
RUN \
  apt-get update && \
  apt-get install -y --no-install-recommends libgl1-mesa-glx libosmesa6 freeglut3 && \
  apt-get clean

COPY pytorch3d-0.6.1-cp39-cp39-linux_x86_64.whl /
RUN \
  source /opt/conda/bin/activate torch_1_10 && \
  pip install --no-input --no-cache-dir --pre --no-index --find-links / pytorch3d && \
  rm -f /pytorch3d-0.6.1-cp39-cp39-linux_x86_64.whl

RUN mkdir /mnt/data /mnt/code

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.3
