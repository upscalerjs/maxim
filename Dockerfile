FROM tensorflow/tensorflow:latest-devel-gpu
RUN apt update && \
    apt install -y \
    curl \
    less \
    vim \
    git

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install \
        jax \
        flax \
        tensorflow_hub \
        tensorflowjs \
        tqdm \
        einops \
        ml_collections \
        scikit-image \
        scikit-learn
WORKDIR /code
COPY . /code
CMD /bin/bash
