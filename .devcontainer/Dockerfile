FROM mambaorg/micromamba:1.5-bookworm-slim

USER root
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/conda/envs /opt/conda/pkgs && \
    chown -R mambauser:mambauser /opt/conda/envs /opt/conda/pkgs

USER mambauser

RUN micromamba shell init -s bash -p /opt/conda