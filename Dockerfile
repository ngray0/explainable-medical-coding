# Start from the official NVIDIA CUDA image with CUDA 12.0
FROM nvcr.io/nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Set default shell to bash
SHELL ["/bin/bash", "-c"] 

# Update system
RUN apt update
RUN apt upgrade -y

# Install favourite tools
RUN apt install aptitude wget tree -y

# Install key libraries for OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0

# Install additional dependencies for the project
RUN apt-get install -y build-essential curl git make

# Install conda (using Python 3.11 compatible version)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda

# Set the default shell command to be conda environment 'base' + bash
SHELL ["/opt/conda/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]

# Create conda environment called main with Python 3.11.5
RUN conda create -n main python=3.11.5 -y

# Set the default shell command to be conda environment 'main' + bash
SHELL ["/opt/conda/bin/conda", "run", "-n", "main", "/bin/bash", "-c"]

# Ensure conda is configured for bash and fish
RUN conda init bash
RUN conda init fish

# Install PyTorch with CUDA 12.1 support
RUN echo y | pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Poetry for dependency management
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to not create virtual environment (since we're using conda)
RUN poetry config virtualenvs.create false

# Create app directory
RUN mkdir -p /app

# Copy project files
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# Install the project itself
RUN poetry install

# Create necessary directories as mentioned in the README
RUN mkdir -p data/raw data/processed models results \
    explainable_medical_coding/configs/experiment \
    explainable_medical_coding/configs/sweeps \
    .cache/huggingface/transformers \
    wandb

# Set environment variables for ML libraries
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV WANDB_DIR=/app/wandb

# Start from bash when the container launches
ENTRYPOINT ["/bin/bash"]
