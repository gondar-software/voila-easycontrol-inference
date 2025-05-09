# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as base

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Copy project to image
COPY ./ /root

# Install dependencies
RUN pip install runpod requests pillow && \
    pip install -r /root/requirements.txt

# Start container
CMD ["/root/start.sh"]