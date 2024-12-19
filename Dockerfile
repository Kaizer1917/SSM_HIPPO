# Use an official PyTorch image as the base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TVM
RUN git clone --recursive https://github.com/apache/tvm tvm
RUN cd tvm && mkdir build && cp cmake/config.cmake build
RUN cd tvm/build && cmake .. && make -j4

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/*

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models/checkpoints /app/logs

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV TVM_HOME=/app/tvm
ENV PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Default command
CMD ["./scripts/run_market_maker.sh"]
