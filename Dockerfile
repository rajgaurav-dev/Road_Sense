# -------------------------------------------------
# Base Image (CPU Only)
# -------------------------------------------------
FROM python:3.10-slim

# -------------------------------------------------
# System Dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Set Working Directory
# -------------------------------------------------
WORKDIR /workspace

# -------------------------------------------------
# Copy Your Project (Without mmdetection)
# -------------------------------------------------
COPY . .

# -------------------------------------------------
# Upgrade pip
# -------------------------------------------------
RUN pip install --upgrade pip

# -------------------------------------------------
# Install PyTorch CPU
# -------------------------------------------------
RUN pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# -------------------------------------------------
# Install Base Requirements
# -------------------------------------------------
RUN pip install -r requirements.txt

# -------------------------------------------------
# Install OpenMMLab Core via MIM
# -------------------------------------------------
RUN mim install mmcv==2.1.0

# -------------------------------------------------
# Clone MMDetection
# -------------------------------------------------
RUN git clone https://github.com/open-mmlab/mmdetection.git

# -------------------------------------------------
# Checkout Stable Version (Recommended)
# -------------------------------------------------
WORKDIR /workspace/mmdetection
RUN git checkout v3.3.0

# -------------------------------------------------
# Install MMDetection Editable
# -------------------------------------------------
RUN pip install -e .

# -------------------------------------------------
# Set Default Workdir
# -------------------------------------------------
WORKDIR /workspace/Road_Sense

# -------------------------------------------------
# Default Command
# -------------------------------------------------
CMD ["python", "train.py"]
