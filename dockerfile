FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir \
    flask>=3.0.0 \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    opencv-python-headless>=4.8.0 \
    && pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    && pip install --no-cache-dir ultralytics==8.1.0

COPY . .

ENV PORT=8080
CMD python app.py