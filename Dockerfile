
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-compatible PyTorch
RUN python -m pip install --upgrade pip && \
    python -m pip install torch \
      --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install -r requirements.txt

COPY . .

# Download the MONAI pretrained model during build
RUN python -c "from monai.bundle import download; download(name='brats_mri_segmentation', version='0.5.4', bundle_dir='models')"

EXPOSE 8000

CMD ["sh", "-c", "python run.py --host 0.0.0.0 --port ${PORT:-8000}"]
