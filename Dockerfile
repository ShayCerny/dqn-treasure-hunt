FROM python:3.11-slim

# System deps for tensorflow-cpu / h5py wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces convention: run as a non-root UID 1000 user.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

ENV FLASK_DEBUG=false \
    FLASK_HOST=0.0.0.0 \
    PORT=7860

EXPOSE 7860

CMD ["python", "run.py"]
