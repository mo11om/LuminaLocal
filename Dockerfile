# Multi-stage build for production-ready CV Matcher application
# Stage 1: Build
FROM continuumio/miniconda3:latest as builder

WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Create the conda environment and install dependencies
RUN conda create -n job python=3.10 -y && \
    /opt/conda/envs/job/bin/pip install --upgrade pip setuptools wheel && \
    /opt/conda/envs/job/bin/pip install -r requirements.txt

# Stage 2: Runtime
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs /opt/conda/envs

# Create symlink for miniconda3 path compatibility (matching expected ~/code/miniconda3)
RUN ln -s /opt/conda /opt/code_miniconda3 && \
    mkdir -p /home/code && \
    ln -s /opt/conda /home/code/miniconda3

# Activate conda environment by default
ENV PATH="/opt/conda/envs/job/bin:$PATH" \
    CONDA_DEFAULT_ENV=job

# Copy application files
COPY config.json .
COPY pyproject.toml .
COPY src/ src/
COPY data/ data/

# Install the local package in development mode
RUN /opt/conda/envs/job/bin/pip install -e .

# Create output directory
RUN mkdir -p data/output

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /opt/conda/envs/job/bin/python -c "import cv_matcher; print('healthy')" || exit 1

# Default command: Run the batch job
CMD ["bash", "-c", "source /opt/conda/bin/activate job && python -m cv_matcher.main"]
