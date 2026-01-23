# Start from a specific tested version
FROM hepstore/rivet-pythia:4.1.0

# Set environment to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    cmake \
    build-essential \
    vim \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir bilby surmise scikit-learn

# Copy Bayes_HEP package to /usr/local/share/Bayes_HEP
COPY Bayes_HEP /usr/local/share/Bayes_HEP

# Make sure all __init__.py files exist for package recognition
RUN touch /usr/local/share/Bayes_HEP/__init__.py && \
    touch /usr/local/share/Bayes_HEP/Design_Points/__init__.py && \
    touch /usr/local/share/Bayes_HEP/Emulation/__init__.py && \
    touch /usr/local/share/Bayes_HEP/Calibration/__init__.py

RUN chmod -R a+rwX /usr/local/share/Bayes_HEP

# Set PYTHONPATH so Bayes_HEP is importable everywhere
ENV PYTHONPATH="/usr/local/lib/python3.10/site-packages:/usr/local/share"

#unset PYTHIA8DATA
ENV PYTHIA8DATA=

# Optionally copy your full project to /workdir for scripts, configs, etc.
# COPY . /workdir

# Set the working directory
WORKDIR /workdir

# Default command
CMD ["/bin/bash"]