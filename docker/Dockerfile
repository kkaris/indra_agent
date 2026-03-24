FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy local source and install
COPY . /tmp/indra_agent
RUN pip install --no-cache-dir /tmp/indra_agent && \
    rm -rf /tmp/indra_agent

# Copy pre-built resources (built by docker/prebuild.sh to avoid OOM).
# The INDRA ontology build requires ~8 GB RAM — more than Docker Desktop
# typically allows on 16 GB machines.
COPY docker/resources/protmapper/ /root/.data/protmapper/
COPY docker/resources/indra_ontology/ /root/.indra/bio_ontology/
COPY docker/resources/adeft/ /root/.local/share/adeft/
COPY docker/resources/gilda/ /root/.data/gilda/

RUN mkdir /app
WORKDIR /app

COPY docker/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Run the MCP server
CMD ["/app/startup.sh"]
