FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update && apt-get install -y wget bzip2 && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p $CONDA_DIR && \
    rm ~/miniforge.sh && \
    conda install python=3.10 -y && \
    conda clean -afy

WORKDIR /workspace

COPY requirements.txt /workspace/

RUN pip install --no-cache-dir -r requirements.txt && \
    conda install -c conda-forge python-pdal -y && \
    conda clean -afy

RUN python -c "import pdal; print('✅ PDAL installed successfully')" && \
    python -c "import laspy; print('✅ Laspy installed successfully')" && \
    python -c "import open3d; print('✅ Open3D installed successfully')" && \
    echo "✅ All critical packages verified"

COPY . /workspace/

CMD ["/bin/bash"]