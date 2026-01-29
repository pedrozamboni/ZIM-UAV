# FROM ubuntu:22.04

# ENV DEBIAN_FRONTEND=noninteractive
# ENV CONDA_DIR=/opt/conda
# ENV PATH=$CONDA_DIR/bin:$PATH

# RUN apt-get update && apt-get install -y wget bzip2 && \
#     rm -rf /var/lib/apt/lists/*

# RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
#     bash ~/miniforge.sh -b -p $CONDA_DIR && \
#     rm ~/miniforge.sh && \
#     conda install python=3.10 -y && \
#     conda clean -afy

# WORKDIR /workspace

# COPY requirements.txt /workspace/

# RUN pip install --no-cache-dir -r requirements.txt && \
#     conda install -c conda-forge python-pdal open3d -y && \
#     conda clean -afy

# RUN python -c "import pdal; print('✅ PDAL installed successfully')" && \
#     python -c "import laspy; print('✅ Laspy installed successfully')" && \
#     python -c "import open3d; print('✅ Open3D installed successfully')" && \
#     echo "✅ All critical packages verified"

# COPY . /workspace/

# CMD ["/bin/bash"]
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PIP_ROOT_USER_ACTION=ignore
ENV PROJ_LIB=/opt/conda/share/proj  

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p $CONDA_DIR && \
    rm ~/miniforge.sh && \
    conda install python=3.10 -y && \
    conda clean -afy

WORKDIR /workspace

COPY requirements.txt /workspace/

# Install packages INCLUDING proj and proj-data
RUN pip install --no-cache-dir -r requirements.txt && \
    conda install -c conda-forge python-pdal open3d proj proj-data -y && \
    conda clean -afy

# Verify installations
RUN python -c "import pdal; print('✅ PDAL installed successfully')" && \
    python -c "import numpy; print('✅ NumPy version:', numpy.__version__)" && \
    python -c "import scipy; print('✅ SciPy version:', scipy.__version__)" && \
    python -c "import pandas; print('✅ Pandas installed successfully')" && \
    python -c "import laspy; print('✅ Laspy installed successfully')" && \
    python -c "import open3d; print('✅ Open3D installed successfully')" && \
    python -c "import jakteristics; print('✅ Jakteristics installed successfully')" && \
    python -c "import sklearn; print('✅ Scikit-learn installed successfully')" && \
    python -c "import skimage; print('✅ Scikit-image installed successfully')" && \
    python -c "import alphashape; print('✅ Alphashape installed successfully')" && \
    python -c "import shapely; print('✅ Shapely installed successfully')" && \
    echo "✅ All critical packages verified"

COPY . /workspace/

CMD ["/bin/bash"]