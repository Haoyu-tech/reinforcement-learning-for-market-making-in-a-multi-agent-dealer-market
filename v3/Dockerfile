FROM python:3.7.17-slim-bullseye
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
COPY requirements-historical.txt /tmp/requirements-historical.txt
RUN python -m pip install --no-cache-dir pip==23.0.1 setuptools==57.5.0 wheel==0.37.1 && python -m pip install --no-cache-dir -r /tmp/requirements-historical.txt
RUN python -m pip install --no-cache-dir requests==2.31.0 && python -m pip install --no-cache-dir --no-deps --force-reinstall opencv-python-headless==4.5.5.64
RUN python -m pip uninstall -y opencv-python opencv-python-headless && python -m pip install --no-cache-dir --no-deps opencv-python==4.5.5.64 && python -m pip install --no-cache-dir --no-deps opencv-python-headless==4.5.5.64
WORKDIR /work
