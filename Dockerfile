FROM nvcr.io/nvidia/pytorch:21.11-py3

# update repo info
RUN apt update -y

# other python stuff
RUN pip install wandb && \
    pip install ruamel.yaml && \
    pip install --upgrade tqdm && \
    pip install timm && \
    pip install einops && \
    pip install mpi4py && \
    pip install h5py && \
    pip install xarray && \
    pip install netcdf4 && \
    pip install dask && \
    pip install cdsapi && \
    pip install torchinfo && \
    pip install pandas


# benchy
RUN pip install git+https://github.com/romerojosh/benchy.git

# copy source code
COPY config .
COPY networks .
COPY utils .
COPY inference .
COPY data_process .
COPY *.py .
COPY *.sh .

RUN pip list
