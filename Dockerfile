# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN apt-get install -y dpkg
RUN apt-get install -y gpg
RUN apt-get install -y sed
RUN apt-get install -y coreutils

# Add repos for the Cuda-Toolkit and Drivers
RUN wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN dpkg -i cuda-repo-debian12-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Add repos for cuDNN Library as required by faster-whisper
RUN wget https://developer.download.nvidia.com/compute/cudnn/9.7.0/local_installers/cudnn-local-repo-debian12-9.7.0_1.0-1_amd64.deb
RUN dpkg -i cudnn-local-repo-debian12-9.7.0_1.0-1_amd64.deb
RUN cp /var/cudnn-local-repo-debian12-9.7.0/cudnn-local-EAC291EE-keyring.gpg /usr/share/keyrings/
#RUN cp /var/cuda-repo-debian12-9-7-local/cudnn-*-keyring.gpg /usr/share/keyrings/


# Add repos for the Nvidia-Container-Toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update
RUN apt-get install -y nvidia-open
RUN apt-get install -y nvidia-container-toolkit
RUN apt-get -y install cuda-toolkit-12-8
RUN apt-get -y install cudnn-cuda-12

# Install some networking tools
RUN apt-get install -y ifupdown
RUN apt-get install -y nano

RUN nvidia-ctk runtime configure --runtime=docker

ENV NVIDIA_VISIBLE_DEVICES=all
ENV PATH=$PATH:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib/

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
EXPOSE 11199

# Create the log folder
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install nvidia-cudnn

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "speech_transcriber.py"]