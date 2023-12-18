FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        libglib2.0-0

WORKDIR /app
COPY requirements.txt /app
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip3 install Flask Flask-Uploads Flask-Cors pandas numpy scikit-learn

COPY . /app
EXPOSE 8080
CMD ["flask","run","--host","0.0.0.0","--port","8080"]