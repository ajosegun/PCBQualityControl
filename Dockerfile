FROM python:3.8-slim-buster
# Install Git
RUN apt-get update && apt-get install -y git
# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
#ENTRYPOINT ["streamlit","run"]
#CMD ["app.py", "--server.port", "8080"]

# The command to launch streamlit app and expose it's port 8080, when container is up and running
CMD streamlit run --server.port 8080 app.py