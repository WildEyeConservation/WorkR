# WorkR
# Copyright (C) 2023

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /code
COPY requirements.txt .
COPY r-packages.txt .

# Install required system packages and add the R repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    dirmngr \
    ca-certificates \
    apt-transport-https \
    lsb-release \
    && echo "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -sc)-cran40/" >> /etc/apt/sources.list.d/r-cran.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9 \
    && apt-get update

# Install necessary packages
RUN apt-get install -y \
    python3.7 \
    python3-pip \
    libimage-exiftool-perl \
    libmagickwand-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    r-base \
    r-base-dev \
    cmake \
    libgdal-dev \
    libproj-dev \
    libudunits2-dev \
    libgeos-dev 


# Install Python packages from requirements.txt
RUN pip install -r requirements.txt

# Install specific versions of PyTorch
# RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install R packages from r-packages.txt
RUN Rscript -e "install.packages(readLines('r-packages.txt'))"
# Try this because vegan does not want to install in the above line
# RUN Rscript -e "install.packages('vegan', dependencies=TRUE, repos='http://cran.rstudio.com/')"
# Install oSCR from GitHub
RUN Rscript -e "install.packages('remotes', dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN Rscript -e "remotes::install_github('jaroyle/oSCR')"

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*