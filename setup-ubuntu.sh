#!/bin/bash

set -e

echo "Updating and Upgrading System Packages"
apt-get update && apt-get upgrade -y

echo "Installing Essential System & Development Tools"
apt-get install -y \
  apt-transport-https \
  build-essential \
  ca-certificates \
  curl \
  git \
  gnupg \
  htop \
  lsb-release \
  net-tools \
  software-properties-common \
  tree \
  tmux \
  unzip \
  vim \
  wget \
  zsh
