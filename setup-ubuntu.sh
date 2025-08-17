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

# Configure Git
GIT_USERNAME=ludoviclupus
GIT_EMAIL=ludoviclupus@gmail.com
REPO_URL=https://github.com/LudovicLupus/homelab.git
CLONE_DIR=~/homelab

git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_EMAIL"
echo "Git configured with username: $GIT_USERNAME and email: $GIT_EMAIL"

# Create directory if it doesn't exist
mkdir -p "$(dirname "$CLONE_DIR")"

# Clone the repository
git clone "$REPO_URL" "$CLONE_DIR"

echo "Installing Coral Edge TPU Runtime"
# Add the Coral package repository and GPG key to address apt-key deprecation
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/coral.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null

# Update package list and install the Edge TPU runtime and PCIe drivers
sudo apt-get update

# Install kernel headers for DKMS (if available for the running kernel)
if apt-cache show "linux-headers-$(uname -r)" >/dev/null 2>&1; then
  sudo apt-get install -y "linux-headers-$(uname -r)"
fi

sudo apt-get install -y libedgetpu1-std dkms gasket-dkms

# Load Coral PCIe kernel modules now and ensure they load at boot
sudo modprobe gasket || true
sudo modprobe apex || true
echo -e "gasket\napex" | sudo tee /etc/modules-load.d/coral.conf >/dev/null
