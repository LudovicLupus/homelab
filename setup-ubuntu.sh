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

# Coral TPU and IOMMU configuration has been moved to a dedicated script for AMD systems:
echo "To configure Coral TPU (PCIe) and AMD IOMMU, run: ./homelab/setup-coral-amd.sh"
