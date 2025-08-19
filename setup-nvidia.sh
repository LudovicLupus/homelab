#!/bin/bash

set -e

echo "Starting NVIDIA Setup..."

sudo apt-get update
sudo apt-get purge nvidia-* -y
sudo apt-get autoremove -y
sudo update-pciids

echo "Available NVIDIA drivers:"
ubuntu-drivers devices

read -r -p "Install recommended NVIDIA driver? [y/N]: " install_nvidia
if [[ $install_nvidia =~ ^[Yy]$ ]]; then
  RECOMMENDED_DRIVER=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')
  if [ -n "$RECOMMENDED_DRIVER" ]; then
    echo "Installing $RECOMMENDED_DRIVER..."
    sudo apt-get install "$RECOMMENDED_DRIVER" -y

    # Install corresponding nvidia-utils
    UTILS_VERSION=$(echo "$RECOMMENDED_DRIVER" | grep -o '[0-9]\+' | head -1)
    sudo apt-get install nvidia-utils-"$UTILS_VERSION" -y

    echo "Installing CUDA Toolkit..."
    sudo apt-get install nvidia-cuda-toolkit -y

    echo "NVIDIA setup complete. Testing..."
    nvidia-smi || echo "Reboot required for NVIDIA driver activation"

    echo "Reboot required to fully activate NVIDIA drivers."
    read -r -p "Reboot now? [y/N]: " reboot_now
    if [[ $reboot_now =~ ^[Yy]$ ]]; then
      sudo reboot
    fi
  else
    echo "No recommended driver found. Please install manually."
  fi
else
  echo "Skipping NVIDIA driver installation."
fi
