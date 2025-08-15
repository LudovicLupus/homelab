#!/bin/bash

# ==============================================================================
# Ubuntu Server Setup Script for an NVIDIA AI/ML System
# ==============================================================================
# This script documents the steps to configure a fresh Ubuntu Server install,
# install the recommended NVIDIA drivers, and set up the CUDA Toolkit.
#
# Author: ludoviclupus
# Version: 1.0
# Last Updated: 2025-08-15
# ==============================================================================

# --- Stop script on any error ---
set -e

echo "🚀 Starting AI Server Setup..."

# ------------------------------------------------------------------------------
# SECTION 1: INITIAL SYSTEM UPDATE
# ------------------------------------------------------------------------------
# First, ensure all existing system packages are up-to-date.
echo "🔄 [1/4] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y
echo "✅ System packages are up-to-date."


# ------------------------------------------------------------------------------
# SECTION 2: NVIDIA DRIVER INSTALLATION (UBUNTU REPOSITORY METHOD)
# ------------------------------------------------------------------------------
# NOTE: We are intentionally NOT using the downloaded NVIDIA .run files:
# - NVIDIA-Linux-x86_64-580.76.05.run
# - cuda_13.0.0_580.65.06_linux.run
# The .run installer detected a conflict with Ubuntu's package manager.
# The 'apt' method is preferred for stability and automatic handling of kernel updates.

echo "🧹 [2/4] Purging any existing NVIDIA drivers for a clean install..."
# The -y flag automatically confirms the action.
sudo apt-get purge nvidia-* -y
sudo apt autoremove -y
echo "✅ Old drivers purged."

echo "🔍 Finding the recommended NVIDIA driver for your hardware..."
# This command probes the hardware and lists available drivers.
ubuntu-drivers devices

echo "👉 Please identify the 'recommended' driver from the list above."
echo "Example: 'nvidia-driver-575-open'"
read -p "Press Enter to continue once you've noted the recommended driver..."

# On our system, the recommended driver was 'nvidia-driver-575-open'.
# We will proceed with installing that package.
echo "📦 Installing the recommended NVIDIA driver (nvidia-driver-575-open)..."
sudo apt install nvidia-driver-575-open -y
echo "✅ NVIDIA driver installed successfully."


# ------------------------------------------------------------------------------
# SECTION 3: CUDA TOOLKIT INSTALLATION
# ------------------------------------------------------------------------------
# With the driver installed, we now install the full toolkit (compilers, libraries).
# This package integrates with the driver we just installed via apt.
echo "🛠️ [3/4] Installing the NVIDIA CUDA Toolkit..."
sudo apt install nvidia-cuda-toolkit -y
echo "✅ CUDA Toolkit installed successfully."


# ------------------------------------------------------------------------------
# SECTION 4: VERIFICATION & REBOOT
# ------------------------------------------------------------------------------
echo "🖥️ [4/4] Verifying installations..."

# Check that the NVIDIA driver is loaded and the GPU is recognized.
echo "Running 'nvidia-smi'..."
nvidia-smi

# Check that the CUDA compiler (nvcc) is available.
echo "Running 'nvcc --version'..."
nvcc --version

echo "🎉 --- SETUP COMPLETE --- 🎉"
echo "A reboot is required to ensure all components are properly loaded."
read -p "Press Enter to reboot the system now."
sudo reboot