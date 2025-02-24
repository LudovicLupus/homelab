#! /bin/bash

sudo apt update && sudo apt upgrade -y

:'
What is qemu-guest-agent and Why Use It?
The QEMU Guest Agent is a service that runs inside the VM and helps Proxmox interact with it more efficiently.

Benefits of qemu-guest-agent:
Graceful shutdown & reboots from Proxmox (instead of force-stopping).
Accurate resource monitoring (CPU, RAM usage inside the VM).
Better disk performance (optimizes virtual disk I/O).
Improved network handling (better IP discovery from Proxmox).
'
sudo apt install qemu-guest-agent -y
sudo systemctl enable --now qemu-guest-agent

: 'Docker installation'
# Add dependencies
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
# Add Docker’s Official GPG Key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# Add the Docker Repositorecho "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
# Install Docker & Docker Compose
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y
# Verify Docker Installation
sudo systemctl enable --now docker
docker --version
# Add Your User to the Docker Group. By default, Docker commands require sudo.
# If you want to run Docker without typing sudo every time:
sudo usermod -aG docker $USER
newgrp docker

:'Configure git'
git config --global user.name "ludoviclupus"
git config --global user.email "ludoviclupus@gmail.com"
git clone https://github.com/LudovicLupus/homelab.git

