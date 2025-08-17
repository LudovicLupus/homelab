#!/bin/bash

echo "Installing Docker and Docker Compose"

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources
# shellcheck disable=SC1091
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

# Update package index and install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to the docker group to run docker without sudo
# Note: You will need to log out and log back in for this to take effect.
if [ -n "$SUDO_USER" ]; then
  sudo usermod -aG docker "$SUDO_USER"
  echo "Added user '$SUDO_USER' to the docker group. To apply the changes, please log out and back in, or run 'newgrp docker'."
else
  echo "Warning: Could not determine the user to add to the docker group. Please add manually with 'sudo usermod -aG \$USER docker'."
fi

# Enable and start the Docker service
sudo systemctl enable docker
sudo systemctl start docker

echo "Docker and Docker Compose installed and enabled successfully."

# Create context for remotely managing docker
docker context create homelab-server --docker "host=ssh://ludoviclupus@192.168.50.183"
echo "Docker context successfully created."
