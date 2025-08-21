#!/bin/bash

set -e

echo "Frigate Storage Setup Script"
echo "============================"
echo "This script will format and mount a dedicated SSD for Frigate storage."
echo ""

# Show available drives
echo "Available storage devices:"
lsblk -d -o NAME,SIZE,MODEL | grep -E "(nvme|sd)"
echo ""

# Get user input for target drive
while true; do
  read -r -p "Enter the target drive (e.g., nvme1n1, sda): " TARGET_DRIVE

  if [ -z "$TARGET_DRIVE" ]; then
    echo "Error: Please enter a drive name."
    continue
  fi

  if [ ! -b "/dev/$TARGET_DRIVE" ]; then
    echo "Error: Drive /dev/$TARGET_DRIVE does not exist."
    continue
  fi

  # Show drive info
  echo ""
  echo "Selected drive: /dev/$TARGET_DRIVE"
  lsblk "/dev/$TARGET_DRIVE"
  echo ""

  read -r -p "WARNING: This will DESTROY ALL DATA on /dev/$TARGET_DRIVE. Continue? [y/N]: " confirm
  if [[ $confirm =~ ^[Yy]$ ]]; then
    break
  else
    echo "Cancelled. Please select a different drive."
    echo ""
  fi
done

# Check if mount point already exists and is in use
MOUNT_POINT="/mnt/frigate-storage"
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  echo "Error: $MOUNT_POINT is already mounted. Please unmount it first."
  exit 1
fi

echo ""
echo "Formatting and mounting /dev/$TARGET_DRIVE..."

# Wipe the drive completely
echo "Wiping existing filesystem signatures..."
wipefs -a "/dev/$TARGET_DRIVE"

# Create new partition table and partition
echo "Creating new GPT partition table..."
parted "/dev/$TARGET_DRIVE" --script -- mklabel gpt
parted "/dev/$TARGET_DRIVE" --script -- mkpart primary ext4 0% 100%

# Wait for partition to be available
sleep 2

# Format as ext4
echo "Formatting as ext4..."
mkfs.ext4 -F -L "frigate-storage" "/dev/${TARGET_DRIVE}p1" 2>/dev/null ||
  mkfs.ext4 -F -L "frigate-storage" "/dev/${TARGET_DRIVE}1"

# Determine correct partition name
PARTITION="/dev/${TARGET_DRIVE}p1"
if [ ! -b "$PARTITION" ]; then
  PARTITION="/dev/${TARGET_DRIVE}1"
fi

# Create mount point
echo "Creating mount point at $MOUNT_POINT..."
mkdir -p "$MOUNT_POINT"

# Mount the partition
echo "Mounting partition..."
mount "$PARTITION" "$MOUNT_POINT"

# Set proper permissions
echo "Setting permissions..."
chown -R root:root "$MOUNT_POINT"
chmod 755 "$MOUNT_POINT"

# Get UUID and add to fstab
echo "Adding to /etc/fstab for automatic mounting..."
UUID=$(blkid "$PARTITION" -s UUID -o value)
if grep -q "$UUID" /etc/fstab; then
  echo "Entry already exists in /etc/fstab"
else
  echo "UUID=$UUID $MOUNT_POINT ext4 defaults 0 2" >>/etc/fstab
  echo "Added to /etc/fstab"
fi

# Create Frigate directory structure
echo "Creating Frigate directory structure..."
mkdir -p "$MOUNT_POINT/recordings"
mkdir -p "$MOUNT_POINT/clips"
mkdir -p "$MOUNT_POINT/exports"
chown -R root:root "$MOUNT_POINT"

# Verify setup
echo ""
echo "Storage setup complete!"
echo "======================"
df -h "$MOUNT_POINT"
echo ""
echo "Frigate storage is now available at: $MOUNT_POINT"
echo "Directory structure:"
ls -la "$MOUNT_POINT"
echo ""
echo "Next steps:"
echo "1. Update docker-compose.yml to use $MOUNT_POINT:/media/frigate"
echo "2. Restart Frigate: docker-compose down && docker-compose up -d"
echo ""
