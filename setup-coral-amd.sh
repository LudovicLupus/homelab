#!/bin/bash
# AMD-focused Coral Edge TPU (PCIe) setup for Ubuntu 24.04
# - Purges any DKMS/gasket remnants
# - Installs only the Edge TPU runtime
# - Enables AMD IOMMU for PCIe passthrough
# - Loads in-kernel 'apex' driver and configures auto-load
set -e

echo "=== Coral TPU (PCIe) setup for AMD on Ubuntu 24.04 ==="

# 1) Clean slate: purge any previously installed DKMS/gasket/Coral packages
echo "[1/5] Purging any existing Coral TPU and DKMS packages (clean slate)"
sudo apt-get remove --purge -y gasket-dkms dkms libedgetpu1-std libedgetpu1-max libedgetpu-dev || true
sudo apt-get autoremove -y || true
sudo rm -rf /var/lib/dkms/gasket || true
sudo find /lib/modules/"$(uname -r)"/ -type f \( -name "*gasket*" -o -name "*apex*" \) -exec rm -f {} + || true
sudo depmod -a || true

# 2) Enable AMD IOMMU (required for reliable PCIe device handling)
echo "[2/5] Configuring AMD IOMMU (amd_iommu=on iommu=pt) in GRUB if missing"
GRUB_FILE="/etc/default/grub"
IOMMU_ARGS="amd_iommu=on iommu=pt"
REBOOT_REQUIRED=false
if [ -f "$GRUB_FILE" ]; then
  if grep -q '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB_FILE"; then
    CURRENT=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB_FILE" | cut -d'"' -f2 || true)
    if ! echo "$CURRENT" | grep -q "$IOMMU_ARGS"; then
      NEW="$CURRENT $IOMMU_ARGS"
      sudo sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=\".*\"|GRUB_CMDLINE_LINUX_DEFAULT=\"$NEW\"|" "$GRUB_FILE"
      sudo update-grub
      REBOOT_REQUIRED=true
      echo "Added IOMMU args to GRUB."
    else
      echo "IOMMU args already present in GRUB."
    fi
  else
    echo "GRUB_CMDLINE_LINUX_DEFAULT=\"$IOMMU_ARGS\"" | sudo tee -a "$GRUB_FILE" >/dev/null
    sudo update-grub
    REBOOT_REQUIRED=true
    echo "Initialized GRUB_CMDLINE_LINUX_DEFAULT with IOMMU args."
  fi
else
  echo "Warning: $GRUB_FILE not found; skipping GRUB configuration."
fi

# 3) Add Coral APT repository and install the runtime
echo "[3/5] Adding Coral APT repo and installing libedgetpu1-std (runtime only)"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/coral.gpg
echo "deb [signed-by=/etc/apt/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list >/dev/null
sudo apt-get update
sudo apt-get install -y libedgetpu1-std

# 4) Load in-kernel 'apex' driver now and ensure it loads at boot
echo "[4/5] Loading 'apex' kernel module and configuring auto-load"
sudo modprobe apex || true
echo -e "apex" | sudo tee /etc/modules-load.d/coral.conf >/dev/null

# Refresh udev and show device status
sudo udevadm control --reload || true
sudo udevadm trigger || true

# 5) Verification
echo "[5/5] Verifying Coral PCIe visibility and device nodes"
echo "- lspci:"
lspci -nn | grep -Ei "1ac1|apex|google" || echo "No Coral PCIe device detected by lspci (check seating/PCIe slot)."
echo "- device nodes:"
ls -l /dev/apex* 2>/dev/null || echo "No /dev/apex* yet."

if [ "$REBOOT_REQUIRED" = true ]; then
  echo ""
  echo "IOMMU configuration was updated. Please reboot this host to apply GRUB changes,"
  echo "then re-run this script once after reboot if /dev/apex* is still missing."
fi

echo ""
echo "If running Frigate in Docker, pass the device to the container, e.g.:"
echo "  docker run ... --device /dev/apex_0:/dev/apex_0 ... ghcr.io/blakeblackshear/frigate:stable"
echo "Done."
