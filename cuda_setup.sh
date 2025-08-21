#!/bin/bash
# install cuda 12.4 and nvidia driver 570
apt purge nvidia*
apt remove --autoremove nvidia-*
rm /etc/apt/sources.list.d/cuda*
apt autoremove && sudo apt autoclean
add-apt-repository ppa:graphics-drivers/ppa
apt update
apt install nvidia-driver-570 # or a suitable version
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
apt update
apt install cuda-toolkit-12-4
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
