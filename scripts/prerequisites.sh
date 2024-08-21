#!/bin/bash

version_ge() {
    # Converte le versioni in array di numeri
    IFS='.' read -r -a ver1 <<< "$1"
    IFS='.' read -r -a ver2 <<< "$2"

    # Trova il numero massimo di segmenti tra le due versioni
    local len1=${#ver1[@]}
    local len2=${#ver2[@]}
    local max_len=$(( len1 > len2 ? len1 : len2 ))

    # Confronta i segmenti delle versioni
    for (( i=0; i<max_len; i++ )); do
        local v1=${ver1[i]:-0}  # Imposta a 0 se il segmento non esiste
        local v2=${ver2[i]:-0}  # Imposta a 0 se il segmento non esiste

        if (( v1 > v2 )); then
            return 0
        elif (( v1 < v2 )); then
            return 1
        fi
    done

    return 0
}



#################################
# Check input arguments
#################################
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ${0##*/} path/to/python3"
    exit 1
fi

#################################
# expected prerequisites
#################################
python3_version="3.10.12"
virtualenv_version="20.26.3"
driver_nvidia_minversion="550.54.14"  # check \url{https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html}


#################################
# get input arguments
#################################
python3_cmd=$1
gpu_uuid=$2

#################################
# Verify prerequisites
#################################

# python3 version
if [ "$($python3_cmd --version | awk '{print $2}')" != "$python3_version" ]; then
    echo
    echo "Python3 version is not correct."
    echo "Excpeted version: $python3_version"
    echo "Installed version: $($python3_cmd --version | awk '{print $2}')"
    echo
    echo
    exit 1
fi

# virtualenv version
if [ "$($python3_cmd -m virtualenv --version | awk '{print $2}')" != "$virtualenv_version" ]; then
    echo
    echo "Virtualenv version is not correct"
    echo "Expected version: $virtualenv_version"
    echo "Installed version: $($python3_cmd -m virtualenv --version)"
    echo
    echo
    exit 1
fi

# check nvidia-smi command
if [ -z "$(command -v nvidia-smi)" ]; then
    echo
    echo "nvidia-smi command not found"
    echo "Please install nvidia driver"
    echo
    echo
    exit 1
fi

# check gpu
if [ -z "$(nvidia-smi --query-gpu=name --id="$gpu_uuid" --format=csv,noheader)" ]; then
    echo
    echo "GPU with UUID $gpu_uuid not found"
    echo "Please check the UUID in the config file"
    echo
    echo
    exit 1
fi

# check driver version
driver_version=$(nvidia-smi --query-gpu=driver_version --id="$gpu_uuid" --format=csv,noheader)
if ! dpkg --compare-versions "$driver_version" ge "$driver_nvidia_minversion"; then
    echo
    echo "Nvidia driver version is not correct"
    echo "Expected version: >=$driver_nvidia_minversion"
    echo "Installed version: $driver_version"
    echo "Please check the UUID or the driver version"
    echo
    echo
    exit 1
fi
