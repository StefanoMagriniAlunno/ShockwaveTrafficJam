#!/bin/bash

#################################
# Params
#################################
config_file="conf.ini"
venv=".venv"
venv_pip_version="24.2"
cache_dir=".dist"
base_packages=("wheel" "invoke" "pre-commit" "flake8" "doc8" "mypy" "black" "autoflake" "isort" "shellcheck-py")


#################################
# Help text
#################################
show_help() {
    echo "Usage: ${0##*/} [-h|--help] [-b|--base] [-i|--install] [-r|--reinstall] [-u|--update] [package [package ...]]"
    echo
    echo "This script initialize the repository."
    echo "Check conf.ini to set parameters of the installation."
    echo
    echo "Options:"
    echo "  -h, --help            Show this help message and exit"
    echo "  -b, --base            Prepare basic mode of the repository"
    echo "  -i, --install         Install all the contents of the repository"
    echo "  -r, --reinstall       Reinstall all the contents of the repository"
    echo "  -u, --update          Update the repository with the given package"
    echo "                        -u [package [package ...]]"
    echo
    echo
}

#################################
# Base installation
#################################
base() {
    # read config file
    if [ -f "$config_file" ]; then
        python3_cmd=$(grep "python" "$config_file" | cut -d'=' -f2)
        gpu_uuid=$(grep "gpu_uuid" "$config_file" | cut -d'=' -f2)
    else
        echo
        echo -e "\e[31mERROR\e[0m Config file $config_file not found"
        echo
        echo
        exit 1
    fi

    # report
    echo -e "\e[36mINFO\e[0m Used python3 command: $python3_cmd"
    echo -e "\e[36mINFO\e[0m Used GPU UUID: $gpu_uuid"

    # check prerequisites
    if ! ./scripts/prerequisites.sh "$python3_cmd" "$gpu_uuid"; then
        echo
        echo -e "\e[31mERROR\e[0m Prerequisites check failed"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Prerequisites check passed"

    # Prepare repository to install
    mkdir -p "$cache_dir"

    # remove virtual environment if exists
    if [ -d "$venv" ]; then
        rm -rf "$venv"
    fi
    # remove cache directory if exists
    if [ -d "$cache_dir" ]; then
        rm -rf "$cache_dir"
    fi

    # make python environment with virtualenv
    if ! "$python3_cmd" -m virtualenv "$venv" --prompt="DLAI" --pip "$venv_pip_version" --quiet; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to create virtual environment or save logs"
        echo
        echo
        exit 1
    fi
    python3_cmd="$(pwd)/$venv/bin/python3"
    if ! "$python3_cmd" -m pip download --no-cache-dir --dest "$cache_dir" --quiet "${base_packages[@]}"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to download packages"
        echo
        echo
        exit 1
    fi
    if ! "$python3_cmd" -m pip install --compile --no-index --find-links="$cache_dir" --quiet "${base_packages[@]}"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to install packages"
        echo
        echo
        exit 1
    fi
    pre_commit_cmd="$(pwd)/$venv/bin/pre-commit"
    invoke_cmd="$(pwd)/$venv/bin/invoke"
    echo -e "\e[36mINFO\e[0m Virtual environment created"

    # download contents
    if ! "$invoke_cmd" download --cache "$cache_dir"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to download contents"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Contents downloaded"

    echo
    echo "You can activate the virtual environment with the following command:"
    echo "    source $venv/bin/activate"
    echo
    echo
}

#################################
# Full installation
#################################
install() {

    # Prepare repository to install
    mkdir -p data/db
    mkdir -p data/models

    # make python environment with virtualenv
    python3_cmd="$(pwd)/$venv/bin/python3"
    pre_commit_cmd="$(pwd)/$venv/bin/pre-commit"
    invoke_cmd="$(pwd)/$venv/bin/invoke"

    # report
    echo -e "\e[36mINFO\e[0m Environment: $venv"
    echo -e "\e[36mINFO\e[0m Python3 command: $python3_cmd"
    echo -e "\e[36mINFO\e[0m Pre-commit command: $pre_commit_cmd"
    echo -e "\e[36mINFO\e[0m Invoke command: $invoke_cmd"

    # install contents
    if ! "$invoke_cmd" install --cache "$cache_dir"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to install contents"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Contents installed"

    # build repository
    if ! "$invoke_cmd" build --sphinx "$venv/bin/sphinx-build"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to build contents"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Contents built"

    # prepare repository
    if ! "$pre_commit_cmd" install > /dev/null; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to install pre-commit"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Pre-commit installed"

    echo
    echo "You can activate the virtual environment with the following command:"
    echo "    source $venv/bin/activate"
    echo
    echo
}

#################################
# re-installation
#################################
reinstall() {
    # remove virtual environment
    if [ -d "$venv" ]; then
        rm -rf "$venv"
    fi

    # read config file
    if [ -f "$config_file" ]; then
        python3_cmd=$(grep "python" "$config_file" | cut -d'=' -f2)
        gpu_uuid=$(grep "gpu_uuid" "$config_file" | cut -d'=' -f2)
    else
        echo
        echo -e "\e[31mERROR\e[0m Config file $config_file not found"
        echo
        echo
        exit 1
    fi

    # report
    echo -e "\e[36mINFO\e[0m Used python3 command: $python3_cmd"
    echo -e "\e[36mINFO\e[0m Used GPU UUID: $gpu_uuid"

    # check prerequisites
    if ! ./scripts/prerequisites.sh "$python3_cmd" "$gpu_uuid"; then
        echo
        echo -e "\e[31mERROR\e[0m Prerequisites check failed"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Prerequisites check passed"

    # Prepare repository to install
    mkdir -p "$cache_dir"

    # make python environment with virtualenv
    if ! "$python3_cmd" -m virtualenv "$venv" --prompt="DLAI" --pip "$venv_pip_version" --quiet; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to create virtual environment or save logs"
        echo
        echo
        exit 1
    fi
    python3_cmd="$(pwd)/$venv/bin/python3"
    if ! "$python3_cmd" -m pip install --compile --no-index --find-links="$cache_dir" --quiet "${base_packages[@]}"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to install packages"
        echo
        echo
        exit 1
    fi
    pre_commit_cmd="$(pwd)/$venv/bin/pre-commit"
    invoke_cmd="$(pwd)/$venv/bin/invoke"
    echo -e "\e[36mINFO\e[0m Virtual environment created"

    echo -e "\e[36mINFO\e[0m Reinstalling contents..."
    install > /dev/null
    echo -e "\e[36mINFO\e[0m Contents reinstalled"

    echo
    echo "You can activate the virtual environment with the following command:"
    echo "    source $venv/bin/activate"
    echo
    echo
}

#################################
# add package
#################################
add_pkg() {
    pkg=$1
    python3_cmd="$(pwd)/$venv/bin/python3"

    if ! "$python3_cmd" -m pip download --no-cache-dir --dest "$cache_dir" --quiet "$pkg"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to download package $pkg"
        echo
        echo
        exit 1
    fi
    if ! "$python3_cmd" -m pip install --compile --no-index --find-links="$cache_dir" --quiet "$pkg"; then
        echo
        echo -e "\e[31mERROR\e[0m Failed to install package $pkg"
        echo
        echo
        exit 1
    fi
    echo -e "\e[36mINFO\e[0m Package $pkg installed"
}


if [ "$#" == 0 ]; then
    show_help
    exit 1
elif [ "$#" == 1 ]; then
    if [ "$1" == "-b" ] || [ "$1" == "--base" ]; then
        base
        exit 0
    elif [ "$1" == "-i" ] || [ "$1" == "--install" ]; then
        install
        exit 0
    elif [ "$1" == "-r" ] || [ "$1" == "--reinstall" ]; then
        reinstall
        exit 0
    elif [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        show_help
        exit 0
    else
        echo
        echo "option  $1  not found"
        exit 1
    fi
else
    if [ "$1" == "-u" ] || [ "$1" == "--update" ]; then
        for i in $(seq 2 "$#"); do
            pkg="${!i}"
            add_pkg "$pkg"
        done
    else
        echo
        echo "invalid option"
        exit 1
    fi
fi
