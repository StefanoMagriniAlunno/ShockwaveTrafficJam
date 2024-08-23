# Shockwave Traffic Jam
This is the public GitHub repository of Project 2, entitled "Shockwave Traffic Jam."
Please consult the repository's documentation for instructions on downloading and installing the software.
For further information on the code, please refer to the `docs` folder.

## Get start

To `download` this repository
~~~bash
    cd complete_path/where/install/repository
    # Now we create a directory called `Machine Unlearning` with repository:
    git clone https://github.com/StefanoMagriniAlunno/ShockwaveTrafficJam
~~~

### Prerequisites
The installer checks following system prerequisites:
1. `python3` version `3.10.12` with:
   1. `virtualenv` version `20.26.3`
2. `nvidia-drivers` version >=`550.54.14`
See the file `conf.ini` to set the appropriate parameters:

1. `python3_cmd`: path to the python3 executable (e.g. `/usr/bin/python3`). To find this path type:
~~~bash
    command -v python3
~~~
2. `gpu_uuid`: uuid of your GPU (e.g. `GPU-560e41e1-6e4e-a484-4b92-7d92ebc90683`). To find this uuid type:
~~~bash
    nvidia-smi --query-gpu=name,uuid --format=csv,noheader
~~~
The output is as follower example:
`NVIDIA GeForce GTX 1650, GPU-560e41e1-6e4e-a484-4b92-7d92ebc90683`

To `install` this repository
~~~bash
    cd complete_path/where/install/repository
    cd Machine_Unlearning
    ./repo.sh
~~~

* **Remark** :  The installer does not have administrator permissions, it only changes the local folder.
* **Remark** : You can examine the properties of the installer by typing `./repo.sh --help`.


## Navigate

### Documentation
It is possible consult documentation opening the html file `doc/_build/html/index.html` in your browser
It's contain details about python code in `src`, each file is documentated.

### Console
