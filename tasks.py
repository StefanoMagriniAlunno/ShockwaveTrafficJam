import sys

from invoke import task  # type: ignore

list_packages = [
    # base
    "pytest jupyter",
    "sphinx sphinxcontrib-plantuml esbonio sphinx_rtd_theme",
    # suite torch
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
    # suite cuda
    "pycuda",
    "--extra-index-url=https://pypi.nvidia.com cudf-cu12==24.8.* cuml-cu12==24.8.*",
    # scientifing computing
    "numpy scipy pandas scikit-learn",
    # data visualization
    "matplotlib seaborn plotly",
    # utilities
    "tqdm colorama",
]


@task
def download(c, cache: str):
    """download contents"""

    for pkg in list_packages:
        c.run(
            f"{sys.executable} -m pip download --no-cache-dir --dest {cache} --quiet {pkg} "
        )


@task
def install(c, cache: str):
    """Install contests"""

    for pkg in list_packages:
        c.run(
            f"{sys.executable} -m pip install --compile --no-index --find-links={cache} --quiet {pkg} "
        )


@task
def build(c, sphinx: str):
    """Build contents"""

    c.run("rm -rf doc/_build ")
    c.run("mkdir -p doc/_build ")
    c.run("mkdir -p doc/_static ")
    c.run("mkdir -p doc/_templates ")
    c.run(f"{sphinx} -b html doc doc/_build/html --quiet ")
