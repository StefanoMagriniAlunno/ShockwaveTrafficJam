import sys

from invoke import task  # type: ignore

list_packages = [
    "pytest jupyter",
    "sphinx sphinxcontrib-plantuml esbonio sphinx_rtd_theme",
    "pillow imageio",
    "numpy scipy pandas scikit-learn",  # scientific computing
    "matplotlib seaborn plotly",  # plotting
    "tqdm colorama",  # utilities
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",  # gpu computing
    "pycuda",  # gpu computing
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
