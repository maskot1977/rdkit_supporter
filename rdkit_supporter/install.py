import os
import shutil
import subprocess
import sys
from logging import INFO, StreamHandler, getLogger

import requests

logger = getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(INFO)


def to_google_colaboratory():
    subprocess.call("wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh")
    subprocess.call("chmod +x Miniconda3-latest-Linux-x86_64.sh")
    subprocess.call("bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local")
    subprocess.call("conda install -q -y -c rdkit rdkit python=3.7")
    import sys
    sys.path.append('/usr/local/lib/python3.7/site-packages/')

    
def from_miniconda(
    chunk_size=4096,
    file_name="Miniconda3-4.7.12-Linux-x86_64.sh",
    url_base="https://repo.continuum.io/miniconda/",
    conda_path=os.path.expanduser(os.path.join("~", "miniconda")),
    rdkit_version=None,
    add_python_path=True,
    force=False,
):
    """install rdkit from miniconda
    ```
    import rdkit_installer
    rdkit_installer.install()
    ```
    """

    python_path = os.path.join(
        conda_path,
        "lib",
        "python{0}.{1}".format(*sys.version_info),
        "site-packages",
    )

    if add_python_path and python_path not in sys.path:
        logger.info("add {} to PYTHONPATH".format(python_path))
        sys.path.append(python_path)

    if os.path.isdir(os.path.join(python_path, "rdkit")):
        logger.info("rdkit is already installed")
        if not force:
            return

        logger.info("force re-install")

    url = url_base + file_name
    python_version = "{0}.{1}.{2}".format(*sys.version_info)

    logger.info("python version: {}".format(python_version))

    if os.path.isdir(conda_path):
        logger.warning("remove current miniconda")
        shutil.rmtree(conda_path)
    elif os.path.isfile(conda_path):
        logger.warning("remove {}".format(conda_path))
        os.remove(conda_path)

    logger.info("fetching installer from {}".format(url))
    res = requests.get(url, stream=True)
    res.raise_for_status()
    with open(file_name, "wb") as f:
        for chunk in res.iter_content(chunk_size):
            f.write(chunk)
    logger.info("done")

    logger.info("installing miniconda to {}".format(conda_path))
    subprocess.check_call(["bash", file_name, "-b", "-p", conda_path])
    logger.info("done")

    logger.info("installing rdkit")
    subprocess.check_call(
        [
            os.path.join(conda_path, "bin", "conda"),
            "install",
            "--yes",
            "-c",
            "rdkit",
            # "python=={}".format(python_version),
            "rdkit" if rdkit_version is None else "rdkit=={}".format(rdkit_version),
        ]
    )
    logger.info("done")

    import rdkit

    logger.info("rdkit-{} installation finished!".format(rdkit.__version__))
