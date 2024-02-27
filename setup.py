from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    lst_packages = []
    with open(file_path) as packages:
        lst_packages = packages.readlines()
        lst_packages = [pkg.replace("\n","") for pkg in lst_packages]

        if HYPEN_E_DOT in lst_packages:
            lst_packages.remove(HYPEN_E_DOT)
    return lst_packages

setup(
    name="mlprojects",
    version="0.0.1",
    author="Sana Kazi",
    author_email="sanabhatkar@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)