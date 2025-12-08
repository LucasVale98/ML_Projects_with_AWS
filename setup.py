from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements if req.strip()]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    except FileNotFoundError:
        pass  # se não tiver requirements.txt, ignora
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author="Lucas Vale",
    author_email="lucasvale1998.lv@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.8',
)