from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filte_path:str) -> List[str]:
    '''
        Will return a List of Requirement from requirement.txt
    '''
    requirement = []

    with open(filte_path,'r') as file:
        requirement = file.readlines()
        requirement = [req.replace("\n","") for req in requirement]

    if HYPHEN_E_DOT in requirement:
        requirement.remove(HYPHEN_E_DOT)
    
    return requirement

setup(
    name = "Student Performance Prediction",
    version = "0.0.1",
    author= "Ankit Zanzmera",
    author_email="22msrds052@jainuniversity.ac.in",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)