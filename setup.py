from setuptools import find_packages ,setup
from typing import List

HYPEN_DOT ='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements mentioned in the requirements.txt file
    '''
    requirements =[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req 
        in requirements]

        if HYPEN_DOT in requirements:
            requirements.remove(HYPEN_DOT)

    return requirements

setup(
    name='GermanBankCreditRisk',
    version='0.0.1',
    author='Govardhan',
    author_email='govagattu7@gmail.com',
    packages=find_packages(),
    install_requires =get_requirements('requirements.txt')
)