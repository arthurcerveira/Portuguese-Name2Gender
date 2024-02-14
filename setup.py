from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='portugese-name2gender',
    version='0.0.1',
    description='Predict the gender of any PT-BR name',
    long_description=long_description,
    url='https://github.com/arthurcerveira/Portuguese-Name2Gender',
    author='Arthur Cerveira',
    author_email='arthurcerveira@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['pt_name2gender'],
    install_requires=required,
    # Include pt_name2gender/data and pt_name2gender/models in the package
    package_data={'pt_name2gender': ['data/*', 'model/*']},
    include_package_data=True,
)
