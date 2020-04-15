from setuptools import setup


with open('requirements.txt', 'rt') as f:
    requirements_list = [req[:-1] for req in f.readlines()]

setup(
    name='wbident',
    version='0.0.1',
    packages=['quore', 'quore.core', 'quore.models', 'quore.filters', 'quore.segments'],
    url='https://gitlab.com/quoretech/ecg-quore',
    license='',
    author='Jeferson Lima',
    author_email='jefersonjl82@gmail.com',
    description='ECG-Quore: Automatic Arrhythmia Classification',
    install_requires = requirements_list)
