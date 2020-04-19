from setuptools import setup


with open('requirements.txt', 'rt') as f:
    requirements_list = [req[:-1] for req in f.readlines()]

setup(
    name='wbident',
    version='0.0.1',
    packages=['wbident', 'wbident.core'],
    url='https://gitlab.com/limajj_articles/core/wbident',
    license='',
    author='Jeferson Lima',
    author_email='jefersonjl82@gmail.com',
    description='wbident: White Box Identification',
    install_requires = requirements_list)
