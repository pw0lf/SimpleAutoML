from setuptools import setup, find_packages

setup(
    name='simpleaml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'optuna',
        'numpy'
    ],
    author='Philipp Wolf',
    author_email='philippwolf99@icloud.com',
    description='Simple AutoML package',
    license='MIT',
    url='https://github.com/pw0lf/SimpleAutoML',
    
)