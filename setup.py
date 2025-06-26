from setuptools import setup, find_packages

# Read requirements from file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='SARAD',
    version='0.1.0',
    description='Synthetic Aperture Radar Anomaly Detection',
    author='Talex Maxim',
    author_email='talex@example.com',
    url='https://github.com/taimax13/SARAD',
    packages=find_packages(include=['sarad', 'sarad.*']),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
