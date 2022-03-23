#!/usr/bin/env python3
# Copyright 2021 Anonymous Author All Rights Reserved.

""" Setup vox2cortex """

from setuptools import setup

setup(
    name='vox2cortex',
    version='0.1.0',
    description='Cortical Surface Reconstruction from MRI with Geometric Deep Learning',
    url='https://gitlab.lrz.de/ga63wus/vox2cortex',
    author='Anonymous Author',
    author_email='anonymous@mail.com',
    license='GNU GPLv3',
    packages=['vox2cortex'],
    keywords=[
        "Surface reconstruction",
        "3D vision",
        "Brain segmentation",
        "Geometric deep learning",
        "Computer vision",
        "AI in medicine"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires="==3.8.*",
    install_requires=[
        'deprecated==1.2.13',
        'nibabel==3.2.1',
        'numpy==1.21.4',
        'torch==1.7.1',
        'tqdm==4.62.3',
        'trimesh==3.9.35',
        'wandb==0.12.6',
        'scikit-image==0.18.1',
    ]
)


