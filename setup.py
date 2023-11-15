from setuptools import find_packages, setup

setup(
    name='beyond-imagenet-accuracy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch==2.0.1',
        'torchvision==0.16.0',
        'pandas==2.1.1',
        'datasets==2.14.5',
        'easyrobust==0.2.4',
        'imagenet_x==0.0.7',
        'open_clip_torch==2.20.0',
        'wandb==0.15.11',
    ],
    dependency_links=[
        'git+https://github.com/modestyachts/ImageNetV2_pytorch#egg=ImageNetV2_pytorch'
    ],
    python_requires='==3.10.*',
)
