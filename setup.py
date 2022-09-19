from setuptools import find_packages, setup

setup(
    name="ray_lightning",
    packages=find_packages(where=".", include="ray_lightning*"),
    version="0.3.0",
    author="Ray Team",
    description="Ray distributed strategies for Pytorch Lightning.",
    long_description="Custom Pytorch Lightning distributed strategies "
    "built on top of distributed computing framework Ray.",
    url="https://github.com/ray-project/ray_lightning_accelerators",
    install_requires=["pytorch-lightning>=1.6,<1.8", "ray"])
