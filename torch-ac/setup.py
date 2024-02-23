from setuptools import setup, find_packages

setup(
    name="torch_ac",
    version="1.1.0",
    keywords="reinforcement learning, actor-critic, a2c, ppo, multi-processes, gpu",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.6'  # tested on 1.21.6, but it should work with newer versions as well
        "torch"  # tested on 1.13.1+cu117
    ]
)
