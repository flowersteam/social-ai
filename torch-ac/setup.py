from setuptools import setup, find_packages

setup(
    name="torch_ac",
    version="1.1.0",
    keywords="reinforcement learning, actor-critic, a2c, ppo, multi-processes, gpu",
    packages=find_packages(),
    install_requires=[
        "numpy==1.17.0",
        #"torch>=1.10.2"
        "torch==1.10.2"
        #"torch==1.10.2+cu102"
    ]
)
