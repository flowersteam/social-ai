from setuptools import setup

setup(
    name='gym_minigrid',
    version='1.0.1',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=['gym_minigrid', 'gym_minigrid.envs'],
    install_requires=[
        'gym==0.18.0',  # must be 0.18.0
        'numpy>=1.21.6'  # tested on 1.21.6, but it should work with newer versions as well
    ]
)
