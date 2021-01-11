import setuptools
from breinforce.constants import VERSION


setuptools.setup(
    name='breinforce',
    version=VERSION,
    author='dmitrynvm',
    author_email='dmitrynvm@gmail.com',
    description='A Toolkit for Reinforcement Learning in Poker',
    long_description='',
    long_description_content_type='text/markdown',
    url='https://www.github.com/dmitrynvm/breinforce',
    keywords=['Reinforcement Learning', 'Poker', 'RL', 'AI'],
    packages=setuptools.find_packages(exclude=('tests',)),
    package_data={},
    install_requires=[
        'numpy>=1.16.6',
        'gym>=0.12.0'
    ],
    extras_require={},
    requires_python='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
