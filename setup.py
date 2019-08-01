from setuptools import setup

setup(
    name='desietcimg',
    version='0.1.dev',
    description='Sky camera and guide image analysis for the DESI exposure-time calculator',
    url='http://github.com/dkirkby/desietcimg',
    author='David Kirkby',
    author_email='dkirkby@uci.edu',
    license='MIT',
    packages=['desietcimg'],
    install_requires=['numpy', 'scipy'],
    include_package_data=False,
    zip_safe=False,
)
