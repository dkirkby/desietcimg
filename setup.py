from setuptools import setup, find_packages

setup(
    name='desietcimg',
    version='0.1.dev0',
    description='Sky camera and guide image analysis for the DESI exposure-time calculator',
    url='http://github.com/dkirkby/desietcimg',
    author='David Kirkby',
    author_email='dkirkby@uci.edu',
    license='MIT',
    packages=find_packages(exclude=["tests",]),
    install_requires=['numpy', 'scipy'],
    include_package_data=True, # specified in MANIFEST.in
    zip_safe=False,
    entry_points = {
        'console_scripts': [
            'simskycam=desietcimg.scripts.sky:simulate',
            'ciproc=desietcimg.scripts.guide:ciproc',
            'etccalib=desietcimg.scripts.calib:etccalib',
            'gfadiq=desietcimg.scripts.gfa:gfadiq',
        ],
    }
)
