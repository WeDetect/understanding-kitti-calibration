from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name='point_cloud_handler',
    version='0.1.0',
    packages=find_packages(), 
    install_requires=requirements,    
    author='Gil Heller',
    description='point cloud trasnformations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
)
