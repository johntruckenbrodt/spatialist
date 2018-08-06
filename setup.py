from setuptools import setup, find_packages
import os

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='spatialist',
      packages=find_packages(),
      include_package_data=True,
      version='0.2.4',
      description='A Python module for spatial data handling',
      classifiers=[
          'Programming Language :: Python',
      ],
      install_requires=['progressbar2',
                        'jupyter',
                        'IPython',
                        'ipywidgets',
                        'matplotlib',
                        'pathos>=0.2',
                        'numpy',
                        'scoop'],
      python_requires='>=2.7.9',
      url='https://github.com/johntruckenbrodt/spatialist.git',
      author='John Truckenbrodt',
      author_email='john.truckenbrodt@uni-jena.de',
      license='MIT',
      zip_safe=False,
      long_description=long_description,
      long_description_content_type='text/markdown')
