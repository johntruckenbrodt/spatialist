from setuptools import setup, find_packages
import os

# Create .spatialist in HOME - Directory
directory = os.path.join(os.path.expanduser('~'), '.spatialist')

if not os.path.exists(directory):
    os.makedirs(directory)

setup(name='spatialist',
      packages=find_packages(),
      include_package_data=True,
      version='0.2.3',
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
      long_description='an in-depth package description can be found on GitHub '
                       '[here]("https://github.com/johntruckenbrodt/spatialist")')
