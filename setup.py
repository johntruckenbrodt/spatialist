from setuptools import setup, find_packages

setup(name='spatialist',
      packages=find_packages(),
      include_package_data=True,
      version='0.1',
      description='A Python module for spatial data handling',
      classifiers=[
          'Programming Language :: Python',
      ],
      install_requires=['progressbar2',
                        'matplotlib',
                        'pathos>=0.2',
                        'numpy',
                        'scoop'],
      url='https://github.com/johntruckenbrodt/spatialist.git',
      author='John Truckenbrodt',
      author_email='john.truckenbrodt@uni-jena.de',
      license='MIT',
      zip_safe=False)
