from setuptools import setup, find_packages
import os

directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(directory, 'requirements.txt'), 'r') as req:
    requirements = req.read().split('\n')

setup(name='spatialist',
      packages=find_packages(),
      include_package_data=True,
      setup_requires=['setuptools_scm'],
      use_scm_version=True,
      description='A Python module for spatial data handling',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
      ],
      install_requires=requirements,
      extras_require={'testing': ['pytest'],
                      'docs': ['sphinx', 'sphinxcontrib-bibtex<2.0.0']},
      python_requires='>=3.0',
      url='https://github.com/johntruckenbrodt/spatialist.git',
      author='John Truckenbrodt',
      author_email='john.truckenbrodt@uni-jena.de',
      license='MIT',
      zip_safe=False,
      long_description=long_description,
      long_description_content_type='text/markdown')
