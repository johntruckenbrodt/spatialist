from setuptools import setup, find_packages
import platform
import zipfile as zf
import os

# Create .spatialist in HOME - Directory
directory = os.path.join(os.path.expanduser('~'), '.spatialist')

if not os.path.exists(directory):
    os.makedirs(directory)

if platform.system() is 'Windows':
    package_data = {'spatialist': ['pkgs/mod_spatialite/*']}
else:
    package_data = {}

setup(name='spatialist',
      packages=find_packages(),
      include_package_data=True,
      package_data=package_data,
      version='0.2.1',
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

if platform.system() is 'Windows':
    subdir = os.path.join(directory, 'mod_spatialite')
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    mod_spatialite = os.path.join(subdir, 'mod_spatialite.dll')
    if not os.path.isfile(mod_spatialite):
        source_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'spatialist', 'pkgs', 'mod_spatialite')
        print('machine: {}'.format(platform.machine()))
        suffix = 'amd64' if platform.machine().endswith('64') else 'x86'
        source = os.path.join(source_dir, 'mod_spatialite-4.3.0a-win-{}.zip'.format(suffix))
        print('extracting {} to {}'.format(os.path.basename(source), subdir))
        archive = zf.ZipFile(source, 'r')
        archive.extractall(subdir)
        archive.close()
