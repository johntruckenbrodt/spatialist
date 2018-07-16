# spatialist
[![Linux Build Status][1]][2] [![Windows Build status][3]][4] [![Coverage Status][5]][6] [![Documentation Status][7]][8] [![Binder][9]][10]

### A Python module for spatial data handling

This package is currently being extracted from the [pyroSAR](https://github.com/johntruckenbrodt/pyroSAR) project.

This software used to run on a local server and is currently being restructured into a Python package.
Not everything is working properly, stay tuned...

### Documentation
A description of spatialist's functionality can be found [here][8].

### Tutorial
We are currently developing a tutorial jupyter notebook [spatialist_tutorial.ipynb][11]. 
You can interactively use it by launching binder on the top.

### Installation of dependencies
If you are using Windows, the easiest way to work with spatialist and Python in general is by using 
[Anaconda](https://www.anaconda.com/download/). It comes with all basic requirements of spatialist.
The more specific instructions below are intended for Linux users.
##### GDAL
spatialist requires GDAL version 2.1 with GEOS and PROJ4 as dependencies as well as the GDAL Python binding. 
Alternatively, one can use <a href="https://github.com/nextgis/pygdal">pygdal</a>, 
a virtualenv and setuptools friendly version of standard GDAL python bindings.
###### Ubuntu
Starting with release Yakkety (16.10), Ubuntu comes with GDAL >2.1. 
See <a href="https://launchpad.net/ubuntu/yakkety/amd64/gdal-bin">here</a>. 
You can install it like this:
```bash
sudo apt-get install python-gdal python3-gdal gdal-bin
```
For older Ubuntu releases you can add the ubuntugis repository to apt prior to installation to install version >2.1:
```sh
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
```
This way the required dependencies (GEOS and PROJ4 in particular) are also installed.
You can check the version by typing:
```sh
gdalinfo --version
```
###### Debian
Starting with Debian 9 (Stretch) GDAL is available in version >2.1 in the official repository.
###### Building from source
Alternatively, you can build GDAL and the dependencies from source. The script `spatialist/install/install_deps.sh` 
gives specific instructions on how to do it. It is not yet intended to run this script via shell, but rather to 
follow the instructions step by step.
##### SQLite + SpatiaLite
While sqlite3 and its Python binding are usually already installed, the spatialite extension needs to be 
added. Two packages exist, libspatialite and mod_spatialite. Both can be used by spatialist.
mod_spatialite has been found to be easier to setup with sqlite and can be installed via apt:
```sh
sudo apt-get install libsqlite3-mod-spatialite
```

The following can be run in Python to test the needed functionality:
```Python
import sqlite3
# setup an in-memory database
con=sqlite3.connect(':memory:')
# enable loading extensions and load spatialite
con.enable_load_extension(True)
try:
    con.load_extension('mod_spatialite.so')
except sqlite3.OperationalError:
    con.load_extension('libspatialite.so')
```
In case loading extensions is not permitted you might need to install the package `pysqlite2`. 
See the script `spatialist/install/install_deps.sh` for instructions. 
There you can also find instructions on how to install spatialite from source.
To test `pysqlite2` you can import it as follows and then run the test above:
```Python
from pysqlite2 import dbapi2 as sqlite3
```
Installing this package is likely to cause problems with the sqlite3 library installed on the system. 
Thus, it is safer to build a static sqlite3 library for it (see installation script).
### Installation of spatialist
For the installation we need the Python tool pip and the version control system git. On Windows pip is 
installed together with Anaconda, git can be downloaded from [here](https://git-scm.com/downloads).
On Linux you can easily install both via command line:
```sh
sudo apt-get install python-pip
sudo apt-get install git
```
Once everything is set up, spatialist is ready to be installed:
```sh
sudo pip install git+https://github.com/johntruckenbrodt/spatialist.git
```
On Windows you need to use the Anaconda Prompt and leave out `sudo` in the above command.


[1]: https://www.travis-ci.org/johntruckenbrodt/spatialist.svg?branch=master
[2]: https://www.travis-ci.org/johntruckenbrodt/spatialist
[3]: https://ci.appveyor.com/api/projects/status/3nxj2nnmp21ig860?svg=true
[4]: https://ci.appveyor.com/project/johntruckenbrodt/spatialist
[5]: https://coveralls.io/repos/github/johntruckenbrodt/spatialist/badge.svg?branch=master
[6]: https://coveralls.io/github/johntruckenbrodt/spatialist?branch=master
[7]: https://readthedocs.org/projects/spatialist/badge/?version=latest
[8]: https://spatialist.readthedocs.io/en/latest/?badge=latest
[9]: https://mybinder.org/badge.svg
[10]: https://mybinder.org/v2/gh/johntruckenbrodt/spatialist/master?filepath=spatialist_tutorial.ipynb
[11]: https://github.com/johntruckenbrodt/spatialist/blob/master/spatialist_tutorial.ipynb