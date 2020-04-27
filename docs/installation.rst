Installation
============
The most convenient way to install spatialist is by using conda:

.. code-block:: bash

    conda install --channel conda-forge spatialist

See below for more detailed Linux installation instructions outside of the Anaconda framework.

Installation of dependencies
----------------------------

GDAL
~~~~
spatialist requires GDAL version >=2.1 built with GEOS and PROJ4 as dependency as well as the GDAL Python binding.
Alternatively, one can use `pygdal <https://github.com/nextgis/pygdal>`_,
a virtualenv and setuptools friendly version of standard GDAL python bindings.

**Ubuntu**

Starting with release Yakkety (16.10), Ubuntu comes with GDAL >2.1.
See `here <https://launchpad.net/ubuntu/yakkety/amd64/gdal-bin>`_.
You can install it like this:

.. code-block:: bash

    sudo apt-get install python-gdal python3-gdal gdal-bin

For older Ubuntu releases you can add the ubuntugis repository to apt prior to installation to install
version >2.1:

.. code-block:: bash

    sudo add-apt-repository ppa:ubuntugis/ppa
    sudo apt-get update

This way the required dependencies (GEOS and PROJ4 in particular) are also installed.
You can check the version by typing:

.. code-block:: bash

    gdalinfo --version

**Debian**

Starting with Debian 9 (Stretch) GDAL is available in version >2.1 in the official repository.

**Building from source**

Alternatively, you can build GDAL and the dependencies from source. The script `spatialist/install/install_deps.sh`
gives specific instructions on how to do it. It is not yet intended to run this script via shell, but rather to
follow the instructions step by step.

SQLite + SpatiaLite
~~~~~~~~~~~~~~~~~~~
**Windows**


While sqlite3 and its Python binding are usually already installed, the spatialite extension needs to be
added. Two packages exist, libspatialite and mod_spatialite. Both can be used by spatialist.
It is strongly recommended to use Ubuntu >= 16.04 (Xenial) or Debian >=9 (Stretch),
which offer the package `libsqlite3-mod-spatialite`. This package is specifically intended to only serve as an
extension to `sqlite3` and can be installed like this:

.. code-block:: bash

    sudo apt-get install libsqlite3-mod-spatialite


After installation, the following can be run in Python to test the needed functionality:

.. code-block:: python

    import sqlite3
    # setup an in-memory database
    con = sqlite3.connect(':memory:')
    # enable loading extensions and load spatialite
    con.enable_load_extension(True)
    try:
        con.load_extension('mod_spatialite.so')
    except sqlite3.OperationalError:
        con.load_extension('libspatialite.so')

In case loading extensions is not permitted you might need to install the package `pysqlite2`.
See the script `spatialist/install/install_deps.sh` for instructions.
There you can also find instructions on how to install spatialite from source.
To test `pysqlite2` you can import it as follows and then run the test above:

.. code-block:: python

    from pysqlite2 import dbapi2 as sqlite3

Installing this package is likely to cause problems with the sqlite3 library installed on the system.
Thus, it is safer to build a static sqlite3 library for it (see installation script).

Installation of spatialist
--------------------------
For the installation we need the Python tool pip.

.. code-block:: bash

    sudo apt-get install python-pip

Once everything is set up, spatialist is ready to be installed. You can install stable releases like this:

.. code-block:: bash

    python -m pip install spatialist

or the latest GitHub master branch using git like this:

.. code-block:: bash

    sudo apt-get install git
    sudo python -m pip install git+https://github.com/johntruckenbrodt/spatialist.git

