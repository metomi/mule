# Mule and Mule/UM Utilities

What is Mule?
-------------

Mule is a Python API for accessing the various file types used by the UM;
the UK Met Office's Unified Model.

Along with the Mule modules itself are some utilities; these are written 
using Mule and provide a handful of common types of operations (printing 
file contents, comparing files etc.).  There are also some Python extension
modules which enable the usage of various libraries that provide better
performance and/or identical behaviour to the UM itself.  Some of these
expose libraries from Shumlib (https://github.com/metomi/shumlib) and
others use UM libraries (available under license only) - see the README
files of the individual module folders for more information.

Mule Development
----------------

Note that this Git repository is provided purely as a means to allow public
access to Mule, rather than as a base for development - the actual
development of Mule takes place on the Met Office's Science Repository
Service at https://code.metoffice.gov.uk/ (this is a restricted site with
controlled access for UM partners).  As a result one should not expect to
see pull-requests etc. actioned directly (issues and feedback are still 
welcomed but will be migrated to tickets within the SRS for development).

Mule Licensing
--------------

Although the UM has a restricted commercial licence, Mule is available under
the more permissive BSD 3-Clause licence. The aim of this is to allow maximum 
flexibility and as few barriers to usage as possible. However we would 
still encourage the feedback of modifications and developments (particularly 
bugfixes) rather than modified redistribution.

Mule Documentation
------------------

Documentation for each module is provided within a doc directory and can
be built using Sphinx.
