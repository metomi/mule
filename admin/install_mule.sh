#!/bin/bash
# *****************************COPYRIGHT******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file LICENCE.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT******************************
#
# Mule, UM packing extension and UM library installation script
#
# In most cases the modules can be directly installed via the usual 
# python setuptools methods.  However in some cases this might not be 
# possible, so this script instead builds all 3 modules to a dummy
# install location in a temporary directory and then copies the 
# results to the chosen destinations.
#
set -eu

if [ $# -lt 3 ] ; then
    echo "USAGE: "
    echo "   $(basename $0) [--sstpert_lib <sstpert_lib>]"
    echo "                  [--wafccb_lib <wafccb_lib>]"
    echo "                  [--spiral_lib ]"
    echo "                  <lib_dest> <bin_dest> <shumlib> "
    echo ""
    echo "   Must be called from the top-level of a working "
    echo "   copy of the UM mule project, containing the 3"
    echo "   module folders (um_packing, um_utils and mule)"
    echo ""
    echo "   Optionally the um_sstpert, um_wafccb and/or "
    echo "   um_spiral extensions can be included (but they are "
    echo "   not by default because they aren't required for "
    echo "   any core Mule functionality, and the sstpert/wafccb "
    echo "   libraries are only available under a UM license)"
    echo ""
    echo "ARGS: "
    echo "  * <lib_dest>"
    echo "      The destination directory for the 3 "
    echo "      libraries to be installed to."
    echo "  * <bin_dest>"
    echo "      The destination directory for the "
    echo "      UM utility execs to be installed to."
    echo "  * <shumlib>"
    echo "      The location of the UM shared library"
    echo "      for linking the um_packing extension."
    echo "  * --sstpert_lib <sstpert_lib>"
    echo "      (Optional) The location of the UM sstpert"
    echo "      library for linking the um_sstpert extension."
    echo "  * --wafccb_lib <wafccb_lib>"
    echo "      (Optional) The location of the UM wafccb"
    echo "      library for linking the um_wafccb extension."
    echo "  * --spiral_lib"
    echo "      (Optional) Toggles the building of the UM spiral_search"
    echo "      extension (uses the Shumlib location from <shumlib>)."
    echo ""
    echo "  After running the script the directory "
    echo "  named in <lib_dest> should be suitable to "
    echo "  add to python's path, and after doing this "
    echo "  the execs in <bin_dest> should become functional."
    echo ""
    exit 1
fi

# Process the optional arguments
SSTPERT_LIB=
WAFCCB_LIB=
SPIRAL_LIB=
while [ $# -gt 3 ] ; do
    case "$1" in
        --sstpert_lib) shift
                     SSTPERT_LIB=$1 ;;
        --wafccb_lib) shift
                     WAFCCB_LIB=$1 ;;
        --spiral_lib) 
                     SPIRAL_LIB="build" ;;
        *) echo "Unrecognised argument: $1"
           exit 1 ;;
    esac
    shift
done

LIB_DEST=$1
BIN_DEST=$2
SHUMLIB=$3

MODULE_LIST="mule um_packing um_utils"
if [ -n "$SSTPERT_LIB" ] ; then
    MODULE_LIST="$MODULE_LIST um_sstpert"
fi
if [ -n "$WAFCCB_LIB" ] ; then
    MODULE_LIST="$MODULE_LIST um_wafccb"
fi
if [ -n "$SPIRAL_LIB" ] ; then
    MODULE_LIST="$MODULE_LIST um_spiral_search"
fi

# A few hardcoded settings
PYTHONEXEC=${PYTHONEXEC:-python2.7}
SCRATCHDIR=$(mktemp -d)
SCRATCHLIB=$SCRATCHDIR/lib/$PYTHONEXEC/site-packages

# Make relative paths absolute
if [ ! ${LIB_DEST:0:1} == "/" ] ; then
    LIB_DEST=$PWD/$LIB_DEST
fi
if [ ! ${BIN_DEST:0:1} == "/" ] ; then
    BIN_DEST=$PWD/$BIN_DEST
fi

# Create install directores - they may already exist but should be 
# empty if they do, also check the modules exist in the cwd
exit=0
for module in $MODULE_LIST ; do
    mkdir -p $LIB_DEST/$module
    if [ "$(ls -A $LIB_DEST/$module)" ] ; then
        echo "Directory '$LIB_DEST/$module' exists but is non-empty"
        exit=1
    fi
    if [ ! -d ./$module ] ; then
	echo "Directory ./$module not found, is this a working copy?"
	exit=1
    fi
done
if [ $exit -eq 1 ] ; then
    echo "Please ensure install directories are clear and re-start"
    echo "from the top level of a UM mule project working copy"
    exit 1
fi

# Likewise for the directory for binaries
mkdir -p $BIN_DEST
if [ "$(ls $BIN_DEST/mule-* 2> /dev/null)" ] ; then
  echo "Execs already exist in '$BIN_DEST'"
  echo "Please ensure these are removed and re-start"
  exit 1
fi

# Check that shumlib is found
if [ ! -d $SHUMLIB ] ; then
  echo "Shumlib directory '$SHUMLIB' not found"
  exit 1
fi

# If using it, check the sstpert lib is found
if [ -n "$SSTPERT_LIB" ] && [ ! -d $SSTPERT_LIB ] ; then
  echo "SSTpert library directory '$SSTPERT_LIB' not found"
  exit 1
fi

# If using it, check the wafccb lib is found
if [ -n "$WAFCCB_LIB" ] && [ ! -d $WAFCCB_LIB ] ; then
  echo "WAFC CB library directory '$WAFCCB_LIB' not found"
  exit 1
fi

# Make a temporary directory to hold the installs
mkdir -p $SCRATCHLIB 
ln -s $SCRATCHDIR/lib $SCRATCHDIR/lib64

# The install command will complain if this directory isn't on the path
# so add it to the path here
export PYTHONPATH=${PYTHONPATH-""}:$SCRATCHLIB

# Save a reference to the top-level directory
wc_root=$(pwd)

#-------------------------#
# Building the libraries  #
#-------------------------#
# Packing library first
echo "Changing directory to packing module:" $wc_root/um_packing
cd $wc_root/um_packing

echo "Building packing module..."
$PYTHONEXEC setup.py build_ext --inplace \
   -I$SHUMLIB/include -L$SHUMLIB/lib -R$SHUMLIB/lib

# SSTPert library (if being used)
if [ -n "$SSTPERT_LIB" ] ; then
    echo "Changing directory to sstpert module:" $wc_root/um_sstpert
    cd $wc_root/um_sstpert

    echo "Building sstpert module..."
    $PYTHONEXEC setup.py build_ext --inplace \
        -I$SSTPERT_LIB/include \
        -L$SSTPERT_LIB/lib:$SHUMLIB/lib \
        -R$SSTPERT_LIB/lib:$SHUMLIB/lib
fi

# WAFC CB library (if being used)
if [ -n "$WAFCCB_LIB" ] ; then
    echo "Changing directory to wafccb module:" $wc_root/um_wafccb
    cd $wc_root/um_wafccb

    echo "Building wafccb module..."
    $PYTHONEXEC setup.py build_ext --inplace \
        -I$WAFCCB_LIB/include \
        -L$WAFCCB_LIB/lib \
        -R$WAFCCB_LIB/lib
fi

# Spiral search library (if being used)
if [ -n "$SPIRAL_LIB" ] ; then
    echo "Changing directory to spiral search module:" $wc_root/um_spiral_search
    cd $wc_root/um_spiral_search

    echo "Building spiral search module..."
    $PYTHONEXEC setup.py build_ext --inplace \
        -I$SHUMLIB/include \
        -L$SHUMLIB/lib \
        -R$SHUMLIB/lib
fi

#----------------------------------------------#
# Temporary installation to scratch directory  #
#----------------------------------------------#
function install(){
    module=$1
    echo "Changing directory to $module module:" $wc_root/$module
    cd $wc_root/$module

    echo "Installing $module module to $SCRATCHDIR"
    $PYTHONEXEC setup.py install --prefix $SCRATCHDIR
}

for module in $MODULE_LIST ; do
    install $module
done

#------------------------------------------------------------#
# Extraction and copying of files to destination directories #
#------------------------------------------------------------#
function unpack_and_copy(){
    module=$1
    egg=$SCRATCHLIB/$module*.egg

    # The egg might be zipped - if it is unzip it in place
    if [ ! -d $egg ] ; then
      echo "Unpacking zipped egg..."
      unzip_dir=$SCRATCHLIB/${module}_unzipped_egg
      unzip $egg -d $unzip_dir
      egg=$unzip_dir
    fi  

    destdir=$LIB_DEST/$module
    echo "Installing $module to $destdir"
    mkdir -p $destdir
    cp -vr $egg/$module/* $destdir

    # For the execs, also copy these to the bin directory
    if [ $module == "um_utils" ] || [ $module == "um_sstpert" ] ; then
        echo "Installing $module execs and info to $BIN_DEST/"
        cp -vr $egg/EGG-INFO $BIN_DEST/$module.egg-info        
        cp -vr $SCRATCHDIR/bin/* $BIN_DEST/
    fi
}

for module in $MODULE_LIST ; do
    unpack_and_copy $module
done

#------------------------#
# Cleanup install files  #
#------------------------#
function cleanup(){
    module=$1
    echo "Changing directory to $module module:" $wc_root/$module
    cd $wc_root/$module

    echo "Cleaning $module module"
    $PYTHONEXEC setup.py clean
}

for module in $MODULE_LIST ; do
   cleanup $module
done

# Cleanup the temporary directory
echo "Cleaning up temporary directory: $SCRATCHDIR"
rm -rf $SCRATCHDIR
