# Neural 3D LTE inversor for Hinode

This repository contains an extremely simple way of using [SICON](https://ui.adsabs.harvard.edu/abs/2019A%26A...626A.102A/abstract), the neural inversion method developed by 
Asensio Ramos & Diaz Baso (2019, A&A, 626, 102A). With only two commands you can
download Hinode data and get the inversion in just a few minutes.

## Download Hinode data

Search for the initial date and time of the Hinode observation you want. It can
be done by using the Hinode database ``http://sdc.uio.no/search/form``. Once 
found, save the date and time of observation and go to the Hinode level 1 database 
that can be found in `http://www.lmsal.com/solarsoft/hinode/level1hao/`. Enter
into the subdirectory structure until you find your observation. For instance, 
let's say you're searching for the observation that started at 19:00 on
2014/02/04. Go to

    http://www.lmsal.com/solarsoft/hinode/level1hao/2014/02/04/SP3D/20140204_190005

and you will find the data as FITS file. Once located, you can call the 
`download_hinode.py` command that will download all the FITS files and generate an HDF5 file 
appropriate for the input to the neural inversor:
    
    python download_hinode.py --url http://www.lmsal.com/solarsoft/hinode/level1hao/2014/02/04/SP3D/20140204_190005 --output output.h5 --downloader curl

By default, the files are downloaded with ``wget``. If you do not have it available in your system, 
which happens in many cases in MacOS, you can use ``curl``. You can find some help of the options 
of the downloader by using ``python download_hinode -h``.

## Run the code

The code needs as input an HDF5 file with the Stokes profiles downloaded from the Hinode database 
and will produce as output another HDF5 file with the resulting physical conditions.
The code is run as:

    python neural_inversion.py --input input.h5 --output output.h5 --normalize 0 100 0 100 --device cpu --resolution 1

The arguments are the input and output files, together with the definition of a box which is used 
as the quiet Sun for normalizing the Stokes profiles. Additionally, you can decide to do
the computation on CPU or GPU. Calculations on GPUs will be faster but the field-of-view
is memory limited. For large maps, we recommend using CPU. Finally, `resolution 1` allows you to output the
results with the same pixel size as the observations, while `resolution 2` upsamples the solution to
double resolution. Use it with care.


## Requirements
    bs4
    numpy
    argparse
    h5py
    tqdm
    astropy
    torch

## Use with Anaconda

We recommend to use Anaconda to run this code. We also recommend to generate a new environment in which all the packages will be installed. Then, install ``PyTorch``as indicated in the webpage ``https://pytorch.org/`` depending on your system . A typical process would be:

    conda create -n inversor python=3.8
    conda activate inversor
    conda install -c conda-forge numpy h5py tqdm astropy bs4 argparse
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
