/* File : interface.i */
%module Demosaic
%{
    #define SWIG_FILE_WITH_INIT
%}

%{
    #include "Demosaic.h"
%}

%include "numpy.i"
%init %{
    import_array();
%}

%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float* rawImg, int numrows, int numcols)};
%include "Demosaic.h"
