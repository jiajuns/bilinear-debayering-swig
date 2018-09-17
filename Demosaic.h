// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Python.h>
#include <ndarrayobject.h>

#define NUMPY_IMPORT_ARRAY_RETVAL

class Demosaic {
public:
    Demosaic(float* rawImg, int numrows, int numcols);
    ~Demosaic(void);
    void demosaic();
    PyObject* getRGB();
    PyObject* getNIR();

private:
    cv::Mat image;
    cv::Mat r;
    cv::Mat g;
    cv::Mat b;
    cv::Mat n;
    int rows;
    int cols;
    cv::Mat demosaicImageRGB;
    cv::Mat demosaicImageNir;
};