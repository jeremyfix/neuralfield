#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <iostream>

// Written following the tutorial : http://dan.iel.fm/posts/python-c-extensions/

/* The module doc string */
PyDoc_STRVAR(ctools__doc__,
        "Utilitary functions written in C");


PyDoc_STRVAR(cconv_onestep_utilitary__doc__,
        "Utilitary function to compute the convolution with a step kernel");
void cconv_onestep_utilitary(double* fu, unsigned int N, const double& A, const double& s, double* res) {

    unsigned int lcorner = ceil(N - s);
    unsigned int rcorner = floor(s);
    unsigned int width_step = floor(s);

    // Computes the contribution for the first position
    //
    // res[0] = np.sum(fu[:rcorner+1]) + np.sum(fu[lcorner:])
    double* resptr = res;
    double* fuptr = fu;
    double* fuptr_end = fu + rcorner + 1;
    *resptr = 0;
    while(fuptr != fuptr_end)
        *resptr += *(fuptr++);

    fuptr = fu + lcorner;
    fuptr_end = fuptr + width_step;
    while(fuptr != fuptr_end)
        *resptr += *(fuptr++);

    double * furight = fu + rcorner;
    double * fuleft = fu + lcorner;
    double * fuend = fu + N;
    double * fu_prevend = fu + N - 1;
    double * resend = res + N;
    double * prev_resptr = res;
    resptr = res+1;

    // We iterate until lcorner reaches N (because of the wrap around)
    while(fuleft != fuend) 
        *(resptr++) = *(prev_resptr++) + *(++furight) - *(fuleft++);
    
    fuleft = fu;
    // We iterate until rcorner reaches N 
    while(furight != fu_prevend)
        *(resptr++) = *(prev_resptr++) + *(++furight) - *(fuleft++);

    furight = fu;
    *(resptr++) = *(prev_resptr++) + *(furight) - *(fuleft++);
    // We iterate for the last positions
    while(resptr != resend)
        *(resptr++) = *(prev_resptr++) + *(++furight) - *(fuleft++);

    // res = A * res;
    resptr = res;
    for(unsigned int i = 0 ; i < N; ++i, ++resptr)
        (*resptr) *= A;
}

// The wrapper to the C function
static PyObject *
py_cconv_onestep_utilitary(PyObject *self, PyObject *args) {
    unsigned int N;   
    double A;
    double s;

    PyObject *obj_fu, *obj_res;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OddO", &obj_fu, &A, &s, &obj_res))
                    return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *py_fu = PyArray_FROM_OTF(obj_fu, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *py_res = PyArray_FROM_OTF(obj_res, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (py_fu == NULL || py_res == NULL) {
        Py_XDECREF(py_fu);
        Py_XDECREF(py_res);
        return NULL;
    }

    /* Get the number of data points */
    N = (unsigned int)PyArray_DIM(py_fu, 0);

    /* Get pointers to the data as C-types. */
    double *fu    = (double*)PyArray_DATA(py_fu);
    double *res    = (double*)PyArray_DATA(py_res);

    /* Call the external C function */
    cconv_onestep_utilitary(fu, N, A, s, res);

    /* Clean up. */
    Py_DECREF(py_fu);
    Py_DECREF(py_res);

    Py_RETURN_NONE;

}

static PyMethodDef module_methods[] = {
    {"cconv_onestep_utilitary", py_cconv_onestep_utilitary, METH_VARARGS, cconv_onestep_utilitary__doc__},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initctools(void)
{
    PyObject *m = Py_InitModule3("ctools", module_methods, ctools__doc__);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

