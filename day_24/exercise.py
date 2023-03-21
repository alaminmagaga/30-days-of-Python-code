{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5927942a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3f7b6397",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "numpy: 1.20.3\n",
      "['ALLOW_THREADS', 'AxisError', 'BUFSIZE', 'Bytes0', 'CLIP', 'ComplexWarning', 'DataSource', 'Datetime64', 'ERR_CALL', 'ERR_DEFAULT', 'ERR_IGNORE', 'ERR_LOG', 'ERR_PRINT', 'ERR_RAISE', 'ERR_WARN', 'FLOATING_POINT_SUPPORT', 'FPE_DIVIDEBYZERO', 'FPE_INVALID', 'FPE_OVERFLOW', 'FPE_UNDERFLOW', 'False_', 'Inf', 'Infinity', 'MAXDIMS', 'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'MachAr', 'ModuleDeprecationWarning', 'NAN', 'NINF', 'NZERO', 'NaN', 'PINF', 'PZERO', 'RAISE', 'RankWarning', 'SHIFT_DIVIDEBYZERO', 'SHIFT_INVALID', 'SHIFT_OVERFLOW', 'SHIFT_UNDERFLOW', 'ScalarType', 'Str0', 'Tester', 'TooHardError', 'True_', 'UFUNC_BUFSIZE_DEFAULT', 'UFUNC_PYVALS_NAME', 'Uint64', 'VisibleDeprecationWarning', 'WRAP', '_NoValue', '_UFUNC_API', '__NUMPY_SETUP__', '__all__', '__builtins__', '__cached__', '__config__', '__deprecated_attrs__', '__dir__', '__doc__', '__expired_functions__', '__file__', '__getattr__', '__git_revision__', '__loader__', '__mkl_version__', '__name__', '__package__', '__path__', '__spec__', '__version__', '_add_newdoc_ufunc', '_distributor_init', '_financial_names', '_globals', '_mat', '_pytesttester', 'abs', 'absolute', 'add', 'add_docstring', 'add_newdoc', 'add_newdoc_ufunc', 'alen', 'all', 'allclose', 'alltrue', 'amax', 'amin', 'angle', 'any', 'append', 'apply_along_axis', 'apply_over_axes', 'arange', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh', 'argmax', 'argmin', 'argpartition', 'argsort', 'argwhere', 'around', 'array', 'array2string', 'array_equal', 'array_equiv', 'array_repr', 'array_split', 'array_str', 'asanyarray', 'asarray', 'asarray_chkfinite', 'ascontiguousarray', 'asfarray', 'asfortranarray', 'asmatrix', 'asscalar', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'bartlett', 'base_repr', 'binary_repr', 'bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'blackman', 'block', 'bmat', 'bool8', 'bool_', 'broadcast', 'broadcast_arrays', 'broadcast_shapes', 'broadcast_to', 'busday_count', 'busday_offset', 'busdaycalendar', 'byte', 'byte_bounds', 'bytes0', 'bytes_', 'c_', 'can_cast', 'cast', 'cbrt', 'cdouble', 'ceil', 'cfloat', 'char', 'character', 'chararray', 'choose', 'clip', 'clongdouble', 'clongfloat', 'column_stack', 'common_type', 'compare_chararrays', 'compat', 'complex128', 'complex64', 'complex_', 'complexfloating', 'compress', 'concatenate', 'conj', 'conjugate', 'convolve', 'copy', 'copysign', 'copyto', 'core', 'corrcoef', 'correlate', 'cos', 'cosh', 'count_nonzero', 'cov', 'cross', 'csingle', 'ctypeslib', 'cumprod', 'cumproduct', 'cumsum', 'datetime64', 'datetime_as_string', 'datetime_data', 'deg2rad', 'degrees', 'delete', 'deprecate', 'deprecate_with_doc', 'diag', 'diag_indices', 'diag_indices_from', 'diagflat', 'diagonal', 'diff', 'digitize', 'disp', 'divide', 'divmod', 'dot', 'double', 'dsplit', 'dstack', 'dtype', 'e', 'ediff1d', 'einsum', 'einsum_path', 'emath', 'empty', 'empty_like', 'equal', 'errstate', 'euler_gamma', 'exp', 'exp2', 'expand_dims', 'expm1', 'extract', 'eye', 'fabs', 'fastCopyAndTranspose', 'fft', 'fill_diagonal', 'find_common_type', 'finfo', 'fix', 'flatiter', 'flatnonzero', 'flexible', 'flip', 'fliplr', 'flipud', 'float16', 'float32', 'float64', 'float_', 'float_power', 'floating', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod', 'format_float_positional', 'format_float_scientific', 'format_parser', 'frexp', 'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'frompyfunc', 'fromregex', 'fromstring', 'full', 'full_like', 'gcd', 'generic', 'genfromtxt', 'geomspace', 'get_array_wrap', 'get_include', 'get_printoptions', 'getbufsize', 'geterr', 'geterrcall', 'geterrobj', 'gradient', 'greater', 'greater_equal', 'half', 'hamming', 'hanning', 'heaviside', 'histogram', 'histogram2d', 'histogram_bin_edges', 'histogramdd', 'hsplit', 'hstack', 'hypot', 'i0', 'identity', 'iinfo', 'imag', 'in1d', 'index_exp', 'indices', 'inexact', 'inf', 'info', 'infty', 'inner', 'insert', 'int0', 'int16', 'int32', 'int64', 'int8', 'int_', 'intc', 'integer', 'interp', 'intersect1d', 'intp', 'invert', 'is_busday', 'isclose', 'iscomplex', 'iscomplexobj', 'isfinite', 'isfortran', 'isin', 'isinf', 'isnan', 'isnat', 'isneginf', 'isposinf', 'isreal', 'isrealobj', 'isscalar', 'issctype', 'issubclass_', 'issubdtype', 'issubsctype', 'iterable', 'ix_', 'kaiser', 'kron', 'lcm', 'ldexp', 'left_shift', 'less', 'less_equal', 'lexsort', 'lib', 'linalg', 'linspace', 'little_endian', 'load', 'loads', 'loadtxt', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'logspace', 'longcomplex', 'longdouble', 'longfloat', 'longlong', 'lookfor', 'ma', 'mafromtxt', 'mask_indices', 'mat', 'math', 'matmul', 'matrix', 'matrixlib', 'max', 'maximum', 'maximum_sctype', 'may_share_memory', 'mean', 'median', 'memmap', 'meshgrid', 'mgrid', 'min', 'min_scalar_type', 'minimum', 'mintypecode', 'mkl', 'mod', 'modf', 'moveaxis', 'msort', 'multiply', 'nan', 'nan_to_num', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'nbytes', 'ndarray', 'ndenumerate', 'ndfromtxt', 'ndim', 'ndindex', 'nditer', 'negative', 'nested_iters', 'newaxis', 'nextafter', 'nonzero', 'not_equal', 'numarray', 'number', 'obj2sctype', 'object0', 'object_', 'ogrid', 'oldnumeric', 'ones', 'ones_like', 'os', 'outer', 'packbits', 'pad', 'partition', 'percentile', 'pi', 'piecewise', 'place', 'poly', 'poly1d', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint', 'polymul', 'polynomial', 'polysub', 'polyval', 'positive', 'power', 'printoptions', 'prod', 'product', 'promote_types', 'ptp', 'put', 'put_along_axis', 'putmask', 'quantile', 'r_', 'rad2deg', 'radians', 'random', 'ravel', 'ravel_multi_index', 'real', 'real_if_close', 'rec', 'recarray', 'recfromcsv', 'recfromtxt', 'reciprocal', 'record', 'remainder', 'repeat', 'require', 'reshape', 'resize', 'result_type', 'right_shift', 'rint', 'roll', 'rollaxis', 'roots', 'rot90', 'round', 'round_', 'row_stack', 's_', 'safe_eval', 'save', 'savetxt', 'savez', 'savez_compressed', 'sctype2char', 'sctypeDict', 'sctypes', 'searchsorted', 'select', 'set_numeric_ops', 'set_printoptions', 'set_string_function', 'setbufsize', 'setdiff1d', 'seterr', 'seterrcall', 'seterrobj', 'setxor1d', 'shape', 'shares_memory', 'short', 'show_config', 'sign', 'signbit', 'signedinteger', 'sin', 'sinc', 'single', 'singlecomplex', 'sinh', 'size', 'sometrue', 'sort', 'sort_complex', 'source', 'spacing', 'split', 'sqrt', 'square', 'squeeze', 'stack', 'std', 'str0', 'str_', 'string_', 'subtract', 'sum', 'swapaxes', 'sys', 'take', 'take_along_axis', 'tan', 'tanh', 'tensordot', 'test', 'testing', 'tile', 'timedelta64', 'trace', 'tracemalloc_domain', 'transpose', 'trapz', 'tri', 'tril', 'tril_indices', 'tril_indices_from', 'trim_zeros', 'triu', 'triu_indices', 'triu_indices_from', 'true_divide', 'trunc', 'typeDict', 'typecodes', 'typename', 'ubyte', 'ufunc', 'uint', 'uint0', 'uint16', 'uint32', 'uint64', 'uint8', 'uintc', 'uintp', 'ulonglong', 'unicode_', 'union1d', 'unique', 'unpackbits', 'unravel_index', 'unsignedinteger', 'unwrap', 'use_hugepage', 'ushort', 'vander', 'var', 'vdot', 'vectorize', 'version', 'void', 'void0', 'vsplit', 'vstack', 'warnings', 'where', 'who', 'zeros', 'zeros_like']\n"
     ]
    }
   ],
   "source": [
    "print('numpy:', np.__version__)\n",
    "print(dir(np))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3670a330",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Type: <class 'list'>\n",
      "[1, 2, 3, 4, 5]\n",
      "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]\n",
      "<class 'numpy.ndarray'>\n",
      "[1 2 3 4 5]\n"
     ]
    }
   ],
   "source": [
    "    # Creating python List\n",
    "    python_list = [1,2,3,4,5]\n",
    "\n",
    "    # Checking data types\n",
    "    print('Type:', type (python_list)) # <class 'list'>\n",
    "    #\n",
    "    print(python_list) # [1, 2, 3, 4, 5]\n",
    "\n",
    "    two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]\n",
    "\n",
    "    print(two_dimensional_list)  # [[0, 1, 2], [3, 4, 5], [6, 7, 8]]\n",
    "\n",
    "    # Creating Numpy(Numerical Python) array from python list\n",
    "\n",
    "    numpy_array_from_list = np.array(python_list)\n",
    "    print(type (numpy_array_from_list))   # <class 'numpy.ndarray'>\n",
    "    print(numpy_array_from_list) # array([1, 2, 3, 4, 5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f7765d37",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 2. 3. 4. 5.]\n"
     ]
    }
   ],
   "source": [
    "    # Python list\n",
    "    python_list = [1,2,3,4,5]\n",
    "\n",
    "    numy_array_from_list2 = np.array(python_list, dtype=float)\n",
    "    print(numy_array_from_list2) # array([1., 2., 3., 4., 5.])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ea83c2f1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[False  True  True False False]\n"
     ]
    }
   ],
   "source": [
    "    numpy_bool_array = np.array([0, 1, -1, 0, 0], dtype=bool)\n",
    "    print(numpy_bool_array) # array([False,  True,  True, False, False])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ba37b3ff",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n",
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n"
     ]
    }
   ],
   "source": [
    "    two_dimensional_list = [[0,1,2], [3,4,5], [6,7,8]]\n",
    "    numpy_two_dimensional_list = np.array(two_dimensional_list)\n",
    "    print(type (numpy_two_dimensional_list))\n",
    "    print(numpy_two_dimensional_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a35424f2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'list'>\n",
      "one dimensional array: [1, 2, 3, 4, 5]\n",
      "two dimensional array:  [[0, 1, 2], [3, 4, 5], [6, 7, 8]]\n"
     ]
    }
   ],
   "source": [
    "# We can always convert an array back to a python list using tolist().\n",
    "np_to_list = numpy_array_from_list.tolist()\n",
    "print(type (np_to_list))\n",
    "print('one dimensional array:', np_to_list)\n",
    "print('two dimensional array: ', numpy_two_dimensional_list.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "0c9ed492",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'tuple'>\n",
      "python_tuple:  (1, 2, 3, 4, 5)\n",
      "<class 'numpy.ndarray'>\n",
      "numpy_array_from_tuple:  [1 2 3 4 5]\n"
     ]
    }
   ],
   "source": [
    "# Numpy array from tuple\n",
    "# Creating tuple in Python\n",
    "python_tuple = (1,2,3,4,5)\n",
    "print(type (python_tuple)) # <class 'tuple'>\n",
    "print('python_tuple: ', python_tuple) # python_tuple:  (1, 2, 3, 4, 5)\n",
    "\n",
    "numpy_array_from_tuple = np.array(python_tuple)\n",
    "print(type (numpy_array_from_tuple)) # <class 'numpy.ndarray'>\n",
    "print('numpy_array_from_tuple: ', numpy_array_from_tuple) # numpy_array_from_tuple:  [1 2 3 4 5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "6f80eb56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5]\n",
      "shape of nums:  (5,)\n",
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n",
      "shape of numpy_two_dimensional_list:  (3, 3)\n",
      "(3, 4)\n"
     ]
    }
   ],
   "source": [
    "    nums = np.array([1, 2, 3, 4, 5])\n",
    "    print(nums)\n",
    "    print('shape of nums: ', nums.shape)\n",
    "    print(numpy_two_dimensional_list)\n",
    "    print('shape of numpy_two_dimensional_list: ', numpy_two_dimensional_list.shape)\n",
    "    three_by_four_array = np.array([[0, 1, 2, 3],\n",
    "        [4,5,6,7],\n",
    "        [8,9,10, 11]])\n",
    "    print(three_by_four_array.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "81e57273",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-3 -2 -1  0  1  2  3]\n",
      "int32\n",
      "[-3. -2. -1.  0.  1.  2.  3.]\n",
      "float64\n"
     ]
    }
   ],
   "source": [
    "int_lists = [-3, -2, -1, 0, 1, 2,3]\n",
    "int_array = np.array(int_lists)\n",
    "float_array = np.array(int_lists, dtype=float)\n",
    "\n",
    "print(int_array)\n",
    "print(int_array.dtype)\n",
    "print(float_array)\n",
    "print(float_array.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e30333e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The size: 5\n",
      "The size: 9\n"
     ]
    }
   ],
   "source": [
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "two_dimensional_list = np.array([[0, 1, 2],\n",
    "                              [3, 4, 5],\n",
    "                              [6, 7, 8]])\n",
    "\n",
    "print('The size:', numpy_array_from_list.size) # 5\n",
    "print('The size:', two_dimensional_list.size)  # 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1c6cc29a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[11 12 13 14 15]\n"
     ]
    }
   ],
   "source": [
    "# Mathematical Operation\n",
    "# Addition\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_plus_original = numpy_array_from_list  + 10\n",
    "print(ten_plus_original)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "b472017e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[-9 -8 -7 -6 -5]\n"
     ]
    }
   ],
   "source": [
    "# Subtraction\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_minus_original = numpy_array_from_list  - 10\n",
    "print(ten_minus_original)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "4563b03b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[10 20 30 40 50]\n"
     ]
    }
   ],
   "source": [
    "# Multiplication\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_times_original = numpy_array_from_list * 10\n",
    "print(ten_times_original)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "353a18a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[0.1 0.2 0.3 0.4 0.5]\n"
     ]
    }
   ],
   "source": [
    "# Division\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_times_original = numpy_array_from_list / 10\n",
    "print(ten_times_original)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d5f7e902",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[1 2 0 1 2]\n"
     ]
    }
   ],
   "source": [
    "# Modulus; Finding the remainder\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_times_original = numpy_array_from_list % 3\n",
    "print(ten_times_original)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "d0a98fc6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[0 0 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "# Floor division: the division result without the remainder\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_times_original = numpy_array_from_list // 10\n",
    "print(ten_times_original)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "70f0e8b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "original array:  [1 2 3 4 5]\n",
      "[ 1  4  9 16 25]\n"
     ]
    }
   ],
   "source": [
    "# Exponential is finding some number the power of another:\n",
    "numpy_array_from_list = np.array([1, 2, 3, 4, 5])\n",
    "print('original array: ', numpy_array_from_list)\n",
    "ten_times_original = numpy_array_from_list  ** 2\n",
    "print(ten_times_original)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "28badf83",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "int32\n",
      "float64\n",
      "bool\n"
     ]
    }
   ],
   "source": [
    "#Int,  Float numbers\n",
    "numpy_int_arr = np.array([1,2,3,4])\n",
    "numpy_float_arr = np.array([1.1, 2.0,3.2])\n",
    "numpy_bool_arr = np.array([-3, -2, 0, 1,2,3], dtype='bool')\n",
    "\n",
    "print(numpy_int_arr.dtype)\n",
    "print(numpy_float_arr.dtype)\n",
    "print(numpy_bool_arr.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d5694674",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4.])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numpy_int_arr = np.array([1,2,3,4], dtype = 'float')\n",
    "numpy_int_arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "76841abd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4])"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numpy_int_arr = np.array([1., 2., 3., 4.], dtype = 'int')\n",
    "numpy_int_arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "7c740773",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True, False,  True,  True,  True])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([-3, -2, 0, 1,2,3], dtype='bool')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "9862df0f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[9, 8, 7],\n",
       "       [6, 5, 4],\n",
       "       [3, 2, 1]])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    two_dimension_array = np.array([[1,2,3],[4,5,6], [7,8,9]])\n",
    "    two_dimension_array[::-1,::-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "3cf24d34",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 3]\n",
      " [4 5 6]\n",
      " [7 8 9]]\n",
      "[[ 1  2  3]\n",
      " [ 4 55 44]\n",
      " [ 7  8  9]]\n"
     ]
    }
   ],
   "source": [
    "    print(two_dimension_array)\n",
    "    two_dimension_array[1,1] = 55\n",
    "    two_dimension_array[1,2] =44\n",
    "    print(two_dimension_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "905943e9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0],\n",
       "       [0, 0, 0],\n",
       "       [0, 0, 0]])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    # Numpy Zeroes\n",
    "    # numpy.zeros(shape, dtype=float, order='C')\n",
    "    numpy_zeroes = np.zeros((3,3),dtype=int,order='C')\n",
    "    numpy_zeroes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "d9a2aef4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 1 1]\n",
      " [1 1 1]\n",
      " [1 1 1]]\n"
     ]
    }
   ],
   "source": [
    "# Numpy Zeroes\n",
    "numpy_ones = np.ones((3,3),dtype=int,order='C')\n",
    "print(numpy_ones)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "42e60f15",
   "metadata": {},
   "outputs": [],
   "source": [
    "twoes = numpy_ones * 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "3daf4d28",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 3]\n",
      " [4 5 6]]\n",
      "[[1 2]\n",
      " [3 4]\n",
      " [5 6]]\n"
     ]
    }
   ],
   "source": [
    "# Reshape\n",
    "# numpy.reshape(), numpy.flatten()\n",
    "first_shape  = np.array([(1,2,3), (4,5,6)])\n",
    "print(first_shape)\n",
    "reshaped = first_shape.reshape(3,2)\n",
    "print(reshaped)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "197878e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "flattened = reshaped.flatten()\n",
    "flattened"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "aa320bf6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5 7 9]\n",
      "Horizontal Append: [1 2 3 4 5 6]\n"
     ]
    }
   ],
   "source": [
    "    ## Horitzontal Stack\n",
    "    np_list_one = np.array([1,2,3])\n",
    "    np_list_two = np.array([4,5,6])\n",
    "\n",
    "    print(np_list_one + np_list_two)\n",
    "\n",
    "    print('Horizontal Append:', np.hstack((np_list_one, np_list_two)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "a334df6a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vertical Append: [[1 2 3]\n",
      " [4 5 6]]\n"
     ]
    }
   ],
   "source": [
    "    ## Vertical Stack\n",
    "    print('Vertical Append:', np.vstack((np_list_one, np_list_two)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "148f1ed1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5324512081318932"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    # Generate a random float  number\n",
    "    random_float = np.random.random()\n",
    "    random_float"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "5192330c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    # Generating a random integers between 0 and 10\n",
    "\n",
    "    random_int = np.random.randint(0, 11)\n",
    "    random_int"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "213cdedc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 8, 9, 2])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    # Generating a random integers between 2 and 11, and creating a one row array\n",
    "    random_int = np.random.randint(2,10, size=4)\n",
    "    random_int"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "1cdad031",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[7, 5, 9],\n",
       "       [3, 9, 6],\n",
       "       [5, 8, 8]])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    # Generating a random integers between 0 and 10\n",
    "    random_int = np.random.randint(2,10, size=(3,3))\n",
    "    random_int"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "7df68fd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 94.79506115,  92.08950213,  77.19557112,  97.30457521,\n",
       "        71.82362506,  93.69890915,  83.97383486,  82.33731663,\n",
       "        90.38066788,  89.51435372,  79.51150549,  80.74455896,\n",
       "        77.02957563,  84.09601264,  56.84561402,  75.37704033,\n",
       "        64.46448478,  66.57161359,  73.86316177,  91.01189769,\n",
       "        66.84066853,  81.49385979,  70.12991776,  61.32332072,\n",
       "        71.83832808,  71.35943522,  75.12860588,  88.26917591,\n",
       "        83.77177362,  57.38023427,  88.65824547,  64.82209245,\n",
       "        99.65301272, 102.46892873, 131.55585312,  81.79019421,\n",
       "        88.83699517,  78.22680283,  88.45288531,  83.82993834,\n",
       "        61.62221916,  75.409939  ,  79.79011049,  76.27048458,\n",
       "        44.85364577,  83.15601308,  71.49328368,  89.57921924,\n",
       "        82.44731921,  54.40278581,  83.04863068,  92.56151784,\n",
       "        90.6882856 ,  71.06714281,  91.17778008,  85.62971346,\n",
       "        73.17804115,  80.58108456,  59.07340871,  52.42442746,\n",
       "        57.44307697,  67.78544903,  74.77581511,  84.3913055 ,\n",
       "        78.11984546,  63.34117016,  30.93952797,  91.44603862,\n",
       "        72.29935042,  74.47285253,  83.62492842, 108.03209603,\n",
       "        77.80972097,  57.64014301,  70.40990522,  75.32609972,\n",
       "        85.06209975,  91.56139913,  66.86348199,  37.34071002])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "    # np.random.normal(mu, sigma, size)\n",
    "    normal_array = np.random.normal(79, 15, 80)\n",
    "    normal_array\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "24825f89",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 4., 0., 2., 3.,\n",
       "        3., 1., 3., 6., 4., 6., 3., 4., 6., 7., 1., 4., 6., 4., 2., 1., 0.,\n",
       "        1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),\n",
       " array([ 30.93952797,  32.95185448,  34.96418098,  36.97650748,\n",
       "         38.98883398,  41.00116049,  43.01348699,  45.02581349,\n",
       "         47.03814   ,  49.0504665 ,  51.062793  ,  53.07511951,\n",
       "         55.08744601,  57.09977251,  59.11209901,  61.12442552,\n",
       "         63.13675202,  65.14907852,  67.16140503,  69.17373153,\n",
       "         71.18605803,  73.19838453,  75.21071104,  77.22303754,\n",
       "         79.23536404,  81.24769055,  83.26001705,  85.27234355,\n",
       "         87.28467006,  89.29699656,  91.30932306,  93.32164956,\n",
       "         95.33397607,  97.34630257,  99.35862907, 101.37095558,\n",
       "        103.38328208, 105.39560858, 107.40793508, 109.42026159,\n",
       "        111.43258809, 113.44491459, 115.4572411 , 117.4695676 ,\n",
       "        119.4818941 , 121.49422061, 123.50654711, 125.51887361,\n",
       "        127.53120011, 129.54352662, 131.55585312]),\n",
       " <BarContainer object of 50 artists>)"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set()\n",
    "plt.hist(normal_array, color=\"grey\", bins=50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "73b7d88f",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "four_by_four_matrix = np.matrix(np.ones((4,4), dtype=float))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "9c982730",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "matrix([[1., 1., 1., 1.],\n",
       "        [1., 1., 1., 1.],\n",
       "        [2., 2., 2., 2.],\n",
       "        [1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.asarray(four_by_four_matrix)[2] = 2\n",
    "four_by_four_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "47be58d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "range(0, 11, 2)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# creating list using range(starting, stop, step)\n",
    "lst = range(0, 11, 2)\n",
    "lst"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "2bdcd973",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "range(0, 11, 2)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "range(0, 11, 2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "7930bc68",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "2\n",
      "4\n",
      "6\n",
      "8\n",
      "10\n"
     ]
    }
   ],
   "source": [
    "for l in lst:\n",
    "    print(l)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "a1795b5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n",
       "       17, 18, 19])"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Similar to range arange numpy.arange(start, stop, step)\n",
    "whole_numbers = np.arange(0, 20, 1)\n",
    "whole_numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "4748072e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\n",
       "       18, 19])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "natural_numbers = np.arange(1, 20, 1)\n",
    "natural_numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "36cab554",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "odd_numbers = np.arange(1, 20, 2)\n",
    "odd_numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "05678f49",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.        , 1.44444444, 1.88888889, 2.33333333, 2.77777778,\n",
       "       3.22222222, 3.66666667, 4.11111111, 4.55555556, 5.        ])"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# numpy.linspace()\n",
    "# numpy.logspace() in Python with Example\n",
    "# For instance, it can be used to create 10 values from 1 to 5 evenly spaced.\n",
    "np.linspace(1.0, 5.0, num=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "87a44772",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1. , 1.8, 2.6, 3.4, 4.2])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# not to include the last value in the interval\n",
    "np.linspace(1.0, 5.0, num=5, endpoint=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "0b17ffee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  100.        ,   464.15888336,  2154.43469003, 10000.        ])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# LogSpace\n",
    "# LogSpace returns even spaced numbers on a log scale. Logspace has the same parameters as np.linspace.\n",
    "\n",
    "# Syntax:\n",
    "\n",
    "# numpy.logspace(start, stop, num, endpoint)\n",
    "\n",
    "np.logspace(2, 4.0, num=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "18a73ae5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# to check the size of an array\n",
    "x = np.array([1,2,3], dtype=np.complex128)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "68550865",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "16"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.itemsize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "2436865f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6]])"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# indexing and Slicing NumPy Arrays in Python\n",
    "np_list = np.array([(1,2,3), (4,5,6)])\n",
    "np_list\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "d09496ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "First row:  [1 2 3]\n",
      "Second row:  [4 5 6]\n"
     ]
    }
   ],
   "source": [
    "print('First row: ', np_list[0])\n",
    "print('Second row: ', np_list[1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "7474270f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "First column:  [1 4]\n",
      "Second column:  [2 5]\n",
      "Third column:  [3 6]\n"
     ]
    }
   ],
   "source": [
    "print('First column: ', np_list[:,0])\n",
    "print('Second column: ', np_list[:,1])\n",
    "print('Third column: ', np_list[:,2])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "0b0b9b08",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "min:  1\n",
      "max:  55\n",
      "mean:  14.777777777777779\n",
      "sd:  18.913709183069525\n"
     ]
    }
   ],
   "source": [
    "np_normal_dis = np.random.normal(5, 0.5, 100)\n",
    "np_normal_dis\n",
    "## min, max, mean, median, sd\n",
    "print('min: ', two_dimension_array.min())\n",
    "print('max: ', two_dimension_array.max())\n",
    "print('mean: ',two_dimension_array.mean())\n",
    "# print('median: ', two_dimension_array.median())\n",
    "print('sd: ', two_dimension_array.std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "2e9f2264",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3]\n",
      " [ 4 55 44]\n",
      " [ 7  8  9]]\n",
      "Column with minimum:  [1 2 3]\n",
      "Column with maximum:  [ 7 55 44]\n",
      "=== Row ==\n",
      "Row with minimum:  [1 4 7]\n",
      "Row with maximum:  [ 3 55  9]\n"
     ]
    }
   ],
   "source": [
    "print(two_dimension_array)\n",
    "print('Column with minimum: ', np.amin(two_dimension_array,axis=0))\n",
    "print('Column with maximum: ', np.amax(two_dimension_array,axis=0))\n",
    "print('=== Row ==')\n",
    "print('Row with minimum: ', np.amin(two_dimension_array,axis=1))\n",
    "print('Row with maximum: ', np.amax(two_dimension_array,axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "e490c9ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tile:    [1 2 3 1 2 3]\n",
      "Repeat:  [1 1 2 2 3 3]\n"
     ]
    }
   ],
   "source": [
    "a = [1,2,3]\n",
    "\n",
    "# Repeat whole of 'a' two times\n",
    "print('Tile:   ', np.tile(a, 2))\n",
    "\n",
    "# Repeat each element of 'a' two times\n",
    "print('Repeat: ', np.repeat(a, 2))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "ddbe2888",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7572743662553332\n"
     ]
    }
   ],
   "source": [
    "# One random number between [0,1)\n",
    "one_random_num = np.random.random()\n",
    "one_random_in = np.random\n",
    "print(one_random_num)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "9804d22a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.31500321 0.13735474 0.76419639]\n",
      " [0.26608814 0.5972242  0.68909323]]\n"
     ]
    }
   ],
   "source": [
    "# Random numbers between [0,1) of shape 2,3\n",
    "r = np.random.random(size=[2,3])\n",
    "print(r)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "3133147b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['e' 'i' 'e' 'i' 'e' 'o' 'a' 'i' 'o' 'o']\n"
     ]
    }
   ],
   "source": [
    "print(np.random.choice(['a', 'e', 'i', 'o', 'u'], size=10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "4ad0fbe2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.19673135, 0.37452078],\n",
       "       [0.79479517, 0.53304351]])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Random numbers between [0, 1] of shape 2, 2\n",
    "rand = np.random.rand(2,2)\n",
    "rand"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "5b0abcd7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 7, 2],\n",
       "       [6, 1, 5],\n",
       "       [5, 6, 3],\n",
       "       [3, 7, 3],\n",
       "       [2, 9, 3]])"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Random integers between [0, 10) of shape 2,5\n",
    "rand_int = np.random.randint(0, 10, size=[5,3])\n",
    "rand_int"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "b1de9ae6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "min:  3.4292852234918225\n",
      "max:  6.439368908821411\n",
      "mean:  4.983619643171388\n",
      "median:  4.972374944720592\n",
      "mode:  ModeResult(mode=array([3.42928522]), count=array([1]))\n",
      "sd:  0.5060874752005449\n"
     ]
    }
   ],
   "source": [
    "from scipy import stats\n",
    "np_normal_dis = np.random.normal(5, 0.5, 1000) # mean, standard deviation, number of samples\n",
    "np_normal_dis\n",
    "## min, max, mean, median, sd\n",
    "print('min: ', np.min(np_normal_dis))\n",
    "print('max: ', np.max(np_normal_dis))\n",
    "print('mean: ', np.mean(np_normal_dis))\n",
    "print('median: ', np.median(np_normal_dis))\n",
    "print('mode: ', stats.mode(np_normal_dis))\n",
    "print('sd: ', np.std(np_normal_dis))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "c95684e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXkAAAD8CAYAAACSCdTiAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAVdklEQVR4nO3df0xV9/3H8dfFy0UtlE16b2kcIVnXpYvZcNkPd6eB2XSIxTsTJIt2qcucaV0WdC6zs0gkmNmhM7kZcTTL0ti0s2nxB4MRhmvGRmohbWRLCatbFgesWgNXbFVauMC95/tHv7srtXK55x683E+fj7+895x7zvvN594Xx8O5n+OyLMsSAMBIGakuAAAwfwh5ADAYIQ8ABiPkAcBghDwAGIyQBwCDzSnkx8bGtGHDBl28eFGS9OKLL2rDhg0KBAJ64oknNDk5KUk6f/68KioqtG7dOu3bt0/T09PzVzkAIK64If/6669ry5YtGhwclCQNDAzo6aef1gsvvKDW1lZFo1E9//zzkqQ9e/Zo//79OnPmjCzLUlNT07wWDwCYnTveCk1NTaqtrdXjjz8uSfJ4PKqtrVV2drYk6bOf/azeeustXbp0SRMTE1q5cqUkqaKiQg0NDXr44YcTKujtt99VNOrs97Py8rI1Ojrm6DZThV4WHlP6kMzpxZQ+pPi9ZGS49MlP3nHL5XFD/uDBgzMeL1++XMuXL5ckXb16VcePH9fPf/5zjYyMyOv1xtbzer0aHh6O28CHRaOW4yH/3+2agl4WHlP6kMzpxZQ+pOR6iRvytzI8PKzt27dr06ZNWrVqlXp7e+VyuWLLLcua8Xiu8vKy7ZY0K683Z162mwr0svCY0odkTi+m9CEl14utkL9w4YK2b9+uRx55RNu2bZMk5efnKxQKxda5cuWKfD5fwtseHR1z/Dew15ujUOiGo9tMFXpZeEzpQzKnF1P6kOL3kpHhmvXgOOFLKMfGxvT9739fu3btigW89P5pnKysLPX29kqSWlpaVFxcnOjmAQAOSvhI/uTJk7py5YqOHTumY8eOSZIeeOAB7dq1S0eOHFFNTY3Gxsa0YsUKbd261fGCAQBz51poUw1zumZ29LLwmNKHZE4vpvQhpeB0DQAgfRDyAGAw25dQAqbJzc2Sx+Ox/dpr18IOVwQkj5AH/p/H41FdXZ2t19bW1koi5LHwEPIwTjJH5IBpCHkYx+4R+ftH44BZ+MMrABiMkAcAgxHyAGAwQh4ADEbIA4DBuLoGSKFkLvecnJzkC1iIi5AHUogvYGG+cboGAAxGyAOAwQh5ADAYIQ8ABiPkAcBghDwAGIyQBwCDEfIAYDBCHgAMRsgDgMEIeQAwGCEPAAYj5AHAYIQ8ABhsTiE/NjamDRs26OLFi5Kk7u5uBQIBlZaWKhgMxtY7f/68KioqtG7dOu3bt0/T09PzUzUAYE7ihvzrr7+uLVu2aHBwUJI0MTGh6upqNTY2qr29Xf39/erq6pIk7dmzR/v379eZM2dkWZaamprmtXgAwOzihnxTU5Nqa2vl8/kkSX19fSosLFRBQYHcbrcCgYA6Ojp06dIlTUxMaOXKlZKkiooKdXR0zGvxAIDZxb0z1MGDB2c8HhkZkdfrjT32+XwaHh6+6Xmv16vh4WEHSwUAJCrh2/9Fo1G5XK7YY8uy5HK5bvl8ovLyshN+zVx4vTnzst1UoJeFKRW9zMc+TRkTU/qQkusl4ZDPz89XKBSKPQ6FQvL5fDc9f+XKldgpnkSMjo4pGrUSft1svN4chUI3HN1mqtDL3LabCnZ6SbZWp39+pry/TOlDit9LRoZr1oPjhC+hLCoq0sDAgIaGhhSJRNTW1qbi4mItX75cWVlZ6u3tlSS1tLSouLg40c0DAByU8JF8VlaW6uvrVVVVpXA4rJKSEpWVlUmSjhw5opqaGo2NjWnFihXaunWr4wUDAOZuziHf2dkZ+7ff71dra+tN69x///06efKkM5UBAJLGN14BwGCEPAAYjJAHAIMR8gBgsISvrgFws+npaaO+fANzEPKAA9xut+rq6hJ+XW1t7TxUA/wPp2sAwGCEPAAYjJAHAIMR8gBgMEIeAAxGyAOAwQh5ADAYIQ8ABiPkAcBghDwAGIyQBwCDEfIAYDBCHgAMRsgDgMEIeQAwGCEPAAYj5AHAYNwZCgtSbm6WPB5PqssA0h4hjwXJ4/HYup2exC31gA/idA0AGIyQBwCDJRXyLS0tKi8vV3l5uQ4dOiRJ6u7uViAQUGlpqYLBoCNFAgDssR3y4+PjOnjwoJ577jm1tLTo3Llz6uzsVHV1tRobG9Xe3q7+/n51dXU5WS8AIAG2Qz4SiSgajWp8fFzT09Oanp5Wdna2CgsLVVBQILfbrUAgoI6ODifrBQAkwPbVNdnZ2dq1a5fWr1+vJUuW6Ctf+YpGRkbk9Xpj6/h8Pg0PDye03by8bLslzcrrzZmX7aYCveC/5uPnZ8qYmNKHlFwvtkP+H//4h06dOqU///nPysnJ0U9+8hMNDg7K5XLF1rEsa8bjuRgdHVM0atkt6yN5vTkKhW44us1U+bj0YtIHdD45/V4w5f1lSh9S/F4yMlyzHhzbPl1z9uxZ+f1+5eXlyePxqKKiQq+++qpCoVBsnVAoJJ/PZ3cXAIAk2Q75+++/X93d3XrvvfdkWZY6OztVVFSkgYEBDQ0NKRKJqK2tTcXFxU7WCwBIgO3TNWvWrNEbb7yhiooKZWZm6vOf/7yqqqq0evVqVVVVKRwOq6SkRGVlZU7WCwBIQFLTGjz66KN69NFHZzzn9/vV2tqaVFEAAGfwjVcAMBghDwAGI+QBwGBMNQykqenpaVvfJ5iamlJmZuYtl8+2zcnJSV27Fk54n0gdQh5IU26329ac+7W1tUnO1U/IpxNO1wCAwQh5ADAYIQ8ABiPkAcBghDwAGIyQBwCDEfIAYDBCHgAMRsgDgMEIeQAwGCEPAAYj5AHAYIQ8ABiMkAcAgxHyAGAwQh4ADMZNQzCvcnOz5PF4brnczp2NAMwdIY955fF4bN+9CEDyOF0DAAYj5AHAYIQ8ABgsqZDv7OxURUWF1q9fr5/97GeSpO7ubgUCAZWWlioYDDpSJADAHtsh/+abb6q2tlaNjY1qbW3VG2+8oa6uLlVXV6uxsVHt7e3q7+9XV1eXk/UCABJgO+RfeuklPfTQQ8rPz1dmZqaCwaCWLFmiwsJCFRQUyO12KxAIqKOjw8l6AQAJsH0J5dDQkDIzM7Vjxw5dvnxZ3/jGN3TffffJ6/XG1vH5fBoeHnakUABA4myHfCQS0blz5/Tcc89p6dKl+sEPfqDFixfL5XLF1rEsa8bjucjLy7Zb0qxM+tKNSb0g/aTL+y9d6pyLZHqxHfJ33XWX/H6/li1bJkl68MEH1dHRoUWLFsXWCYVC8vl8CW13dHRM0ahlt6yP5PXmKBS64eg2UyXdejHpg4b3pcP7L90+J7OJ10tGhmvWg2Pb5+TXrl2rs2fP6vr164pEInr55ZdVVlamgYEBDQ0NKRKJqK2tTcXFxXZ3AQBIku0j+aKiIm3fvl0PP/ywpqamtHr1am3ZskWf/vSnVVVVpXA4rJKSEpWVlTlZLwAgAUnNXVNZWanKysoZz/n9frW2tiZVFADAGXzjFQAMRsgDgMEIeQAwGCEPAAYj5AHAYIQ8ABiMkAcAgxHyAGAwbuSNuHJzs+TxeFJdBgAbCHnE5fF4VFdXZ+u1tbW1DlcDIBGcrgEAgxHyAGAwQh4ADMY5eQBzNj09betGMJOTk7p2LTwPFSEeQh7AnLndblt/hH//D/CEfCpwugYADEbIA4DBCHkAMBghDwAGI+QBwGCEPAAYjJAHAIMR8gBgMEIeAAxGyAOAwQh5ADAYIQ8ABnMk5A8dOqS9e/dKkrq7uxUIBFRaWqpgMOjE5gEANiUd8j09PWpubpYkTUxMqLq6Wo2NjWpvb1d/f7+6urqSLhIAYE9SIf/OO+8oGAxqx44dkqS+vj4VFhaqoKBAbrdbgUBAHR0djhQKAEhcUiG/f/9+7d69W3feeackaWRkRF6vN7bc5/NpeHg4uQoBALbZvmnIiRMndM8998jv9+v06dOSpGg0KpfLFVvHsqwZj+ciLy/bbkmzsnM3m4XKpF7w8XG737cmfU6S6cV2yLe3tysUCmnjxo26du2a3nvvPV26dEmLFi2KrRMKheTz+RLa7ujomKJRy25ZH8nrzVEodMPRbaZKKnox6cOC1Lmd79uP02c+I8M168Gx7ZA/duxY7N+nT5/Wa6+9prq6OpWWlmpoaEif+tSn1NbWpk2bNtndBQAgSY7e4zUrK0v19fWqqqpSOBxWSUmJysrKnNwFACABjoR8RUWFKioqJEl+v1+tra1ObBYAkCS+8QoABiPkAcBghDwAGIyQBwCDEfIAYDBCHgAM5uh18ljYcnOz5PF4Ul0GgNuIkP8Y8Xg8qqurS/h1tbW181ANgNuB0zUAYDBCHgAMRsgDgMEIeQAwGCEPAAYj5AHAYIQ8ABiM6+TTzPT0NLfjQ9pJ5n07OTmpa9fCDlf08UHIpxm3223rC00SX2pC6iT/viXk7eJ0DQAYjJAHAIMR8gBgMEIeAAxGyAOAwQh5ADAYl1ACWNDsXmOfm5vF9fUi5AEscHavsef6+vdxugYADEbIA4DBkgr5o0ePqry8XOXl5Tp8+LAkqbu7W4FAQKWlpQoGg44UCQCwx3bId3d36+zZs2pubtbvfvc7/f3vf1dbW5uqq6vV2Nio9vZ29ff3q6ury8l6AQAJsB3yXq9Xe/fulcfjUWZmpu69914NDg6qsLBQBQUFcrvdCgQC6ujocLJeAEACbIf8fffdp5UrV0qSBgcH9Yc//EEul0terze2js/n0/DwcNJFAgDsSfoSyn/961967LHH9Pjjj2vRokUaHByMLbMsSy6XK6Ht5eVlJ1vSR2IOduDjx5TPfTJ9JBXyvb292rlzp6qrq1VeXq7XXntNoVAotjwUCsnn8yW0zdHRMUWjVjJl3cTrzVEodMPRbaaKKW9a4HYw4XMfL78yMlyzHhzbPl1z+fJl/fCHP9SRI0dUXl4uSSoqKtLAwICGhoYUiUTU1tam4uJiu7sAACTJ9pH8008/rXA4rPr6+thzmzdvVn19vaqqqhQOh1VSUqKysjJHCjVNbm6WPB5PqssAYDjbIV9TU6OampqPXNba2mq7oI8Lj8eTxFe1AWBu+MYrABiMkAcAgxHyAGAwQh4ADEbIA4DBCHkAMBghDwAGI+QBwGCEPAAYjJAHAIMR8gBgsKTnkweAhWh6etr21NyTk5O6di3scEWpQcgDMJLb7bY1CaD034kAzQh5TtcAgMEIeQAwGCEPAAYj5AHAYIQ8ABiMkAcAgxHyAGAwQh4ADEbIA4DBCHkAMBjTGgDAh9id92YhznlDyAPAh9id92YhznljTMjn5mbJ4/Hccvlsv5WnpqaUmZmZ8D7tvg6AmRbizJfzEvK///3v9dRTT2l6elrf/e539Z3vfGc+djODx+NJasY5u7+1k5vlDoBJFuLMl46H/PDwsILBoE6fPi2Px6PNmzdr1apV+sxnPuP0rgAAcTh+dU13d7e+9rWv6ROf+ISWLl2qdevWqaOjw+ndAADmwPEj+ZGREXm93thjn8+nvr6+Ob8+I8Nle9+5ubm3/bXsk32yT/bp1D5vlX+z5WK8zHRZlmXZrugjPPXUUwqHw/rRj34kSWpqalJ/f78OHDjg5G4AAHPg+Oma/Px8hUKh2ONQKCSfz+f0bgAAc+B4yH/9619XT0+Prl69qvHxcf3xj39UcXGx07sBAMyB4+fk7777bu3evVtbt27V1NSUKisr9YUvfMHp3QAA5sDxc/IAgIWDCcoAwGCEPAAYjJAHAIMR8gBgMEIeAAxmzFTDkvTLX/5SZ86ckcvlUmVlpb73ve/NWH706FGdOnVKd955pyTp29/+9m2ZITMZhw4d0ttvv636+voZz58/f1779u3Tu+++qy9/+cuqq6uT271wh/NWfaTTmDzyyCO6evVq7Od84MABFRUVxZany5jE6yOdxqSzs1NHjx7V+Pi4Vq9erZqamhnL02VM4vWR1JhYhnj11VetzZs3W1NTU9b4+Li1du1a68KFCzPWeeyxx6y//vWvKaowcd3d3daqVausn/70pzctKy8vt/72t79ZlmVZTzzxhHX8+PHbXN3czdZHuoxJNBq11qxZY01NTd1ynXQYk7n0kS5j8p///Mdas2aNdfnyZWtyctLasmWL9Ze//GXGOukwJnPpI5kxMeZ0zVe/+lU9++yzcrvdGh0dVSQS0dKlS2es09/fr1//+tcKBAI6cOCAwuGFdQeXD3rnnXcUDAa1Y8eOm5ZdunRJExMTWrlypSSpoqJiwc70OVsfUvqMyb///W9J0rZt2/Stb31Lv/3tb2csT5cxideHlD5j8tJLL+mhhx5Sfn6+MjMzFQwGZ/yPJF3GJF4fUnJjYkzIS1JmZqYaGhpUXl4uv9+vu+++O7bs3Xff1ec+9znt2bNHzc3Nun79uhobG1NY7ez279+v3bt3x/579kEfnunT6/VqeHj4dpY3Z7P1kU5jcv36dfn9fv3qV7/SM888oxdeeEGvvPJKbHm6jEm8PtJpTIaGhhSJRLRjxw5t3LhRzz///IwZINNlTOL1keyYGBXykrRz50719PTo8uXLampqij1/xx136De/+Y3uvfdeud1ubdu2TV1dXSms9NZOnDihe+65R36//yOXR6NRuVz/m17UsqwZjxeKeH2k05h88Ytf1OHDh5WTk6Nly5apsrJyRq3pMibx+kinMYlEIurp6dGTTz6pF198UX19fWpubo4tT5cxiddHsmNiTMhfuHBB58+flyQtWbJEpaWl+uc//xlb/tZbb+nkyZOxx5ZlLcg/wEhSe3u7XnnlFW3cuFENDQ3q7OzUk08+GVv+4Zk+r1y5siBn+ozXRzqNyblz59TT0xN7/OFa02VM4vWRTmNy1113ye/3a9myZVq8eLEefPDBGfeuSJcxiddHsmNiTMhfvHhRNTU1mpyc1OTkpP70pz/pS1/6Umz54sWL9Ytf/EJvvvmmLMvS8ePH9c1vfjOFFd/asWPH1NbWppaWFu3cuVMPPPCAqqurY8uXL1+urKws9fb2SpJaWloW5Eyf8fpIpzG5ceOGDh8+rHA4rLGxMTU3N8+oNV3GJF4f6TQma9eu1dmzZ3X9+nVFIhG9/PLLWrFiRWx5uoxJvD6SHhObfxBekBoaGqz169dbGzZssBoaGizLsqzt27dbfX19lmVZVkdHh1VeXm6VlpZae/futcLhcCrLnZNTp07Frkr5YC/nz5+3Nm3aZK1bt8768Y9/vOB7uVUf6TQmwWDQKisrs0pLS61nnnnGsqz0HJN4faTTmJw4cSJWa11dnRWJRNJyTOL1kcyYMAslABjMmNM1AICbEfIAYDBCHgAMRsgDgMEIeQAwGCEPAAYj5AHAYIQ8ABjs/wBR4IGwbNCPbwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(np_normal_dis, color=\"grey\", bins=21)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "42c6ecaa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "23"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Linear algebra\n",
    "### Dot product: product of two arrays\n",
    "f = np.array([1,2,3])\n",
    "g = np.array([4,5,3])\n",
    "### 1*4+2*5 + 3*6\n",
    "np.dot(f, g)  # 23"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "e33d87e3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[19, 22],\n",
       "       [43, 50]])"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Matmul: matruc product of two arrays\n",
    "h = [[1,2],[3,4]]\n",
    "i = [[5,6],[7,8]]\n",
    "### 1*5+2*7 = 19\n",
    "np.matmul(h, i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "026de611",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1.999999999999999"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.linalg.det(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "5b7952ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "Z = np.zeros((8,8))\n",
    "Z[1::2,::2] = 1\n",
    "Z[::2,1::2] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "8b9b2aa2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 1., 0., 1., 0., 1., 0., 1.],\n",
       "       [1., 0., 1., 0., 1., 0., 1., 0.],\n",
       "       [0., 1., 0., 1., 0., 1., 0., 1.],\n",
       "       [1., 0., 1., 0., 1., 0., 1., 0.],\n",
       "       [0., 1., 0., 1., 0., 1., 0., 1.],\n",
       "       [1., 0., 1., 0., 1., 0., 1., 0.],\n",
       "       [0., 1., 0., 1., 0., 1., 0., 1.],\n",
       "       [1., 0., 1., 0., 1., 0., 1., 0.]])"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "785aef9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_list = [ x + 2 for x in range(0, 11)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "bae5a829",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "d67c9ea4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np_arr = np.array(range(0, 11))\n",
    "np_arr + 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "97ddf308",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 7,  9, 11, 13, 15])"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "temp = np.array([1,2,3,4,5])\n",
    "pressure = temp * 2 + 5\n",
    "pressure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "fae909ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(temp,pressure)\n",
    "plt.xlabel('Temperature in oC')\n",
    "plt.ylabel('Pressure in atm')\n",
    "plt.title('Temperature vs Pressure')\n",
    "plt.xticks(np.arange(0, 6, step=0.5))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "01ed3a1a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Al Amin\\anaconda3\\lib\\site-packages\\seaborn\\distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).\n",
      "  warnings.warn(msg, FutureWarning)\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "mu = 28\n",
    "sigma = 15\n",
    "samples = 100000\n",
    "\n",
    "x = np.random.normal(mu, sigma, samples)\n",
    "ax = sns.distplot(x);\n",
    "ax.set(xlabel=\"x\", ylabel='y')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e8b3f1df",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
