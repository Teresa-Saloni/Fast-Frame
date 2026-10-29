# Locate a prebuilt ONNX Runtime (https://github.com/microsoft/onnxruntime/releases).
# Set ONNXRUNTIME_ROOT to the extracted tarball directory, e.g. ~/onnxruntime-linux-x64-1.26.0
find_path(ONNXRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT} PATH_SUFFIXES include)
find_library(ONNXRuntime_LIBRARY NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT} $ENV{ONNXRUNTIME_ROOT} PATH_SUFFIXES lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime DEFAULT_MSG ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)
if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime SHARED IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()
