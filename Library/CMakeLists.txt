set(SOURCES Raisr.cpp RaisrHandler.cpp)
set(HEADERS Raisr.h ThreadPool.h RaisrHandler.h RaisrDefaults.h)

add_library(raisr STATIC ${SOURCES} ${HEADERS})

# Link our library
# for IPP
if( UNIX )
include_directories($ENV{IPPROOT}/include)
find_library(IPP_CORE libippcore.a PATHS $ENV{IPPROOT}/lib/intel64)
find_library(IPP_VM libippvm.a PATHS     $ENV{IPPROOT}/lib/intel64)
find_library(IPP_S libipps.a PATHS       $ENV{IPPROOT}/lib/intel64)
find_library(IPP_I libippi.a PATHS       $ENV{IPPROOT}/lib/intel64)
find_package(Threads REQUIRED)
else()
include_directories("C:/Program Files (x86)/Intel/oneAPI/ipp/2021.2.0/include")
find_library(IPP_CORE ippcore.lib PATHS "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.2.0/lib/intel64")
find_library(IPP_VM ippvm.lib PATHS     "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.2.0/lib/intel64")
find_library(IPP_S ipps.lib PATHS       "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.2.0/lib/intel64")
find_library(IPP_I ippi.lib PATHS       "C:/Program Files (x86)/Intel/oneAPI/ipp/2021.2.0/lib/intel64")
endif(UNIX)
target_link_libraries(raisr ${IPP_I} ${IPP_VM} ${IPP_S} ${IPP_CORE} ${CMAKE_THREAD_LIBS_INIT})
  include(GNUInstallDirs)
  install(DIRECTORY "." DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/raisr" FILES_MATCHING PATTERN "*.h")
  install(TARGETS "raisr" LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

