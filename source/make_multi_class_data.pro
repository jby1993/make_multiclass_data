QT += core
 QT -= gui

CONFIG += c++11

TARGET = datamake
CONFIG += console
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -fopenmp

 TEMPLATE = app

  SOURCES += main.cpp \
     io_utils.cpp \
     make_data_for_multi_classlearn.cpp \
     myrender.cpp

 HEADERS += \
#    random_num_generator.h \
     tri_mesh.h \
     io_utils.h \
    make_data_for_multi_classlearn.h \
     myrender.h

 INCLUDEPATH +=./ \
                             /usr/include \
                           /usr/include/eigen3 \
                           /usr/include/suitesparse \
                             /usr/local/include \
#                           /home/john/boost_1_62_0 \



  LIBS += -L/usr/lib \
          -L/usr/local/lib \
         -L/usr/local/lib/OpenMesh \
         -L/usr/lib/x86_64-linux-gnu \

 LIBS+=-lOpenMeshCore \
         -lOpenMeshTools \
         -lopencv_core \
         -lopencv_highgui \
         -lopencv_imgproc \
          -lgomp -lpthread \
