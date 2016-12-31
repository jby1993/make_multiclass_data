QT += core opengl
QT -= gui

CONFIG += c++11

TARGET = datamake
CONFIG += console
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -fopenmp

TEMPLATE = app

SOURCES += main.cpp \
#    randommesh.cpp \
    render.cpp \
#    train_test.cpp \
#    rectify_img_with_facerect.cpp \
#    featuredetector.cpp \
#    siftdectector.cpp \
#    test/test_detector.cpp \
#    rotate_fitted_face_image.cpp \
#    random_patch_segmentation_mesh.cpp \
#    utils.cpp \
    io_utils.cpp \
    make_data_for_multi_classlearn.cpp

HEADERS += \
#    randommesh.h \
    random_num_generator.h \
    tri_mesh.h \
    render.h \
#    train_test.h \
#    rectify_img_with_facerect.h \
#    FaceRecognitionLib.h \
#    featuredetector.h \
#    siftdectector.h \
#    test/test_detector.h \
#    rotate_fitted_face_image.h \
#    random_patch_segmentation_mesh.h \
#    utils.h \
    io_utils.h \
    make_data_for_multi_classlearn.h

INCLUDEPATH +=./ \
                             /usr/include \
                            /usr/include/eigen3 \
                            /usr/include/suitesparse \
                            /usr/local/include \
                            /home/john/boost_1_62_0 \
                            /home/john/vlfeat-0.9.20/vl \

#LIBS += ../Lib/FaceRecognitionLib.a

LIBS += -L/usr/lib \
        -L../Lib \
        -L/usr/local/lib \
        -L/usr/local/lib/OpenMesh \
        -L/home/john/vlfeat-0.9.20/bin/glnxa64 \
        -L/usr/lib/x86_64-linux-gnu \

LIBS+=-lOpenMeshCore \
        -lOpenMeshTools \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc \
#        -lopencv_objdetect \
#        -lcholmod \
#        -lspqr \
#        -llapack \
#        -lblas \
#        -lvl \
        -lgomp -lpthread \
#        -lCGAL -lCGAL_Core -lCGAL_ImageIO   -lgmp
