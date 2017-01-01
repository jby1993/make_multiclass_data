#include <fstream>
#include <QString>
#include <QDir>
#include <QStringList>
#include <QString>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "io_utils.h"
#include "make_data_for_multi_classlearn.h"
#include <QDir>
int main(int argc, char *argv[])
{

//    if(argc!=4)
//    {
//        std::cout<<"input 3 para, one is data root, the other is save root!"<<std::endl;
//        return -1;
//    }
    make_data_for_multi_classlearn make;
//    if(std::string(argv[1])=="true")
//        make.set_render_to_label(true);
//    else if(std::string(argv[1])=="false")
        make.set_render_to_label(true);
//    else
//    {
//        std::cout<<"label flag input wrong!!!"<<std::endl;
//        return -1;
//    }
//    make.make_data(argv[2],argv[3]);
        make.make_data("../../multi_learn_data/data0/","../../patch_result/");




    return 0;
}
