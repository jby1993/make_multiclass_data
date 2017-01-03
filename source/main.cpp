#include "io_utils.h"
#include "make_data_for_multi_classlearn.h"
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char *argv[])
{

    if(argc!=7)
    {
        std::cout<<"input 6 para, one is thread num, second is if render_to_label, third is data root, the firth is save root, the fifth is start seg num, the last is less end seg num!"<<std::endl;
        return -1;
    }
    make_data_for_multi_classlearn make;
    std::string num = argv[1];
    std::stringstream ss;
    ss<<num;
    int thread_num=-1;
    ss>>thread_num;
    if(thread_num<=0)
    {
        std::cout<<"thread num input is wrong!!! input greater than 0 integer."<<std::endl;
        return -1;
    }
    else
        make.set_thread_num(thread_num);
    if(std::string(argv[2])=="true")
        make.set_render_to_label(true);
    else if(std::string(argv[2])=="false")
        make.set_render_to_label(false);
    else
    {
        std::cout<<"label flag input wrong!!!"<<std::endl;
        return -1;
    }
    num = argv[5];
    std::stringstream tss;
    tss<<num;
    int start_segnum=-1;
    tss>>start_segnum;
    if(start_segnum<0||start_segnum>=100)
    {
        std::cout<<"start seg num input is wrong!!! input 0~99 integer."<<std::endl;
        return -1;
    }


    num = argv[6];
    std::stringstream ttss;
    ttss<<num;
    int lessend_segnum=-1;
    ttss>>lessend_segnum;
    if(lessend_segnum<=0||lessend_segnum>100)
    {
        std::cout<<"less end seg num input is wrong!!! input 1~100 integer."<<std::endl;
        return -1;
    }
    if(start_segnum>=lessend_segnum)
    {
        std::cout<<"start seg num is greater than lessend segnum!!! input start seg num less than less end seg num."<<std::endl;
        return -1;
    }
    else
    {
        make.set_startseg_num(start_segnum);
        make.set_lessendseg_num(lessend_segnum);
    }

    make.make_data(argv[3],argv[4]);


    std::cout<<"done"<<std::endl;
    return 0;
}
