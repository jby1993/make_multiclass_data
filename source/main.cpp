#include "io_utils.h"
#include "make_data_for_multi_classlearn.h"
#include <opencv2/highgui/highgui.hpp>
int main(int argc, char *argv[])
{

    if(argc!=6)
    {
        std::cout<<"input 5 para, one is thread num, second is if render_to_label, third is data root, the firth is save root, last is start seg num!"<<std::endl;
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
    ss<<num;
    int start_segnum=-1;
    ss>>start_segnum;
    if(start_segnum<0||start_segnum>=100)
    {
        std::cout<<"start seg num input is wrong!!! input 0~99 integer."<<std::endl;
        return -1;
    }
    else
        make.set_startseg_num(start_segnum);
    make.make_data(argv[3],argv[4]);


    std::cout<<"done"<<std::endl;
    return 0;
}
