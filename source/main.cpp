#include <QGuiApplication>
#include <fstream>
#include <QString>
#include <QDir>
#include <QStringList>
#include <QString>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "io_utils.h"
#include "make_data_for_multi_classlearn.h"
//#include "utils.h"
int main(int argc, char *argv[])
{
    QGuiApplication a(argc, argv);
//    RandomMesh mesh_gen;
//    mesh_gen.saveRandomImages(1);
//    train_test  TrainTest("../test_data/","../Data/");
//    //train on my synthesis imgs
//    TrainTest.read_groundtruth_data();
//    TrainTest.train_para_only();
//    TrainTest.save_para_result();
    //test for my synthesis imgs
//    TrainTest.set_test(true);
//    TrainTest.read_groundtruth_data();
//    TrainTest.test_para_only();

//    //test for true data
//    TrainTest.set_test(true);
//    TrainTest.test_true_imgs("../true_imgs/");

//    rotate_fitted_face_image tool;
//    std::string root = "/home/john/桌面/PoseFaceImage/savetop/";
//    std::string save_root = "../rotate_images/";
//    tool.readImage(root+"afw_1_ori.jpg");
//    tool.readData(root+"afw_1_parat.txt");
//    tool.updataData();
//    tool.saveMesh(save_root+"mesh.obj");
//    tool.compute_faceimage_corresponding_pointcloud();
//    tool.savePointClound(save_root+"pointcloud.off");


//    int patchnum=500;
//    std::string name="part_face.obj";
//    std::string base_name=name.substr(0,name.size()-4);
//    std::string mesh_file_root = "../3dmm_mesh_part/";
//    std::string save_root = "../part_face_as_whole_seg_result/";
//    TriMesh mesh;
//    OpenMesh::IO::read_mesh(mesh,mesh_file_root+name);
//    random_patch_segmentation_mesh segtool;
//    segtool.setMesh(&mesh);
//    std::cout<<name<<" start seg!"<<std::endl;
//    for(int j=0;j<100;j++)
//    {
//        std::vector<std::vector<int> > patches,neighbors;
//        segtool.random_seg_mesh_with_geodesic_distance(patchnum,patches,neighbors);
//        std::string save_name = save_root+base_name+"_segs_"+std::to_string(j)+".txt";
//        segtool.save_patches(save_name,patches);
//        save_name = save_root+base_name+"_seg_neighbors_"+std::to_string(j)+".txt";
//        segtool.save_patch_neighbors(save_name,neighbors);
//        std::cout<<name<<" "<<std::to_string(j)<<" done."<<std::endl;
//    }


    if(argc!=4)
    {
        std::cout<<"input 3 para, one is data root, the other is save root!"<<std::endl;
        return -1;
    }
    make_data_for_multi_classlearn make;
    if(std::string(argv[1])=="true")
        make.set_render_to_label(true);
    else if(std::string(argv[1])=="false")
        make.set_render_to_label(false);
    else
    {
        std::cout<<"label flag input wrong!!!"<<std::endl;
        return -1;
    }
    make.make_data(argv[2],argv[3]);





    
    

    
    
//    rectify_img_with_faceRect img_preprocess;
//    img_preprocess.ReadAndSaveRectImgs();
//    test_detector test_dec;
//    test_dec.test_SIFT_on_my_data();
//    test_dec.read_an_32f_img_and_features_descriptors_from_matlab_code("../Data/test_feature_data/roofs1.jpg",
//                                                                       "../Data/test_feature_data/frames.bin",
//                                                                       "../Data/test_feature_data/descriptors.bin");
//    test_dec.compare_SIFT_with_matlab_result();






    return 0;
}
