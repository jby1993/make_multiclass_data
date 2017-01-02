#ifndef MAKE_DATA_FOR_MULTI_CLASSLEARN_H
#define MAKE_DATA_FOR_MULTI_CLASSLEARN_H
#include <Eigen/Dense>
#include <opencv/cv.h>
#include "tri_mesh.h"
#include "myrender.h"
//#include "random_num_generator.h"
using namespace Eigen;
class make_data_for_multi_classlearn
{
public:
    make_data_for_multi_classlearn();
    void set_render_to_label(bool val){m_render_to_label=val;}
    void set_thread_num(int num){m_thread_num=num;}
    void set_startseg_num(int num){m_start_segnum=num;}
    void set_lessendseg_num(int num){m_lessend_segnum=num;}
    void make_data(const std::string &root, const std::string &save_root);
    static void decode_img_to_compression_label_bin(const cv::Mat &img, const std::string &out);
    static void decode_compression_label_bin_to_img_label(const std::string &in,std::vector<uint16_t> &label, int &rows, int &cols);
    void set_threads(int thread_num);
private:
    void render_patches_on_img_for_multithread(cv::Mat &result, const cv::Mat &img, TriMesh &mesh, const Matrix3f &R, const Vector2f &T, const float &scale, int thread_id);
    void read_patches_and_neighbors(const std::string &file0, const std::string &file1);
    void read_mesh_para_thread(const std::string &file, int thread_num);
    void read_pose_to_local(const std::string &file, Matrix3f &R, Vector2f &weak_T, float &scale);
    void read_img_to_local(const std::string &file, cv::Mat &img);
    void update_local_mesh(TriMesh &mesh, int thread_id);
    void compute_predefined_colors();
    void set_patches_colors_for_multithread(std::vector<TriMesh::Color> &colors);
    void code_patches_colors_formultithread(std::vector<TriMesh::Color> &colors);
    void color_patch_mesh(TriMesh &mesh, const vector<TriMesh::Color> &colors);

    void get_meshpara_names(const std::string &root,std::vector<std::string> &names);
    void get_permesh_imgnames(const std::string &root, const std::vector<std::string> &meshfiles, std::vector<std::vector<std::string> > &names);
    void random_get_patch_color(const std::set<int> &exclude_color_ids, int &choosed_id/*, base_generator_type &gen*/);


private:
    MatrixXf m_mean_shape;
    MatrixXf m_pca_shape;
    MatrixXf m_shape_st;
    MatrixXf m_mean_exp;
    MatrixXf m_pca_exp;
    MatrixXf m_exp_st;
    int m_thread_num;
    int m_vnum;
    int m_shape_pcanum;
    int m_exp_pcanum;
    vector<VectorXf>    m_shapes;   //for multi thread
    vector<VectorXf>    m_exps;
    vector<TriMesh> m_meshs;    //for multi thread
    std::vector<int> m_partv_2_wholev;
    vector<myRender>    m_renders;
    std::vector<std::vector<int> >  m_patches;
    std::vector<std::vector<int> >  m_patch_neighbors;
    std::vector<int> m_keypoints_id;
    std::vector<TriMesh::Color> m_predefine_colors;
    int m_start_segnum;
    int m_lessend_segnum;
    bool    m_render_to_label;  //true: save label data to bin, used for learning; false: save label data to img, used for visiual check
};

#endif // MAKE_DATA_FOR_MULTI_CLASSLEARN_H
