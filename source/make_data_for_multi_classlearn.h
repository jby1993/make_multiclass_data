#ifndef MAKE_DATA_FOR_MULTI_CLASSLEARN_H
#define MAKE_DATA_FOR_MULTI_CLASSLEARN_H
#include <Eigen/Dense>
#include <opencv/cv.h>
#include "tri_mesh.h"
#include "myrender.h"
#include "random_num_generator.h"
using namespace Eigen;
class make_data_for_multi_classlearn
{
public:
    make_data_for_multi_classlearn();
    void set_render_to_label(bool val){m_render_to_label=val;}
    void make_data(const std::string &root, const std::string &save_root);
    static void decode_img_to_compression_label_bin(const cv::Mat &img, const std::string &out);
    static void decode_compression_label_bin_to_img_label(const std::string &in,std::vector<uint16_t> &label, int &rows, int &cols);
private:
    void render_patches_on_img(cv::Mat &result);
    void render_keypoints_on_img(cv::Mat &result);
    void read_patches_and_neighbors(const std::string &file0, const std::string &file1);
    void read_mesh_para(const std::string &file);
    void read_pose_para(const std::string &file);
    void read_img(const std::string &file);
    void update_mesh();
    void compute_predefined_colors();
    void set_patches_colors();
    void code_patches_colors();
    void set_keypoints_colors();
    void code_keypoints_colors();

    void random_get_patch_color(const std::set<int> &exclude_color_ids, int &choosed_id, base_generator_type &gen);
private:
    MatrixXf m_mean_shape;
    MatrixXf m_pca_shape;
    MatrixXf m_shape_st;
    MatrixXf m_mean_exp;
    MatrixXf m_pca_exp;
    MatrixXf m_exp_st;
    int m_vnum;
    int m_shape_pcanum;
    int m_exp_pcanum;
    VectorXf m_shape;
    VectorXf m_exp;
    cv::Mat m_img;
    Matrix3f m_R;
    Vector2f m_weak_T;
    TriMesh m_mesh;
    float m_scale;
    std::vector<int> m_partv_2_wholev;
    myRender m_render;
    std::vector<std::vector<int> >  m_patches;
    std::vector<std::vector<int> >  m_patch_neighbors;
    std::vector<int> m_keypoints_id;
    std::vector<TriMesh::Color> m_predefine_colors;
    bool    m_render_to_label;  //true: save label data to bin, used for learning; false: save label data to img, used for visiual check
};

#endif // MAKE_DATA_FOR_MULTI_CLASSLEARN_H
