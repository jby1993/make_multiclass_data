#ifndef MYRENDER_H
#define MYRENDER_H
#include <Eigen/Dense>
#include <vector>
#include "tri_mesh.h"
#include <opencv/cv.h>
using namespace Eigen;
using namespace std;
//do not use opengl, use computation to get visible render
//left bottom is origin point
class myRender
{
public:
    myRender();
    void initialSize(int width, int height);
    void setWeakPerspecPara(const Matrix3f &R, float tx, float ty, float tz, float scale);
    void RenderPatchMesh(TriMesh *mesh, cv::Mat &result);
    static void color_code(int N, std::vector<Eigen::Vector4i> &colors);
    static void color_decode(const Eigen::Vector4i &color, int &id);
    static void color_decode(const int &r,const int &g,const int &b, int &id);
private:
    void rasterizeTriangle(float x0, float y0, float x1, float y1, float x2, float y2, vector<int> &rastered_pixel_id);//left bottom origin, row major
    void bound_box(int x0,int y0, int x1, int y1, int x2, int y2, int &xmin, int &xmax, int &ymin, int &ymax);
    bool check_box_pixel_in_triangle(int x, int y, int x0,int y0, int x1, int y1, int x2, int y2);
    bool check_point_overlap(int x0, int y0, int x1, int y1);
    bool check_point_online(int x, int y, int a0, int b0, int a1, int b1);  //a1b1 and a0 b0 make sure are different points, and point is not a b
    bool check_point_oninfiniteline(int x, int y, int a0, int b0, int a1, int b1);
    bool check_point_in_triangle(int x, int y, int x0,int y0, int x1, int y1, int x2, int y2);  //all 4 point are no overlap
    float cross_sin(int l0x,int l0y,int l1x,int l1y);   //2 dian bu tong , han you fang xiang xin xi
    bool cull_face(const Vector3f &norm);   //weak perspective check method
    void clear_depth_buffer();
private:
    Matrix3f m_R;
    Vector3f m_T;
    float m_scale;
    vector<float>   m_depthbuffer;  //yue xiao yue yuan
    float  m_tolerance;
    int     m_width;
    int     m_height;
};

#endif // MYRENDER_H
