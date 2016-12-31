#ifndef RENDER_H
#define RENDER_H
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions_1_2>
#include "tri_mesh.h"
#include <Eigen/Dense>
#include <opencv/cv.h>

class Render
{
public:
    Render();
    void initialOpenGL();
    void RenderBackground(cv::Mat &back);
    void setMVPmatrix(const Eigen::Matrix3f &R, const Eigen::Vector2f &T,float scale, const BOX &box );
    void setModelViewMatrix(const Eigen::Matrix3f &R, float tx, float ty, float tz);
    void setOrtho(float left, float right, float bottom, float top, float zNear, float zFar );
    void setViewPort(int x, int y, int width, int height);
    void set_render_env(const BOX &box);
    void RenderImage(TriMesh *mesh, cv::Mat &img);
    void RenderPatchMesh(TriMesh *mesh, cv::Mat &img);
    void RenderVisibleKeyPoints(TriMesh *mesh, const std::vector<int> &ids,cv::Mat &img);
    void compute_mesh_points_visible(TriMesh *mesh, const std::vector<int> &ids, std::vector<bool> &visibles);
    void initialFBO(QSize size);
    void estimation_face_image_whole_depthmesh(TriMesh *mesh, Eigen::MatrixXf &pro, const Eigen::Matrix3f &R, const Eigen::Vector2f T, float scale, int width, int height, int anchor, std::vector<float> &depths);   //ref Face Alignment Across Large Poses: A 3D Solution, used to rotate face image to argument data

    static void color_code(int N, std::vector<Eigen::Vector4i> &colors);
    static void color_decode(const Eigen::Vector4i &color, int &id);
    static void color_decode(const int &r,const int &g,const int &b, int &id);

private:
    GLuint make_texture(uchar* img, int width, int height, int type);


private:
    QOpenGLFunctions_1_2* m_f;
    QOffscreenSurface m_surface;
    QOpenGLContext   m_context;
    QOpenGLFramebufferObject *m_fbo;
    int m_width;
    int m_height;
};

#endif // RENDER_H
