#include "myrender.h"

myRender::myRender()
{
    m_tolerance = 0.01;
}

void myRender::initialSize(int width, int height)
{
    m_width = width;
    m_height= height;
    clear_depth_buffer();
}

void myRender::setModelViewMatrix(const Matrix3f &R, float tx, float ty, float tz)
{
    m_R = R;
    m_T(0) = tx;
    m_T(1) = ty;
    m_T(2) = tz;
}
//need mesh has face normal status.
void myRender::RenderPatchMesh(TriMesh *mesh, cv::Mat &result)
{
    clear_depth_buffer();
    TriMesh::ConstFaceIter        f_it(mesh->faces_begin()),
                               f_end(mesh->faces_end());
    TriMesh::ConstFaceVertexIter  fv_it;

    float *addrv = mesh->points(mesh->vertices_begin().handle()).data();
    Eigen::MatrixXf temp = Eigen::Map<Eigen::MatrixXf>(addrv, 3, mesh->n_vertices());
    Eigen::MatrixXf points = m_R*temp+m_T;
    std::vector<float>  img;
    img.resize(3*m_width*m_height,0.0);
    for (; f_it!=f_end; ++f_it)       //per vertex normal must be seted before per vertex position!!!
    {
        TriMesh::Normal tn = mesh->normal(fv_it);
        Vector3f norm(tn[0],tn[1],tn[2]);
        norm = m_R*norm;
        if(cull_face(norm))
            continue;
        TriMesh::Color color = mesh->color(f_it);
        Vector3f p;
        fv_it = mesh->cfv_iter(f_it.handle());
        p = points.col((*fv_it).idx());
        float x0,y0,z0;
        x0=p(0);    y0=p(1);    z0=p(2);
        ++fv_it;
        p = points.col((*fv_it).idx());
        float x1,y1,z1;
        x1=p(0);    y1=p(1);    z1=p(2);
        ++fv_it;
        p = points.col((*fv_it).idx());
        float x2,y2,z2;
        x2=p(0);    y2=p(1);    z2=p(2);
        float zmin=z0;
        if(z1<zmin) zmin=z1;
        if(z2<zmin) zmin=z2;        //zhe li yong san jiao xin zui xiao de shen du dai ti zheng ge san jiao xing shen du, wei le jian dan
        std::vector<int>    ids;
        rasterizeTriangle(x0,y0,x1,y1,x2,y2, ids);
        for(int i=0; i<ids.size(); i++)
        {
            int id=ids[i];
            if(m_depthbuffer[id]<zmin)
            {
                img[3*id]=color[0];
                img[3*id+1]=color[1];
                img[3*id+2]=color[2];
                m_depthbuffer[id]=zmin;
            }
        }
    }

    result.create(m_height,m_width,CV_8UC3);
    int rv, gv, bv;
    for (int i = 0; i < m_height; i++)
    {
        for (int j = 0; j < m_width; j++)
        {
            rv = (int)img[(i * m_width + j) * 3]*255.0;
            gv = (int)img[(i * m_width + j) * 3 + 1]*255.0;
            bv = (int)img[(i * m_width + j) * 3 + 2]*255.0;
            result.at<cv::Vec3b>(m_height - 1 - i, j)[2] = rv;
            result.at<cv::Vec3b>(m_height - 1 - i, j)[1] = gv;
            result.at<cv::Vec3b>(m_height - 1 - i, j)[0] = bv;
        }
    }
}

void myRender::rasterizeTriangle(float x0, float y0, float x1, float y1, float x2, float y2, vector<int> &rastered_pixel_id)
{
    int a0=int(x0+0.5);
    int b0=int(y0+0.5);
    int a1=int(x1+0.5);
    int b1=int(y1+0.5);
    int a2=int(x2+0.5);
    int b2=int(y2+0.5);
    int amin,amax,bmin,bmax;
    bound_box(a0,b0,a1,b1,a2,b2,amin,amax,bmin,bmax);
    rastered_pixel_id.clear();
    for(int y=bmin; y<=bmax; y++)
    {
        for(int x=amin; x<=amax; x++)
        {
                if(check_box_pixel_in_triangle(x,y,a0,b0,a1,b1,a2,b2))
                    rastered_pixel_id.push_back(y*m_width+x);
        }
    }

}

void myRender::bound_box(int x0, int y0, int x1, int y1, int x2, int y2, int &xmin, int &xmax, int &ymin, int &ymax)
{
    xmin=x0;
    xmax=x0;
    ymin=y0;
    ymax=y0;
    if(x1<xmin)
        xmin=x1;
    if(x1>xmax)
        xmax=x1;
    if(y1<ymin)
        ymin=y1;
    if(y1>ymax)
        ymax=y1;
    if(x2<xmin) xmin=x2;
    if(x2>xmax) xmax=x2;
    if(y2<ymin) ymin=y2;
    if(y2>ymax) ymax=y2;
}

bool myRender::check_box_pixel_in_triangle(int x, int y, int x0, int y0, int x1, int y1, int x2, int y2)
{
    if(check_point_overlap(x,y,x0,y0)||check_point_overlap(x,y,x1,y1)||check_point_overlap(x,y,x2,y2))
        return true;
    if(check_point_overlap(x0,y0,x1,y1))
    {
        if(check_point_overlap(x0,y0,x2,y2))
            return false;
        else
        {
            return check_point_online(x,y,x0,y0,x2,y2);
        }
    }
    else if(check_point_overlap(x0,y0,x2,y2))
    {
        return check_point_online(x,y,x0,y0,x1,y1);
    }
    else
    {
        if(check_point_overlap(x1,y1,x2,y2))
            return check_point_online(x,y,x0,y0,x1,y1);
        else    //3 ge ding dian bu overlap, x y ye bu yu ren yi ge overlap
        {
            return check_point_in_triangle(x,y,x0,y0,x1,y1,x2,y2);
        }
    }
}

bool myRender::check_point_overlap(int x0, int y0, int x1, int y1)
{
    if(x0==x1&&y0==y1)
        return true;
    else
        return false;
}

bool myRender::check_point_online(int x, int y, int a0, int b0, int a1, int b1)//3 dian hu bu xiang deng
{
    float l0x = float(a0-x);
    float l0y = float(b0-y);
    float l1x = float(a1-x);
    float l1y = float(b1-y);
    float l0 = sqrt(l0x*l0x+l0y*l0y);
    float l1 = sqrt(l1x*l1x+l1y*l1y);
    float cos = (l0x*l1x+l0y*l1y)/(l0*l1);
    if(fabs(cos+1)<m_tolerance)
        return true;
    else
        return false;
}

bool myRender::check_point_in_triangle(int x, int y, int x0, int y0, int x1, int y1, int x2, int y2)
{
    int l01x = x1-x0;
    int l01y = y1-y0;
    int l02x = x2-x0;
    int l02y = y2-y0;
    float sign_sin201=cross_sin(l02x,l02y,l01x,l01y);
    if(fabs(sign_sin201)<m_tolerance) //triangle is a line
    {
        if(check_point_online(x,y,x0,y0,x1,y1)||check_point_online(x,y,x1,y1,x2,y2))
            return true;
        else
            return false;
    }
    else    //triang is normal triangle, jian cha dian yu san bian dui ying ding dian shi fou dou tong ce
    {
        int l0x = x-x0;
        int l0y = y-y0;
        float sign_sinp01=cross_sin(l0x,l0y,l01x,l01y);
        bool check1=false;
        if(sign_sinp01*sign_sin201>0)   check1=true;

        int l1x = x-x1;
        int l1y = y-y1;
        int l12x = x2-x1;
        int l12y = y2-y1;
        float sign_sinp12=cross_sin(l1x,l1y,l12x,l12y);
        int l10x = x0-x1;
        int l10y = y0-y1;
//        int l12x = x2-x1;
//        int l12y = y2-y1;
        float sign_sin012=cross_sin(l10x,l10y,l12x,l12y);
        bool check2=false;
        if(sign_sinp12*sign_sin012>0)   check2=true;

        int l2x = x-x2;
        int l2y = y-y2;
        int l20x = x0-x2;
        int l20y = y0-y2;
        float sign_sinp20=cross_sin(l2x,l2y,l20x,l20y);
        int l21x = x1-x2;
        int l21y = y1-y2;
        float sign_sin120=cross_sin(l21x,l21y,l20x,l20y);
        bool check3 = false;
        if(sign_sinp20*sign_sin120>0)   check3=true;

        if(check1&&check2&&check3)
            return true;
        else
            return false;

    }


}

float myRender::cross_sin(int l0x, int l0y, int l1x, int l1y)   //2 dian are different
{
    float len0 = sqrt(float(l0x*l0x+l0y*l0y));
    float len1 = sqrt(float(l1x*l1x+l1y*l1y));
    return float(l0x*l1y-l1x*l0y)/(len0*len1);
}

bool myRender::cull_face(const Vector3f &norm)
{
    if(norm(2)>0)
        return true;
    else
        return false;
}

void myRender::clear_depth_buffer()
{
    m_depthbuffer.resize(m_width*m_height, -100000000);
}
