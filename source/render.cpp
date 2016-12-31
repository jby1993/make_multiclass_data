#include "render.h"
#include <QOpenGLFunctions>
#include <QOpenGLExtraFunctions>
#include <QVector>
#include <QDebug>
#include <QMatrix4x4>
Render::Render()
{
    initialOpenGL();
}

void Render::setMVPmatrix(const Eigen::Matrix3f &R, const Eigen::Vector2f &T, float scale, const BOX &box)
{
    Eigen::Matrix4f mv; mv.setZero();
    mv.block<3,3>(0,0) = R;
    Eigen::Vector2f n_T = T/scale;
    mv(0,3) = n_T(0); mv(1,3) = n_T(1); mv(3,3) = 1.0;
    Eigen::Matrix4f temp;   temp.setZero();
    temp(0,0)=1.0;  temp(1,1)=-1.0; temp(2,2)=-1.0;   temp(3,3)=1.0;
    mv = temp*mv;
    m_f->glMatrixMode(GL_MODELVIEW);
    m_f->glLoadMatrixf(mv.data());
    m_f->glMatrixMode(GL_PROJECTION);
    m_f->glLoadIdentity();
    float width = m_width/scale;
    float height = m_height/scale;
    m_f->glOrtho(0.0,width,-height,0.0,box.z_min-box.z_len/1000.0,box.z_max+box.z_len/1000.0);

    m_f->glViewport(0,0,m_width,m_height);
    set_render_env(box);

}

void Render::setModelViewMatrix(const Eigen::Matrix3f &R, float tx, float ty, float tz)
{
    Eigen::Matrix4f mv; mv.setZero();
    mv.block<3,3>(0,0) = R;
    mv(0,3) = tx; mv(1,3) = ty; mv(2,3) = tz;   mv(3,3) = 1.0;
    m_f->glMatrixMode(GL_MODELVIEW);
    m_f->glLoadIdentity();
    m_f->glLoadMatrixf(mv.data());
}

void Render::setOrtho(float left, float right, float bottom, float top, float zNear, float zFar)
{
    m_f->glMatrixMode(GL_PROJECTION);
    m_f->glLoadIdentity();
    m_f->glOrtho(left,right,bottom,top,zNear,zFar);
}

void Render::setViewPort(int x, int y, int width, int height)
{
    m_f->glViewport(x,y,width,height);
}

void Render::RenderImage(TriMesh *mesh, cv::Mat &img)
{
//    initialFBO(QSize(width,height));
    m_fbo->bind();
//    m_context.functions()->glBindFramebuffer(GL_FRAMEBUFFER, m_fbo->handle());

//    set_render_env(m_f,0.5*x_len,0.5*y_len,0.5*z_len);
//    m_f->glMatrixMode(GL_PROJECTION);
//    m_f->glLoadIdentity();
//    m_f->glOrtho(-0.5*len,0.5*len,-0.5*len,0.5*len, -0.6*z_len, 0.6*z_len);
//    m_f->glViewport(0,0,width,height);


    m_f->glEnable(GL_DEPTH_TEST);
    m_f->glEnable(GL_CULL_FACE);
    m_f->glEnable(GL_LIGHTING);
    m_f->glEnable(GL_COLOR_MATERIAL);
    m_f->glEnable(GL_SMOOTH);

    m_f->glEnableClientState(GL_VERTEX_ARRAY);
    m_f->glEnableClientState(GL_NORMAL_ARRAY);
    m_f->glEnableClientState(GL_COLOR_ARRAY);
    m_f->glVertexPointer(3,GL_FLOAT,0,mesh->points());
    m_f->glNormalPointer(GL_FLOAT,0,mesh->vertex_normals());
    m_f->glColorPointer(3,GL_FLOAT,0,mesh->vertex_colors());

    TriMesh::ConstFaceIter        f_it(mesh->faces_sbegin()),
                               f_end(mesh->faces_end());
    TriMesh::ConstFaceVertexIter  fv_it;
    std::vector<unsigned int> indices;
    indices.reserve(mesh->n_faces()*3);
    for (; f_it!=f_end; ++f_it)
    {
      indices.push_back((fv_it=mesh->cfv_iter(f_it)).handle().idx());
      indices.push_back((++fv_it).handle().idx());
      indices.push_back((++fv_it).handle().idx());
    }

    m_f->glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

    m_f->glDisableClientState(GL_VERTEX_ARRAY);
    m_f->glDisableClientState(GL_NORMAL_ARRAY);
    m_f->glDisableClientState(GL_COLOR_ARRAY);
    m_f->glDisable(GL_DEPTH_TEST);
    m_f->glDisable(GL_CULL_FACE);
    m_f->glDisable(GL_LIGHTING);
    m_f->glDisable(GL_COLOR_MATERIAL);

    m_context.extraFunctions()->glReadBuffer(GL_COLOR_ATTACHMENT0);
    QVector<unsigned char> m_pixels;
    m_pixels.resize(m_width*m_height*4);
    m_f->glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, &m_pixels[0]);

    m_fbo->release();
    img.create(m_height,m_width,CV_8UC3);
    int rv, gv, bv;
    for (int i = 0; i < m_height; i++)
    {
        for (int j = 0; j < m_width; j++)
        {
            rv = (int)m_pixels[(i * m_width + j) * 4];
            gv = (int)m_pixels[(i * m_width + j) * 4 + 1];
            bv = (int)m_pixels[(i * m_width + j) * 4 + 2];
            img.at<cv::Vec3b>(m_height - 1 - i, j)[2] = rv;
            img.at<cv::Vec3b>(m_height - 1 - i, j)[1] = gv;
            img.at<cv::Vec3b>(m_height - 1 - i, j)[0] = bv;
        }
    }
}

void Render::RenderPatchMesh(TriMesh *mesh, cv::Mat &img)
{
        m_fbo->bind();

        TriMesh::ConstFaceIter        f_it(mesh->faces_begin()),
                                   f_end(mesh->faces_end());
        TriMesh::ConstFaceVertexIter  fv_it;

        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_LIGHTING);
        glShadeModel(GL_SMOOTH);
        glEnable(GL_SMOOTH);
        glDisable(GL_COLOR_MATERIAL);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
//        glEnable(GL_BLEND);
//        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR);
        glBegin(GL_TRIANGLES);
        for (; f_it!=f_end; ++f_it)       //per vertex normal must be seted before per vertex position!!!
        {
          glColor3fv(mesh->color(f_it).data());
          fv_it = mesh->cfv_iter(f_it.handle());
//          glNormal3fv(mesh->normal(fv_it).data());
          glVertex3fv(mesh->point(fv_it).data());

          ++fv_it;
//          glNormal3fv(mesh->normal(fv_it).data());
          glVertex3fv(mesh->point(fv_it).data());
          ++fv_it;
//          glNormal3fv(mesh->normal(fv_it).data());
          glVertex3fv(mesh->point(fv_it).data());
        }
        glEnd();
//this method has problem, color is not right!
//        m_f->glEnableClientState(GL_VERTEX_ARRAY);
//        m_f->glEnableClientState(GL_COLOR_ARRAY);
//        m_f->glVertexPointer(3,GL_FLOAT,0,mesh->points());
//        m_f->glColorPointer(3,GL_FLOAT,0,mesh->property(mesh->face_colors_pph()).data());

//        std::vector<unsigned int> indices;
//        indices.reserve(mesh->n_faces()*3);
//        for (; f_it!=f_end; ++f_it)
//        {
//          indices.push_back((fv_it=mesh->cfv_iter(f_it)).handle().idx());
//          indices.push_back((++fv_it).handle().idx());
//          indices.push_back((++fv_it).handle().idx());
//        }

//        m_f->glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

//        m_f->glDisableClientState(GL_VERTEX_ARRAY);
//        m_f->glDisableClientState(GL_COLOR_ARRAY);

        m_f->glDisable(GL_DEPTH_TEST);
        m_f->glDisable(GL_CULL_FACE);
        m_f->glDisable(GL_LIGHTING);
//        glDisable(GL_BLEND);
        m_f->glDisable(GL_COLOR_MATERIAL);

        m_context.extraFunctions()->glReadBuffer(GL_COLOR_ATTACHMENT0);
        QVector<unsigned char> m_pixels;
        m_pixels.resize(m_width*m_height*4);
        m_f->glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, &m_pixels[0]);

        m_fbo->release();
        img.create(m_height,m_width,CV_8UC3);
        int rv, gv, bv;
        for (int i = 0; i < m_height; i++)
        {
            for (int j = 0; j < m_width; j++)
            {
                rv = (int)m_pixels[(i * m_width + j) * 4];
                gv = (int)m_pixels[(i * m_width + j) * 4 + 1];
                bv = (int)m_pixels[(i * m_width + j) * 4 + 2];
                img.at<cv::Vec3b>(m_height - 1 - i, j)[2] = rv;
                img.at<cv::Vec3b>(m_height - 1 - i, j)[1] = gv;
                img.at<cv::Vec3b>(m_height - 1 - i, j)[0] = bv;
            }
        }
}

void Render::RenderVisibleKeyPoints(TriMesh *mesh, const std::vector<int> &ids, cv::Mat &img)
{
    m_fbo->bind();

    m_f->glEnable(GL_DEPTH_TEST);
    m_f->glEnable(GL_CULL_FACE);
    m_f->glDisable(GL_LIGHTING);
    m_f->glDisable(GL_COLOR_MATERIAL);
    m_f->glEnable(GL_SMOOTH);

    m_f->glEnableClientState(GL_VERTEX_ARRAY);
    m_f->glEnableClientState(GL_COLOR_ARRAY);
    Eigen::MatrixXf tcolors(3,mesh->n_vertices());   tcolors.setZero();
    m_f->glVertexPointer(3,GL_FLOAT,0,mesh->points());
    m_f->glColorPointer(3,GL_FLOAT,0,tcolors.data());

    TriMesh::ConstFaceIter        f_it(mesh->faces_sbegin()),
                               f_end(mesh->faces_end());
    TriMesh::ConstFaceVertexIter  fv_it;
    std::vector<unsigned int> indices;
    indices.reserve(mesh->n_faces()*3);
    for (; f_it!=f_end; ++f_it)
    {
      indices.push_back((fv_it=mesh->cfv_iter(f_it)).handle().idx());
      indices.push_back((++fv_it).handle().idx());
      indices.push_back((++fv_it).handle().idx());
    }

    m_f->glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

    m_f->glDisableClientState(GL_VERTEX_ARRAY);
    m_f->glDisableClientState(GL_COLOR_ARRAY);

    m_f->glPointSize(2.0);
    m_f->glBegin(GL_POINTS);
    for(int i=0;i<ids.size();i++)
    {
        m_f->glColor3fv(mesh->color(TriMesh::VertexHandle(ids[i])).data());
        m_f->glVertex3fv(mesh->point(TriMesh::VertexHandle(ids[i])).data());
    }
    m_f->glEnd();
    m_f->glPointSize(1.0);
//    m_f->glColor4fv(old_color);
    m_f->glDisable(GL_DEPTH_TEST);
    m_f->glDisable(GL_CULL_FACE);
    m_f->glEnable(GL_LIGHTING);
//    m_f->glEnable(GL_COLOR_MATERIAL);

    m_context.extraFunctions()->glReadBuffer(GL_COLOR_ATTACHMENT0);
    QVector<unsigned char> m_pixels;
    m_pixels.resize(m_width*m_height*4);
    m_f->glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, &m_pixels[0]);
    m_fbo->release();
    img.create(m_height,m_width,CV_8UC3);
    int rv, gv, bv;
    for (int i = 0; i < m_height; i++)
    {
        for (int j = 0; j < m_width; j++)
        {
            rv = (int)m_pixels[(i * m_width + j) * 4];
            gv = (int)m_pixels[(i * m_width + j) * 4 + 1];
            bv = (int)m_pixels[(i * m_width + j) * 4 + 2];
            img.at<cv::Vec3b>(m_height - 1 - i, j)[2] = rv;
            img.at<cv::Vec3b>(m_height - 1 - i, j)[1] = gv;
            img.at<cv::Vec3b>(m_height - 1 - i, j)[0] = bv;
        }
    }
}

void Render::compute_mesh_points_visible(TriMesh *mesh, const std::vector<int> &ids, std::vector<bool> &visibles)
{
        m_fbo->bind();

        m_f->glEnable(GL_DEPTH_TEST);
        m_f->glEnable(GL_CULL_FACE);
        m_f->glDisable(GL_LIGHTING);
        m_f->glDisable(GL_COLOR_MATERIAL);
        m_f->glEnable(GL_SMOOTH);

        m_f->glEnableClientState(GL_VERTEX_ARRAY);
        m_f->glEnableClientState(GL_COLOR_ARRAY);
        Eigen::MatrixXf tcolors(3,mesh->n_vertices());   tcolors.setZero();
//        colors.row(0).setOnes();
        m_f->glVertexPointer(3,GL_FLOAT,0,mesh->points());
        m_f->glColorPointer(3,GL_FLOAT,0,tcolors.data());

        TriMesh::ConstFaceIter        f_it(mesh->faces_sbegin()),
                                   f_end(mesh->faces_end());
        TriMesh::ConstFaceVertexIter  fv_it;
        std::vector<unsigned int> indices;
        indices.reserve(mesh->n_faces()*3);
        for (; f_it!=f_end; ++f_it)
        {
          indices.push_back((fv_it=mesh->cfv_iter(f_it)).handle().idx());
          indices.push_back((++fv_it).handle().idx());
          indices.push_back((++fv_it).handle().idx());
        }

        m_f->glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

        m_f->glDisableClientState(GL_VERTEX_ARRAY);
        m_f->glDisableClientState(GL_COLOR_ARRAY);

        std::vector<Eigen::Vector4i> colors;
        color_code(ids.size(),colors);
        m_f->glPointSize(2.0);
        float old_color[4];
        m_f->glGetFloatv(GL_CURRENT_COLOR, old_color);
        m_f->glBegin(GL_POINTS);
        for(int i=0;i<ids.size();i++)
        {
            m_f->glColor3f(colors[i](0)/255.0,colors[i](1)/255.0,colors[i](2)/255.0);
            m_f->glVertex3fv(mesh->point(TriMesh::VertexHandle(ids[i])).data());
        }
        m_f->glEnd();
        m_f->glPointSize(1.0);
        m_f->glColor4fv(old_color);
        m_f->glDisable(GL_DEPTH_TEST);
        m_f->glDisable(GL_CULL_FACE);
        m_f->glEnable(GL_LIGHTING);
        m_f->glEnable(GL_COLOR_MATERIAL);

        m_context.extraFunctions()->glReadBuffer(GL_COLOR_ATTACHMENT0);
        QVector<unsigned char> m_pixels;
        m_pixels.resize(m_width*m_height*4);
        m_f->glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, &m_pixels[0]);
        m_fbo->release();
        std::vector<bool>   checkd;
        checkd.resize(ids.size(),false);
        visibles.resize(ids.size(),false);
//        img.create(m_height,m_width,CV_8UC3);
        int rv, gv, bv;
        for (int i = 0; i < m_height; i++)
        {
            for (int j = 0; j < m_width; j++)
            {
                rv = (int)m_pixels[(i * m_width + j) * 4];
                gv = (int)m_pixels[(i * m_width + j) * 4 + 1];
                bv = (int)m_pixels[(i * m_width + j) * 4 + 2];
//                img.at<cv::Vec3b>(m_height - 1 - i, j)[2] = rv;
//                img.at<cv::Vec3b>(m_height - 1 - i, j)[1] = gv;
//                img.at<cv::Vec3b>(m_height - 1 - i, j)[0] = bv;
                if(rv!=0&&gv!=0&&bv!=0)
                {
                    int id;
                    color_decode(rv,gv,bv,id);
                    if(!checkd[id])
                    {
                        visibles[id] = true;
                        checkd[id] = true;
                    }
                }
            }
        }
}

void Render::initialOpenGL()
{
    m_surface.create();
    if(!m_surface.isValid())
            std::cout<<"QOffenScreenSurface is not valid"<<std::endl;
    m_context.setFormat(m_surface.requestedFormat());
    m_context.create();
    if(!m_context.isValid())
        std::cout<<"QOpenGLContext is not valid"<<std::endl;
    if(!m_context.makeCurrent(&m_surface))
        {
            std::cout<<"initial OpenGL fail"<<std::endl;
            return;
        }
    m_f = m_context.versionFunctions<QOpenGLFunctions_1_2>();
    if (!m_f) {
      qWarning() << "Could not obtain required OpenGL context version";
      exit(1);
    }
    m_fbo = NULL;
}

void Render::RenderBackground(cv::Mat &back)
{
    int width = back.cols;
    int height = back.rows;
    m_fbo->bind();
    m_f->glMatrixMode(GL_MODELVIEW);
    m_f->glLoadIdentity();
    m_f->glMatrixMode(GL_PROJECTION);
    m_f->glLoadIdentity();
    m_f->glOrtho(0.0,width,0.0,height,-1.0,1.0);
    m_f->glViewport(0,0,m_width,m_height);

    int tr, tg, tb;
    QVector<unsigned char> img_data;
    img_data.resize(width*height*4);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            tr=back.at<cv::Vec3b>(height - 1 - i, j)[2] ;
            tg=back.at<cv::Vec3b>(height - 1 - i, j)[1];
            tb=back.at<cv::Vec3b>(height - 1 - i, j)[0];
            img_data[(i * width + j) * 4] = tr;
            img_data[(i * width + j) * 4+1] = tg;
            img_data[(i * width + j) * 4+2] = tb;
            img_data[(i * width + j) * 4+3] = 255;
        }
    }

    m_f->glDisable(GL_DEPTH_TEST);
    m_f->glDisable(GL_CULL_FACE);
    m_f->glDisable(GL_LIGHTING);
    m_f->glEnable(GL_TEXTURE_2D);
    m_f->glColor3f(1,1,1);
    GLuint texture=make_texture(img_data.data(),width,height, GL_RGBA);
    m_context.functions()->glActiveTexture(GL_TEXTURE0);
    m_f->glEnable(GL_TEXTURE_2D);
    m_f->glBindTexture(GL_TEXTURE_2D, texture);

    float points[]={0,0,0,float(width),0,0,float(width),float(height),0,0,float(height),0};
    float texcoods[]={0,0,1,0,1,1,0,1};
    unsigned int indices[]={0,1,2,2,3,0};
    m_f->glEnableClientState(GL_VERTEX_ARRAY);
    m_f->glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    m_f->glVertexPointer(3,GL_FLOAT,0,points);
    m_f->glTexCoordPointer(2,GL_FLOAT,0,texcoods);
    m_f->glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, indices);
    m_f->glDisableClientState(GL_VERTEX_ARRAY);
    m_f->glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    m_f->glDisable(GL_TEXTURE_2D);
    m_f->glDeleteTextures(1,&texture);
    m_fbo->release();
}
void Render::initialFBO(QSize size)
{
    if (m_fbo)
    {
        m_fbo->release();
        delete m_fbo;
        m_fbo = nullptr;
    }
    m_width = size.width();
    m_height = size.height();
    QOpenGLFramebufferObjectFormat format;
    format.setSamples(0);
    format.setAttachment(QOpenGLFramebufferObject::Depth);
    m_fbo = new QOpenGLFramebufferObject(size, format);
    if (!m_fbo->isValid())
    {
        std::cout<<"initial fbo fail!"<<std::endl;
    }
    m_fbo->bind();
    m_f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_fbo->release();
    //    cout<<"size: "<<fbo->size().width()<<" "<<fbo->height()<<endl;
}

void Render::estimation_face_image_whole_depthmesh(TriMesh *mesh, Eigen::MatrixXf &pro, const Eigen::Matrix3f &R, const Eigen::Vector2f T, float scale,int width, int height, int anchor, std::vector<float> &depths)
{
    m_fbo->bind();
    m_f->glMatrixMode(GL_MODELVIEW);
    m_f->glLoadIdentity();

//    m_f->glOrtho(0.0,width,0.0,height,-zmax-1.0, -zmin+500.0);
    m_f->glViewport(0,0,width,height);
    TriMesh::VertexIter v_it = mesh->vertices_begin();
    float *addrV = const_cast<float*>(mesh->point(*v_it).data());
    Eigen::MatrixXf temp = Eigen::Map<Eigen::MatrixXf>(addrV,3,mesh->n_vertices());
    Eigen::MatrixXf points = temp;
    points = (scale*R*points).colwise()+Eigen::Vector3f(T(0),T(1),0.0);
    TriMesh::ConstFaceIter        f_it(mesh->faces_sbegin()),
                               f_end(mesh->faces_end());
    TriMesh::ConstFaceVertexIter  fv_it;
    std::vector<unsigned int> indices;
    indices.reserve(mesh->n_faces()*3);
    for (; f_it!=f_end; ++f_it)
    {
      indices.push_back((fv_it=mesh->cfv_iter(f_it)).handle().idx());
      indices.push_back((++fv_it).handle().idx());
      indices.push_back((++fv_it).handle().idx());
    }

    m_f->glEnable(GL_DEPTH_TEST);
    m_f->glEnable(GL_CULL_FACE);
    m_f->glDisable(GL_LIGHTING);
    m_f->glEnable(GL_SMOOTH);
    Eigen::Vector3f anchor_point = points.col(anchor);
    float ratio[3] = {1.3, 1.15, 1.0};
    depths.resize(width*height,1.0);
    for(int i=0; i<3; i++)
    {
        Eigen::MatrixXf inputs = (ratio[i]*(points.colwise()-anchor_point)).colwise()+anchor_point;
        if(i==0) //first time, set projection
        {
            float zmax = inputs.row(2).maxCoeff();
            float zmin = inputs.row(2).minCoeff();
            QMatrix4x4 proj;    //rotate, ortho is right product
            proj.setToIdentity();
            proj.ortho(0,width,0,height,-zmax-1.0, -zmin+1.0);
            pro.resize(4,4);
            pro = Eigen::Map<Eigen::MatrixXf>(proj.data(),4,4);
            m_f->glMatrixMode(GL_PROJECTION);
            m_f->glLoadIdentity();
            m_f->glLoadMatrixf(pro.data());
        }
        m_f->glEnableClientState(GL_VERTEX_ARRAY);
        m_f->glVertexPointer(3,GL_FLOAT,0,inputs.data());
        m_f->glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);
        m_context.extraFunctions()->glReadBuffer(GL_DEPTH_ATTACHMENT);
        std::vector<float> m_pixels;
        m_pixels.resize(width*height,1.0);
        m_f->glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, &m_pixels[0]);
        for(int num=0; num<m_pixels.size(); num++)
        {
            if(fabs(m_pixels[num]-1.0)>0.00001)
            {
                depths[num] = m_pixels[num];
            }
        }
        m_f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    m_f->glDisableClientState(GL_VERTEX_ARRAY);
    m_f->glDisable(GL_DEPTH_TEST);
    m_f->glDisable(GL_CULL_FACE);


}

GLuint Render::make_texture(uchar* img, int width, int height, int type)
{
    GLuint texture;
//    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, type, GL_UNSIGNED_BYTE, img);
//    glDisable(GL_TEXTURE_2D);
    return texture;
}

void Render::set_render_env(const BOX &box)
{
    m_f->glDisable(GL_DITHER);
    m_f->glShadeModel(GL_SMOOTH);
//    // material
//    GLfloat mat_a[] = {0.4, 0.4, 0.4, 1.0};
//    GLfloat mat_d[] = {0.4, 0.4, 0.4, 1.0};
//    GLfloat mat_s[] = {0.8, 0.8, 0.8, 1.0};
//    GLfloat shine[] = {128.0};
//    f->glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_a);
//    f->glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_d);
//    f->glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_s);
//    f->glMaterialfv(GL_FRONT, GL_SHININESS, shine);
    float xmin = box.x_min;     float xmax = box.x_max;     float xlen = box.x_len;
    float ymin = -box.y_min;    float ymax = -box.y_max;    float ylen = box.y_len;
    float zmin = -box.z_min;    float zmax = -box.z_max;    float z_len = box.z_len;


    GLfloat pos1[] = { 1.2*xlen, -ylen/2.0, zmin+z_len/2.0, 0.0};
    GLfloat pos2[] = {-0.2*xlen, -ylen/2.0, zmin+z_len/2.0, 0.0};
    GLfloat pos3[] = { xlen/2.0, -ylen/2.0, zmax, 0.0};
//    GLfloat pos1[] = { -0.1, -0.1, 0.02, 0.0};
//    GLfloat pos2[] = {0.1, -0.1, 0.02, 0.0};
//    GLfloat pos3[] = { 0.0, 0.0, -0.1, 0.0};
    GLfloat col1[] = {.4, .4, .4, 1.0};
    GLfloat col2[] = {.6, .6, .6, 1.0};
    GLfloat col3[] = {.5, .5, .5, 1.0};
    GLfloat col4[] = {.2, .2, .2, 1.0};

    m_f->glEnable(GL_LIGHT0);
    m_f->glLightfv(GL_LIGHT0,GL_POSITION, pos1);
    m_f->glLightfv(GL_LIGHT0,GL_DIFFUSE,  col4);
    m_f->glLightfv(GL_LIGHT0,GL_AMBIENT, col4);
//    m_f->glLightfv(GL_LIGHT0,GL_SPECULAR, col1);

    m_f->glEnable(GL_LIGHT1);
    m_f->glLightfv(GL_LIGHT1,GL_POSITION, pos2);
    m_f->glLightfv(GL_LIGHT1,GL_DIFFUSE,  col2);
//    m_f->glLightfv(GL_LIGHT1,GL_SPECULAR, col1);

    m_f->glEnable(GL_LIGHT2);
    m_f->glLightfv(GL_LIGHT2,GL_POSITION, pos3);
    m_f->glLightfv(GL_LIGHT2,GL_DIFFUSE,  col3);
    //    m_f->glLightfv(GL_LIGHT2,GL_SPECULAR, col2);
}

void Render::color_code(int N, std::vector<Eigen::Vector4i> &colors)
{
    colors.clear();
    int RN = N / 255 / 255;
    int GN = (N - RN * 255 * 255) / 255;
    int BN = N - RN * 255 * 255 - GN * 255;
    for (int i = 1; i <= RN; i++)
    {
        for (int j = 1; j <= 255; j++)
        {
            for (int k = 1; k <= 255; k++)
            {
                colors.push_back(Eigen::Vector4i(i,j,k,255));
            }
        }
    }

    for (int j = 1; j <= GN; j++)
    {
        for (int k = 1; k <= 255; k++)
        {
            colors.push_back(Eigen::Vector4i(RN+1,j,k,255));
        }
    }

    for (int k = 1; k <= BN; k++)
    {
        colors.push_back(Eigen::Vector4i(RN+1,GN+1,k,255));
    }
}

void Render::color_decode(const Eigen::Vector4i &color, int &id)
{
    color_decode(color(0),color(1),color(2),id);
}

void Render::color_decode(const int &r, const int &g, const int &b, int &id)
{
    if(r==0||g==0||b==0)
        id=-1;
    else
        id = (r-1)*255*255+(g-1)*255+b-1;
}
