#include "make_data_for_multi_classlearn.h"
#include "io_utils.h"
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <QDir>
#include <QStringList>
#include <stdint.h>
make_data_for_multi_classlearn::make_data_for_multi_classlearn()
{
    m_render_to_label=false;
    io_utils::read_pca_models("../Data/mainShapePCA.bin",m_mean_shape,m_pca_shape,m_shape_st,
                              m_vnum,m_shape_pcanum);
    io_utils::read_pca_models("../Data/DeltaExpPCA.bin",m_mean_exp,m_pca_exp,m_exp_st,
                              m_vnum,m_exp_pcanum);
    io_utils::read_all_type_file_to_vector<int>("../Data/partv23dmmv.txt",m_partv_2_wholev);
    io_utils::read_all_type_file_to_vector<int>("../Data/part_face_keypoints.txt",m_keypoints_id);
    OpenMesh::IO::read_mesh(m_mesh,"../Data/part_face.obj");
//    m_mesh.request_vertex_normals();
//    m_mesh.request_face_normals();
    m_mesh.request_face_colors();
    m_mesh.request_vertex_colors();
    compute_predefined_colors();
}

void make_data_for_multi_classlearn::make_data(const std::string &root, const std::string &save_root)
{
    std::string segdata_root="../part_face_seg_result/";
    QDir save_dir(QString(save_root.data()));
    int num_seg=100;
    int accu_seg_num_to_pause=15;
    int accu_seg_num=0;
    for(int i=0;i<num_seg+1;i++)  //100 segmentation and 1 keypoints classification
    {

        if(accu_seg_num==accu_seg_num_to_pause)
        {
            std::cout<<"reach pause segmentation num, "<<"segmentation before "<<i<<" has been computed, "<<"please compress data and upload to serve, and clean local data to save space!"<<std::endl;
            std::cout<<"after clean data, input continue to continue."<<std::endl;

            std::string input;
            std::cin>>input;
            while(input!="continue")
            {
                std::cout<<"inpurt continue to continue! "<<std::endl;
                std::cin>>input;
            }
            std::cout<<"continue..."<<std::endl;
            accu_seg_num=0;
        }


        save_dir.mkdir(QString((std::to_string(i)).data()));
        std::string patchfile = segdata_root+"part_face_segs_"+std::to_string(i)+".txt";
        std::string neighfile = segdata_root+"part_face_seg_neighbors_"+std::to_string(i)+".txt";
        read_patches_and_neighbors(patchfile,neighfile);
        if(i<num_seg)
        {
            if(!m_render_to_label)
                set_patches_colors();
            else
                code_patches_colors();
        }
        else
        {
            if(!m_render_to_label)
                set_keypoints_colors();
            else
                code_keypoints_colors();
        }
        QDir path(QString(root.data()));
        path.setFilter(QDir::Files);
        QStringList filters={"*_mesh_para.txt"};
        path.setNameFilters(filters);
        path.setSorting(QDir::Name);
        QStringList entrys = path.entryList();
        int num=0;
        for(QStringList::Iterator vit = entrys.begin(); vit!=entrys.end(); vit++)
        {
            std::string mesh_name = (*vit).toStdString();
            QStringList filters2;
            filters2.push_back(QString((mesh_name.substr(0,mesh_name.size()-14)+"_*.jpg").data()));
            path.setNameFilters(filters2);
            QStringList entrys2 = path.entryList();
            for(QStringList::Iterator sit = entrys2.begin(); sit!=entrys2.end(); sit++)
            {
                std::string img_name = (*sit).toStdString();
                std::string pose_name = img_name.substr(0,img_name.size()-4)+"_pose.txt";
                read_mesh_para(root+mesh_name);
                read_pose_para(root+pose_name);
                read_img(root+img_name);
                update_mesh();
                cv::Mat result;
                if(i<num_seg)
                {
                        render_patches_on_img(result);
                }
                else
                {
                        render_keypoints_on_img(result);
                }
                if(!m_render_to_label)
                    cv::imwrite(save_root+std::to_string(i)+"/"+img_name,result);
                else
                {
                    decode_img_to_compression_label_bin(result,
                                                        save_root+std::to_string(i)+"/"+img_name.substr(0,img_name.size()-4)+"_label.bin");
                }
            }
//            if(num>10)
//                break;
            std::cout<<std::to_string(i)<<" seg, "<<std::to_string(num)<<" person done!"<<std::endl;
            num++;
        }
        accu_seg_num++;
    }
}

void make_data_for_multi_classlearn::render_patches_on_img(cv::Mat &result)
{
    m_render.initialFBO(QSize(m_img.cols,m_img.rows));
    if(!m_render_to_label)
        m_render.RenderBackground(m_img);
    m_render.setModelViewMatrix(m_scale*m_R, m_weak_T(0), m_weak_T(1), 0.0);
    m_render.setOrtho(0,m_img.cols,0,m_img.rows,-500,500);
    m_render.setViewPort(0,0,m_img.cols,m_img.rows);
    m_render.RenderPatchMesh(&m_mesh,result);
    //    m_render.RenderImage(&m_mesh,result);
}

void make_data_for_multi_classlearn::render_keypoints_on_img(cv::Mat &result)
{
    m_render.initialFBO(QSize(m_img.cols,m_img.rows));
//    m_render.RenderBackground(m_img);
    m_render.setModelViewMatrix(m_scale*m_R, m_weak_T(0), m_weak_T(1), 0.0);
    m_render.setOrtho(0,m_img.cols,0,m_img.rows,-500,500);
    m_render.setViewPort(0,0,m_img.cols,m_img.rows);
    m_render.RenderVisibleKeyPoints(&m_mesh,m_keypoints_id,result);
}

void make_data_for_multi_classlearn::read_patches_and_neighbors(const std::string &file0, const std::string &file1)
{
    io_utils::read_all_type_rowsfile_to_2vector<int>(file0,m_patches);
    io_utils::read_all_type_rowsfile_to_2vector<int>(file1, m_patch_neighbors);
}

void make_data_for_multi_classlearn::read_mesh_para(const std::string &file)
{
    std::vector<std::vector<float> > paras;
    io_utils::read_all_type_rowsfile_to_2vector<float>(file, paras);
    m_shape.resize(m_shape_pcanum);
    m_shape.setZero();
    int size=paras[0].size();
    if(m_shape.size()<size) size=m_shape.size();
    memcpy(m_shape.data(),paras[0].data(),size);
    m_exp.resize(m_exp_pcanum);
    m_exp.setZero();
    size = paras[1].size();
    if(m_exp.size()<size)   size=m_exp.size();
    memcpy(m_exp.data(),paras[1].data(),size);
}

void make_data_for_multi_classlearn::read_pose_para(const std::string &file)
{
    std::vector<float> paras;
    io_utils::read_all_type_file_to_vector<float>(file, paras);
    Eigen::Affine3f transformation;
    transformation  = Eigen::AngleAxisf(-paras[0], Eigen::Vector3f::UnitX()) *
                      Eigen::AngleAxisf(-paras[1], Eigen::Vector3f::UnitY()) *
                      Eigen::AngleAxisf(-paras[2], Eigen::Vector3f::UnitZ());
    m_R = transformation.rotation();
    m_weak_T(0)=paras[3];
    m_weak_T(1)=paras[4];
    m_scale=paras[5];
}

void make_data_for_multi_classlearn::read_img(const std::string &file)
{
    m_img=cv::imread(file);
}

void make_data_for_multi_classlearn::update_mesh()
{
    VectorXf verts = m_mean_exp+m_mean_shape+m_pca_shape*m_shape+m_pca_exp*m_exp;
    for(TriMesh::VertexIter vit=m_mesh.vertices_begin(); vit!=m_mesh.vertices_end(); vit++)
    {
        int pid = (*vit).idx();
        int wid = m_partv_2_wholev[pid];
        m_mesh.set_point(*vit, TriMesh::Point(verts(3*wid), verts(3*wid+1), verts(3*wid+2)));
    }
//    m_mesh.update_normals();
//    OpenMesh::IO::write_mesh(m_mesh,"../test_result.obj");
}
void make_data_for_multi_classlearn::compute_predefined_colors()
{
    unsigned int colors[20]={0xFF0000A6, 0xFF63FFAC,0xFF6B7900, 0xFF00C2A0,0xFFA1C299, 0xFF300018,0xFF6F0062, 0xFF0CBD66,
                       0xFF456648, 0xFF0086ED,0xFF636375, 0xFFA3C8C9,0xFF6367A9, 0xFFA05837,0xFF549E79, 0xFFFFF69F,
                       0xFF83AB58, 0xFF001C1E,0xFFD0AC94, 0xFF7ED379};
    m_predefine_colors.clear();
    for(int i=0;i<20;i++)
    {
        unsigned int temp=colors[i];
        unsigned char red = ((temp >> 16) & 0xff);
        unsigned char green = ((temp >> 8) & 0xff);
        unsigned char blue = (temp & 0xff);
        TriMesh::Color color(float(red)/255.0,float(green)/255.0,float(blue)/255.0);
        m_predefine_colors.push_back(color);
    }
}
void make_data_for_multi_classlearn::set_patches_colors()
{
//    m_mesh.request_face_colors();
    std::map<int,int> patch2colorid;
    for(int i=0;i<m_patches.size();i++)
        patch2colorid[i] = -1;
    base_generator_type gen(time(0));
    for(int i=0; i<m_patches.size(); i++)
    {
        const std::vector<int> &temp = m_patches[i];
        const std::vector<int> &neighbors = m_patch_neighbors[i];
        std::set<int> used_colors;
        for(int j=0; j<neighbors.size(); j++)
        {
            std::map<int,int>::iterator itr=patch2colorid.find(neighbors[j]);
            if(itr->second!=-1)
                used_colors.insert(itr->second);
        }
        int color_id=-1;
        random_get_patch_color(used_colors, color_id, gen);
        for(int j=0;j<temp.size();j++)
        {
            m_mesh.set_color(TriMesh::FaceHandle(temp[j]), m_predefine_colors[color_id]);
        }
        patch2colorid.at(i) = color_id;
    }
}

void make_data_for_multi_classlearn::code_patches_colors()
{
     std::vector<Vector4i> colors;
    Render::color_code(m_patches.size(), colors);
    for(int i=0; i<colors.size(); i++)
    {
        const std::vector<int> &temp = m_patches[i];
        TriMesh::Color color(float(colors[i](0))/255.0,float(colors[i](1))/255.0,float(colors[i](2))/255.0);
        for(int j=0;j<temp.size();j++)
        {
            m_mesh.set_color(TriMesh::FaceHandle(temp[j]), color);
        }
    }
}

void make_data_for_multi_classlearn::set_keypoints_colors()
{
    for(TriMesh::VertexIter vit=m_mesh.vertices_begin(); vit!=m_mesh.vertices_end(); vit++)
        m_mesh.set_color(*vit, TriMesh::Color(0.0,0.0,0.0));
    for(int i=0;i<m_keypoints_id.size();i++)
        m_mesh.set_color(TriMesh::VertexHandle(m_keypoints_id[i]),TriMesh::Color(1.0,0.0,0.0));
}

void make_data_for_multi_classlearn::code_keypoints_colors()
{
    for(TriMesh::VertexIter vit=m_mesh.vertices_begin(); vit!=m_mesh.vertices_end(); vit++)
        m_mesh.set_color(*vit, TriMesh::Color(0.0,0.0,0.0));
    std::vector<Vector4i> colors;
    Render::color_code(m_keypoints_id.size(), colors);
    for(int i=0;i<colors.size();i++)
    {
        TriMesh::Color color(float(colors[i](0))/255.0,float(colors[i](1))/255.0,float(colors[i](2))/255.0);
        m_mesh.set_color(TriMesh::VertexHandle(m_keypoints_id[i]),color);
    }
}

void make_data_for_multi_classlearn::decode_img_to_compression_label_bin(const cv::Mat &img, const std::string &out)
{
    //row order save
    if(!img.isContinuous())
        std::cout<<"decode_img_to_label: img is not continuous!!!"<<std::endl;
    uchar* data=img.data;

    uint16_t minx=10000;
    uint16_t maxx=0;
    uint16_t miny=10000;
    uint16_t maxy=0;
    int colnum=img.cols;
    int rownum=img.rows;
    std::vector<uint16_t>   temp;
    for(int i=0; i<rownum; i++)
    {
        for(int j=0;j<colnum;j++)
        {
            int b=int(data[3*(i*colnum+j)]);
            int g=int(data[3*(i*colnum+j)+1]);
            int r=int(data[3*(i*colnum+j)+2]);
            int id;
            Render::color_decode(r,g,b,id);
            if(id!=-1)
            {
                if(j<minx)  minx=j;
                if(j>maxx)  maxx=j;
                if(i<miny)  miny=i;
                if(i>maxy)  maxy=i;
            }
            else
                id=500;     //this data, background's label is 500
            temp.push_back(uint16_t(id));
        }
    }
    std::vector<uint16_t>   result;
    result.push_back(uint16_t(rownum));    result.push_back(uint16_t(colnum));
    result.push_back(uint16_t(minx));   result.push_back(uint16_t(miny));
    result.push_back(uint16_t(maxx));   result.push_back(uint16_t(maxy));
    for(int i=miny; i<=maxy; i++)
    {
        for(int j=minx; j<=maxx; j++)
        {
            result.push_back(temp[i*colnum+j]);
        }
    }
    io_utils::write_all_type_to_bin<uint16_t>(result,out);
}

void make_data_for_multi_classlearn::decode_compression_label_bin_to_img_label(const std::string &in, std::vector<uint16_t> &label, int &rows, int &cols)
{
    FILE *file = fopen(in.data(),"rb");
    uint16_t data[6];
    fread(data, sizeof(uint16_t), 6, file);
    //data is: rows 0, cols 1, minx 2, miny 3, maxx 4, maxy 5
    std::vector<uint16_t> temp;
    temp.resize((data[4]-data[2]+1)*(data[5]-data[3]+1));
    fread(temp.data(), sizeof(uint16_t), temp.size(), file);
    fclose(file);
    rows = data[0];
    cols = data[1];
    label.resize(rows*cols,uint16_t(500));
    int index=0;
    for(int i=data[3]; i<=data[5]; i++)
    {
        for(int j=data[2]; j<=data[4]; j++)
        {
            label[i*cols+j] = temp[index];
            index++;
        }
    }
}
void make_data_for_multi_classlearn::random_get_patch_color(const std::set<int> &exclude_color_ids, int &choosed_id, base_generator_type&gen)
{
    if(exclude_color_ids.size()>=m_predefine_colors.size())
    {
        std::cout<<"predefine colors are to little to color neighbors!!!"<<std::endl;
        choosed_id = 0;
        return;
    }
    std::vector<int> chooseable_id;
    for(int i=0;i<m_predefine_colors.size();i++)
    {
        if(!exclude_color_ids.count(i))
        {
            chooseable_id.push_back(i);
        }
    }
    //using uniform_int_distribution will become ambigious, this is very strange, because other project can work, but here can not work!!!
    boost::variate_generator<base_generator_type&, uniform_int_distribution_type > random_int(gen, uniform_int_distribution_type(0,chooseable_id.size()-1));
    choosed_id = chooseable_id[random_int()];
}
