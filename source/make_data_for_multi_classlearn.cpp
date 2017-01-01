#include "make_data_for_multi_classlearn.h"
#include "io_utils.h"
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <QDir>
#include <QStringList>
#include <stdint.h>
#include <omp.h>
#include <bits/basic_string.h>
make_data_for_multi_classlearn::make_data_for_multi_classlearn()
{
    m_thread_num = 4;
    m_start_segnum=0;
    m_render_to_label=false;
    io_utils::read_pca_models("../Data/mainShapePCA.bin",m_mean_shape,m_pca_shape,m_shape_st,
                              m_vnum,m_shape_pcanum);
    io_utils::read_pca_models("../Data/DeltaExpPCA.bin",m_mean_exp,m_pca_exp,m_exp_st,
                              m_vnum,m_exp_pcanum);
    io_utils::read_all_type_file_to_vector<int>("../Data/partv23dmmv.txt",m_partv_2_wholev);
    io_utils::read_all_type_file_to_vector<int>("../Data/part_face_keypoints.txt",m_keypoints_id);    
    compute_predefined_colors();
}

void make_data_for_multi_classlearn::make_data(const std::string &root, const std::string &save_root)
{
    std::cout<<"start make data; threads num is "<<m_thread_num<<std::endl;
    set_threads(m_thread_num);
    std::string segdata_root="../part_face_seg_result/";
    QDir save_dir(QString(save_root.data()));
    int num_seg=100;
    int accu_seg_num_to_pause=num_seg;
    int accu_seg_num=0;
    std::vector<std::string> mesh_files;
//    get_meshpara_names(root, mesh_files);
    io_utils::read_all_type_file_to_vector<string>("../Data/mesh_para_filenames.txt",mesh_files);
    std::vector<std::vector<std::string > > per_imgfiles;
//    get_permesh_imgnames(root, mesh_files, per_imgfiles);
    io_utils::read_all_type_rowsfile_to_2vector<string>("../Data/permesh_imgfiles.txt",per_imgfiles);
    for(int k=m_start_segnum;k<num_seg;k++)  //myrender do not make keypoints result
    {
//program pause
        if(accu_seg_num==accu_seg_num_to_pause)
        {
            std::cout<<"reach pause segmentation num, "<<"segmentation before "<<k<<" has been computed, "<<"please compress data and upload to serve, and clean local data to save space!"<<std::endl;
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
        QString temp;   temp.setNum(k);
        std::string numk=temp.toStdString();
        save_dir.mkdir(temp);
        std::string patchfile = segdata_root+"part_face_segs_"+numk+".txt";
        std::string neighfile = segdata_root+"part_face_seg_neighbors_"+numk+".txt";
        read_patches_and_neighbors(patchfile,neighfile);
        std::vector<TriMesh::Color> patchcolors;
        if(!m_render_to_label)
            set_patches_colors_for_multithread(patchcolors);
        else
            code_patches_colors_formultithread(patchcolors);
//        int num=0;
        #pragma omp parallel for num_threads(m_thread_num)
        for(int i=0; i<mesh_files.size(); i++)
        {            
            QString itemp;  itemp.setNum(i);
            std::string numi = itemp.toStdString();
            std::cout<<numk<<" seg, "<<numi<<" person start!"<<std::endl;
            std::string mesh_name = mesh_files[i];
            const std::vector<std::string>  &temp = per_imgfiles[i];
            int thread_id = omp_get_thread_num();
            read_mesh_para_thread(root+mesh_name, thread_id);
            update_local_mesh(m_meshs[thread_id],thread_id);
            color_patch_mesh(m_meshs[thread_id],patchcolors);
            for(int j=0; j<temp.size(); j++)
            {
                std::string img_name = temp[j];
                std::string pose_name = img_name.substr(0,img_name.size()-4)+"_pose.txt";
                Matrix3f R; Vector2f T; float scale;
                read_pose_to_local(root+pose_name,R,T,scale);
                cv::Mat img;
                read_img_to_local(root+img_name,img);
                cv::Mat result;
                render_patches_on_img_for_multithread(result,img,m_meshs[thread_id],R,T,scale,thread_id);
                if(!m_render_to_label)
                    cv::imwrite(save_root+numk+"/"+img_name,result);
                else
                {
                    decode_img_to_compression_label_bin(result,
                                                        save_root+numk+"/"+img_name.substr(0,img_name.size()-4)+"_label.bin");
                }
            }
//            if(num>10)
//                break;
//            QString itemp;  itemp.setNum(i);
//            std::string numi = itemp.toStdString();
//            std::cout<<numk<<" seg, "<<numi<<" person done!"<<std::endl;
//            num++;
        }
        accu_seg_num++;
    }
}


void make_data_for_multi_classlearn::render_patches_on_img_for_multithread(cv::Mat &result, const cv::Mat &img, TriMesh &mesh,const Matrix3f &R, const Vector2f &T, const float &scale, int thread_id)
{
    m_renders[thread_id].initialSize(img.cols,img.rows);
    m_renders[thread_id].setWeakPerspecPara(R, T(0), T(1), 0.0, scale);
    m_renders[thread_id].RenderPatchMesh(&mesh,result);
}

void make_data_for_multi_classlearn::read_patches_and_neighbors(const std::string &file0, const std::string &file1)
{
    io_utils::read_all_type_rowsfile_to_2vector<int>(file0,m_patches);
    io_utils::read_all_type_rowsfile_to_2vector<int>(file1, m_patch_neighbors);
}


void make_data_for_multi_classlearn::read_mesh_para_thread(const string &file, int thread_num)
{
    //memcpy zhuyi, zui hou de size shi byte shu
    std::vector<std::vector<float> > paras;
    io_utils::read_all_type_rowsfile_to_2vector<float>(file, paras);
    m_shapes[thread_num].resize(m_shape_pcanum);
    m_shapes[thread_num].setZero();
    int size=paras[0].size();
    if(m_shapes[thread_num].size()<size) size=m_shapes[thread_num].size();
    memcpy(m_shapes[thread_num].data(),paras[0].data(),size*sizeof(float));
    m_exps[thread_num].resize(m_exp_pcanum);
    m_exps[thread_num].setZero();
    size = paras[1].size();
    if(m_exps[thread_num].size()<size)   size=m_exps[thread_num].size();
    memcpy(m_exps[thread_num].data(),paras[1].data(),size*sizeof(float));
}


void make_data_for_multi_classlearn::read_pose_to_local(const string &file, Eigen::Matrix3f &R, Eigen::Vector2f &weak_T, float &scale)
{
    std::vector<float> paras;
    io_utils::read_all_type_file_to_vector<float>(file, paras);
    Eigen::Affine3f transformation;
    transformation  = Eigen::AngleAxisf(-paras[0], Eigen::Vector3f::UnitX()) *
                      Eigen::AngleAxisf(-paras[1], Eigen::Vector3f::UnitY()) *
                      Eigen::AngleAxisf(-paras[2], Eigen::Vector3f::UnitZ());
    R = transformation.rotation();
    weak_T(0)=paras[3];
    weak_T(1)=paras[4];
    scale=paras[5];
}

void make_data_for_multi_classlearn::read_img_to_local(const string &file, cv::Mat &img)
{
    img=cv::imread(file);
}


void make_data_for_multi_classlearn::update_local_mesh(TriMesh &mesh, int thread_id)
{
    VectorXf verts = m_mean_exp+m_mean_shape+m_pca_shape*m_shapes[thread_id]+m_pca_exp*m_exps[thread_id];
    for(TriMesh::VertexIter vit=mesh.vertices_begin(); vit!=mesh.vertices_end(); vit++)
    {
        int pid = (*vit).idx();
        int wid = m_partv_2_wholev[pid];
        mesh.set_point(*vit, TriMesh::Point(verts(3*wid), verts(3*wid+1), verts(3*wid+2)));
    }
    mesh.update_normals();
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

void make_data_for_multi_classlearn::set_patches_colors_for_multithread(std::vector<TriMesh::Color> &colors)
{
        std::map<int,int> patch2colorid;
        for(int i=0;i<m_patches.size();i++)
            patch2colorid[i] = -1;
        colors.clear();
//        base_generator_type gen(time(0));
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
            random_get_patch_color(used_colors, color_id/*, gen*/);
            colors.push_back(m_predefine_colors[color_id]);
            patch2colorid.at(i) = color_id;
        }
}


void make_data_for_multi_classlearn::code_patches_colors_formultithread(std::vector<TriMesh::Color> &colors)
{
    std::vector<Vector4i> tcolors;
   myRender::color_code(m_patches.size(), tcolors);
   colors.clear();
   for(int i=0; i<tcolors.size(); i++)
   {
//       const std::vector<int> &temp = m_patches[i];
       TriMesh::Color color(float(tcolors[i](0))/255.0,float(tcolors[i](1))/255.0,float(tcolors[i](2))/255.0);
        colors.push_back(color);
   }
}



void make_data_for_multi_classlearn::color_patch_mesh(TriMesh &mesh,const vector<TriMesh::Color> &colors)
{
    for(int i=0; i<colors.size(); i++)
    {
        const std::vector<int> &temp = m_patches[i];
        TriMesh::Color color = colors[i];
        for(int j=0;j<temp.size();j++)
        {
            mesh.set_color(TriMesh::FaceHandle(temp[j]), color);
        }
    }
}

void make_data_for_multi_classlearn::get_meshpara_names(const string &root, std::vector<string> &names)
{
    QDir path(QString(root.data()));
    path.setFilter(QDir::Files);
    QStringList filters;
    filters.push_back("*_mesh_para.txt");
    path.setNameFilters(filters);
    path.setSorting(QDir::Name);
    QStringList entrys = path.entryList();
    names.clear();
    for(QStringList::Iterator vit = entrys.begin(); vit!=entrys.end(); vit++)
    {
        names.push_back((*vit).toStdString());
    }
}

void make_data_for_multi_classlearn::get_permesh_imgnames(const string &root, const std::vector<string> &meshfiles, std::vector<std::vector<string> > &names)
{
    QDir path(QString(root.data()));
    names.clear();
    names.resize(meshfiles.size(), std::vector<std::string>());
    for(int i=0;i<meshfiles.size(); i++)
    {
        std::vector<std::string> &temp=names[i];
        std::string mesh_name = meshfiles[i];
        QStringList filters;
        filters.push_back(QString((mesh_name.substr(0,mesh_name.size()-14)+"_*.jpg").data()));
        path.setNameFilters(filters);
        QStringList entrys = path.entryList();
        for(QStringList::Iterator sit = entrys.begin(); sit!=entrys.end(); sit++)
        {
            temp.push_back((*sit).toStdString());
        }
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
            myRender::color_decode(r,g,b,id);
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
void make_data_for_multi_classlearn::random_get_patch_color(const std::set<int> &exclude_color_ids, int &choosed_id/*, base_generator_type&gen*/)
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
//    boost::variate_generator<base_generator_type&, uniform_int_distribution_type > random_int(gen, uniform_int_distribution_type(0,chooseable_id.size()-1));
    choosed_id = chooseable_id[chooseable_id.size()/2];
}

void make_data_for_multi_classlearn::set_threads(int thread_num)
{
    omp_set_num_threads(thread_num);
    m_shapes.resize(thread_num, VectorXf());
    m_exps.resize(thread_num, VectorXf());
    m_renders.resize(thread_num, myRender());
    for(int i=0; i<thread_num; i++)
    {
        TriMesh temp;
        OpenMesh::IO::read_mesh(temp,"../Data/part_face.obj");
        temp.request_vertex_normals();
        temp.request_face_normals();
        temp.request_face_colors();
        temp.request_vertex_colors();
        m_meshs.push_back(temp);
    }
}
