#ifndef IO_UTILS_H
#define IO_UTILS_H
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Eigen/Dense>
namespace io_utils{
using namespace std;
using namespace Eigen;
//some read file, find id correspond operation
//pf2wf_file: part mesh corresponding whole mesh face id, part is index order(0,1,2), file is only one col
//wf2opf_file: whole mesh face to another part mesh face corres, whole mesh is index order, on face is -1
//out corres is part to another part face corres, part is index order, no corres is -1
void partfid2wholef2anotherpartfid(const string &pf2wf_file, const string &wf2opf_file, vector<int> &corres);
template<typename T>
void write_all_type_to_file(const vector<T> &ids, const string &file)
{
    ofstream out(file);
    for(int i=0; i<ids.size(); i++)
        out<<ids[i]<<endl;
    out.close();
}

template<typename T>    //this write method is most roboust, can ref read_all_type_rowsfile_to_2vector's notes
void read_all_type_file_to_vector(const string &file, vector<T> &ids)
{
    ids.clear();
    ifstream in(file.data());
    while(true)
    {
        T id;
        in>>id;
        if(!in.fail())
            ids.push_back(id);
        else
            break;
    }
    in.close();
}

template<typename T>
void write_all_type_to_rowsfile(const vector<vector<T> > &ids, const string &name)
{
    std::ofstream file(name.data());
    for(int i=0; i<ids.size(); i++)
    {
        std::vector<T> temp = ids[i];
        for(int j=0;j<temp.size();j++)
            file<<temp[j]<<" ";
        file<<"\n";
    }
    file.close();
}

template<typename T>
void read_all_type_rowsfile_to_2vector(const string &file, vector<vector<T> > &ids)
{
    ifstream f(file.data());
    vector<std::string> lines;
    string line;
    //getline first reach EOF, f.eof is false, next time use getline, f.eof is true, maybe use f.getline can be different
    //so first do one time getline, then while
    //up's solution is not perfect, I find some case, coped file have different result, one less 1 row, very strange
    //use fail to make it more robust, solved up's problem
    while (!f.eof())
    {
        getline(f, line);
        if(!f.fail())
            lines.push_back(line);
    }
    f.close();
    ids.clear();
    vector<T> tmp;
    for (int i = 0; i < lines.size(); i++)
    {
        istringstream in(lines[i]);
        T a;
        //format io(>>,<<), first reach EOF, .eof is true. fail check >>/<< is or not success, should use fail instead of eof
        //format io use blank char to seperate, so if after last number is EOF, use eof to check can make last number
        // to be not push_back or be push_back 2 times, it depends on >> and eof position.
        //this can also not read blank row
        while(true)
        {
            in>>a;
            if(in.fail())
                break;
            else
                tmp.push_back(a);

        }
        if(tmp.size()>0)
            ids.push_back(tmp);
        tmp.clear();
    }
//    cout<<file<<" has "<<ids.size()<<" rows."<<endl;
}
template<typename T>
void write_all_type_to_bin(const vector<T> &ids, const string &file, bool write_size=false)
{
    FILE *f=fopen(file.data(),"wb");
    if(f==NULL)
    {
        std::cout<<"open file fail!"<<std::endl;
        return ;
    }
    if(write_size)
    {
        int size=ids.size();
        fwrite(&size, sizeof(int),1,f);
    }
    fwrite(ids.data(),sizeof(T),ids.size(),f);
    fclose(f);
}
template<typename T>
void read_all_type_from_bin(const string &file, int size, vector<T> &ids)
{
    FILE* f=fopen(file.data(), "rb");
    if(f==NULL)
    {
        std::cout<<"open file fail!"<<std::endl;
        return ;
    }
    ids.resize(size,T(0));
    fread(ids.data(), sizeof(T), size, f);
    fclose(f);
}

//some io function for my 3dmm data
void read_pca_models(const string &name, MatrixXf &mean, MatrixXf &pca, MatrixXf&st, int &v_num, int &pc_num);
}
#endif // IO_UTILS_H
