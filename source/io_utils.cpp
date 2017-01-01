#include "io_utils.h"
namespace io_utils{
void partfid2wholef2anotherpartfid(const std::string &pf2wf_file, const std::string &wf2opf_file, vector<int> &corres)
{
    vector<int> list1;
    ifstream file(pf2wf_file.data());
    do
    {
       int id;
       file>>id;
       list1.push_back(id);
    }while(!file.eof());
    file.close();
    vector<int> list2;
    file.open(wf2opf_file.data());
    do
    {
        int id;
        file>>id;
        list2.push_back(id);
    }while(!file.eof());
    file.close();
    corres.clear();
    bool check=false;
    for(int i=0; i<list1.size(); i++)
    {
        int id=list2[list1[i]];
        corres.push_back(id);
        if(id==-1)  check=true;
    }
    if(check)
        cout<<"io_utils::partfid2wholef2anotherpartfid: part has some faces that another part hasn't!"<<endl;
}
void read_pca_models(const string &name, MatrixXf &mean, MatrixXf &pca, MatrixXf &st, int &v_num, int &pc_num)
{
    FILE *file = fopen(name.data(),"rb");
    fread(&v_num,sizeof(int), 1, file);
    fread(&pc_num,sizeof(int),1,file);
    mean.resize(3*v_num,1);
    pca.resize(3*v_num,pc_num);
    st.resize(pc_num,1);

    fread(mean.data(),sizeof(float),mean.size(),file);
    fread(pca.data(),sizeof(float),pca.size(),file);
    fread(st.data(),sizeof(float),st.size(),file);
    fclose(file);
}

}

