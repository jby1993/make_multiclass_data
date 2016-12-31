#ifndef TRIMESH_TYPES_HH
#define TRIMESH_TYPES_HH


//== INCLUDES =================================================================
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
// #include <ACG/Scenegraph/StatusNodesT.hh>


struct BOX
{
    float x_min,x_max;
    float y_min,y_max;
    float z_min,z_max;
    float x_len;
    float y_len;
    float z_len;
};

//== TYPEDEFS =================================================================

/** Default traits for the TriMesh
*/
struct TriTraits : public OpenMesh::DefaultTraits
{
  /// Use double precision points
  typedef OpenMesh::Vec3f Point;
  /// Use double precision Normals
  typedef OpenMesh::Vec3f Normal;
//  /// Use double precision TexCood2D
//  typedef OpenMesh::Vec2d TexCoord2D;

  /// Use RGB Color
  typedef OpenMesh::Vec3f Color;
};

/// Simple Name for Mesh
typedef OpenMesh::TriMesh_ArrayKernelT<TriTraits>  TriMesh;

//== TYPEDEFS FOR SCENEGRAPH ===============================================


//=============================================================================
#endif // TRIMESHMESH_TYPES_HH defined
//=============================================================================
