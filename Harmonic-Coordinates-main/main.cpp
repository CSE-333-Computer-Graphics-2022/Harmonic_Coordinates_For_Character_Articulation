// #include <igl/opengl/glfw/Viewer.h>

// int main(int argc, char *argv[])
// {
//   // Inline mesh of a cube
//   MatrixXd V= (MatrixXd(8,3)<<
//     0.0,0.0,0.0,
//     0.0,0.0,1.0,
//     0.0,1.0,0.0,
//     0.0,1.0,1.0,
//     1.0,0.0,0.0,
//     1.0,0.0,1.0,
//     1.0,1.0,0.0,
//     1.0,1.0,1.0).finished();
//   MatrixXi F = (MatrixXi(12,3)<<
//     0,6,4,
//     0,2,6,
//     0,3,2,
//     0,1,3,
//     2,7,6,
//     2,3,7,
//     4,6,7,
//     4,7,5,
//     0,4,5,
//     0,5,1,
//     1,5,7,
//     1,7,3).finished();

//   // Plot the mesh
//   Viewer viewer;
//   viewer.data().set_mesh(V, F);
//   viewer.data().set_face_based(true);
//   viewer.launch();
// }

#include <iostream>
#include <igl/boundary_loop.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/slice.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/unproject_on_plane.h>
#include <igl/slice_into.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <float.h>
#include <math.h>
using namespace std;
using namespace igl::opengl::glfw;
using namespace Eigen;

//model and cage vertices and faces
MatrixXd V,main_vertices,Vertices_cage;;
MatrixXi F,Faces_cage;

MatrixXi Ht;

double h = 7; 
double offsetX = -50; 
double offsetY = 430; 
// weights
MatrixXd cage_weights;
MatrixXd weight;
// interior mesh
MatrixXd original_Vi;
MatrixXd control_vertices;
MatrixXi control_faces;


int current_cage_index = 0;
// used for articulation
VectorXi cage_vertices;
Vector2d prev_mouse;
MatrixXi E(4,2);


int choosen_control_vertex;
int model_mesh, model_mesh_wei;
unsigned int screen;
double min_dist;
VectorXi rest_verties;

enum cell_Type{UNTYPED,INTERIOR,EXTERIOR, BOUNDARY,};
int interpolationPrecision = 40; // point on edges to detect cells
int grid_size = 12; //based on paper, s controls size of grid 2^s
int check=0;

MatrixXd Cage; //Cage vertices
MatrixXi CageEdgesIndices; 
MatrixXd Ec; //Cage edges
MatrixXd Cage_ed ;

int control_vertex(Vector3d &click_point, bool original_cage)
{
  RowVector2d click_point_2d(click_point(0), click_point(1));
  MatrixXd cage;
  MatrixXd inter; 
  if(original_cage)
  {
  cage=main_vertices;
  inter=original_Vi;
  }
  else
  {
  cage=Vertices_cage;
  inter=control_vertices;
  }

  int interior_index;
  double interior_dist;
  int cage_index;

  double cage_dist = (cage.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&cage_index);
  if(inter.rows()>0){ 
  interior_dist = (inter.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&interior_index);
  }
  else
  {
    interior_dist= DBL_MAX;
  }
  double dist =min(cage_dist, interior_dist);
  if (dist > min_dist/3)
  {
    return -1;
  }
  int index;
  if(cage_dist < interior_dist)
  { 
  index = cage_index;
  } 
  else{
    index =cage.rows() + interior_index;
  }
  return index;
}
int Flag = 0;

bool mouse_down(Viewer& viewer, int button, int modifier)
{
  Vector3d Zaxis;
  if (button == (int) Viewer::MouseButton::Right)
  {
    return false;
  }
  bool Select = viewer.current_mouse_x < viewer.core(screen).viewport(2);

    //libigl tut 708
  Vector4d d1=Vector4d(0, 0, 1, 0);
  igl::unproject_on_plane(Vector2i(viewer.current_mouse_x, viewer.core(screen).viewport(3) - viewer.current_mouse_y),viewer.core(screen).proj * viewer.core(screen).view,viewer.core(screen).viewport,d1,Zaxis);
  int mouse_idx = control_vertex(Zaxis, !Select);
  if (mouse_idx < 0 && check==0){
  return false;
  }
  current_cage_index = mouse_idx;
  if (Select==true)
  {
    choosen_control_vertex = mouse_idx;
    prev_mouse << Zaxis(0), Zaxis(1);
    Flag = 1;
  } else
  {
    viewer.data(model_mesh_wei).set_data(cage_weights.row(current_cage_index));
    Flag = 0; 
    return true;
  }
  if(Flag==0)
  return false;
  else
  return true;
}

bool mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
  RowVector3d  r1=RowVector3d(1, 0, 0);
  RowVector3d  r2=RowVector3d(0, 1, 0);
  if (Flag==0) 
  {
    return false;
  }
  Vector3d Zaxis;
  Vector4d d=Vector4d(0, 0, 1, 0);
  igl::unproject_on_plane(
    Vector2i(viewer.current_mouse_x, viewer.core(screen).viewport(3) - viewer.current_mouse_y),
    viewer.core(screen).proj * viewer.core(screen).view,viewer.core(screen).viewport,d,Zaxis);

    
  Vector2d cur_mouse(Zaxis(0), Zaxis(1));
  Vector2d tranlation = cur_mouse - prev_mouse;

  prev_mouse = cur_mouse;
  if (choosen_control_vertex < Vertices_cage.rows())
  {
    double position=choosen_control_vertex;
    Vertices_cage.row(position) += tranlation;
  } else
  {
    double position=choosen_control_vertex - Vertices_cage.rows();
    control_vertices.row(position) += tranlation;
  }

  V.setZero();
  int it=0;
  while(it<V.rows())
  {
    int j=0;
    while (j<Vertices_cage.rows()+control_vertices.rows())
    {
      if (j < Vertices_cage.rows())
          {
            V.row(it) =V.row(it)+ weight(j, it) * Vertices_cage.row(j);
          }
      else
          { 
            V.row(it) = V.row(it)+ weight(j, it) * control_vertices.row(j-Vertices_cage.rows());
          }
      j++;
    }
    it++;
  }
  viewer.data(model_mesh).clear();
  viewer.data(model_mesh).set_mesh(V, F);
  viewer.data(model_mesh).add_points(Vertices_cage, r1);
  int i=0;

  while(i<Vertices_cage.rows())
  {
    viewer.data(model_mesh).add_edges(Vertices_cage.row(Faces_cage(i, 0)),Vertices_cage.row(Faces_cage(i, 1)),r1);
    i++;
  }

  viewer.data(model_mesh).add_points(control_vertices, r2);

  int j=0;
  while(j<control_faces.rows())
  {
    viewer.data(model_mesh).add_edges(control_vertices.row(control_faces(j, 0)),control_vertices.row(control_faces(j, 1)),r2);
    j++;
  }
  
  viewer.data(model_mesh_wei).set_data(cage_weights.row(current_cage_index));

  return true;
}

void update_edge(Viewer& viewer,MatrixXd Vertices,MatrixXd Faces_cage,RowVector3d r,int m,double n)
{
viewer.data(m).add_points(Vertices, r);
int in=0;
while(in<n)
  {
    viewer.data(m).add_edges(Vertices.row(Faces_cage(in, 0)),Vertices.row(Faces_cage(in, 1)),r);
    in++;
  }
}

bool mouse_up(Viewer& viewer, int button, int modifier)
{
  if (Flag==0)  
  {
    return false;
  }
  Flag = 0;
  choosen_control_vertex = -1;
  return true;
}

void calc_harmonic(MatrixXd x,MatrixXd Vertices,VectorXd ma ,VectorXd mi,VectorXd off)
{
  double a=ma(0)+off(0);
  double b=mi(0)-off(0);
  double c=ma(1)+off(1);
  double d=mi(1)-off(1);

  x.row(Vertices.rows()) << a,c;
  x.row(Vertices.rows()+1) << a,d;
  x.row(Vertices.rows()+2) << b,d;
  x.row(Vertices.rows()+3) << b,c;
}
// triangulates inside the cage
MatrixXd triang_vertices;
MatrixXi triang_faces;

void Spacing(int a)
{
  cage_vertices = VectorXi::LinSpaced(a, 0, a-1);
  rest_verties = VectorXi::LinSpaced(triang_vertices.rows()-a, 
                                              a, triang_vertices.rows()-1);
}

void Laplacian(SparseMatrix<double> L,SparseMatrix<double> A,int a,SparseMatrix<double> A_vert,SparseMatrix<double> A_cage)
{ 
  igl::cotmatrix(triang_vertices, triang_faces, L);
  A = (-L).eval();

  igl::slice(A, rest_verties, rest_verties, A_vert);
  igl::slice(A, rest_verties, cage_vertices, A_cage);

  SimplicialLLT<SparseMatrix<double>> solver;
  solver.compute(A_vert);
  
  VectorXd phi(a);
  int in2=0;
  while(in2<a)
  {
    phi.setZero();
    phi(in2) = 1;
    VectorXd h = solver.solve(-A_cage*phi);
    int j=0;
    while(j<triang_vertices.rows())
    {
      if (j < a)
      { 
        cage_weights(in2, j) = phi(j);
      }
      else
      {
        cage_weights(in2, j) = h(j - a);
      }
      j++;
    } 
    in2++;
  }
}

void Barycentric_cood(MatrixXd P1 ,MatrixXd P2, MatrixXd P3)
{
  MatrixXd P;
  igl::slice(triang_vertices, triang_faces.col(2), 1, P3);
  double prev_trip3=P3.rows();
  igl::slice(triang_vertices, triang_faces.col(1), 1, P2);
  double prev_trip2=P2.rows();
  igl::slice(triang_vertices, triang_faces.col(0), 1, P1);
  double prev_trip1=P1.rows();
  int in=0;
    while(in<V.rows())
    {
    P = V.row(in).replicate(triang_faces.rows(), 1);
    MatrixXd bcs ;
    igl::barycentric_coordinates(P, P1, P2, P3, bcs);
    int tri_idx = -1;
    int itr=0;
    while(itr<triang_faces.rows())
    {
      if (bcs(itr,0)<=1 && bcs(itr,0)>=0)
      {
        if(bcs(itr,1)<=1 && bcs(itr,1)>=0)
          {
            if(bcs(itr,2)<=1 && bcs(itr,2)>=0)
      {
        tri_idx = itr;
        break;
      }}}
      ++itr;
    }
    RowVector3d bc = bcs.row(tri_idx);
    int it=0;
    while(it<cage_weights.rows())
    {
      double x = bc(0)*cage_weights(it, triang_faces(tri_idx, 0));
      double y =bc(1)*cage_weights(it, triang_faces(tri_idx, 1));
      double z =bc(2)*cage_weights(it, triang_faces(tri_idx, 2));
      weight(it, in) = x+y+z;
    ++it;
    }
    in++;
  }
}

void Triangulation()
{
  MatrixXd points(Vertices_cage.rows() + control_vertices.rows() + V.rows(), 2);
  points << Vertices_cage, control_vertices, V;
  double size_mesh=points.rows()+Faces_cage.rows();
  MatrixXd mesh(points.rows()+Faces_cage.rows(),2);
  igl::triangle::triangulate(points, Faces_cage, Ht, "1", triang_vertices, triang_faces);
}

void calculate_harmonic_function() 
{
  int num_c = Vertices_cage.rows()+control_vertices.rows();
  Triangulation();
  double size=Vertices_cage.rows()+control_vertices.rows();
  cage_weights.resize(size, triang_vertices.rows());
  weight.resize(size, V.rows());

  Spacing(num_c);
  SparseMatrix<double> L,A;
  SparseMatrix<double> A_ff, A_Faces_cage;

  Laplacian(L,A,num_c,A_ff,A_Faces_cage);
  weight.setZero();
  MatrixXd p1,p2,p3;
  Barycentric_cood(p1,p2,p3);
}

void setM(MatrixXi x,MatrixXd Vertices)
{
  double a=Vertices.rows();
  double b=Vertices.rows()+1;
  double c=Vertices.rows()+2;
  double d=Vertices.rows()+3;
  x << a,b, b,c, c,d, d,a;
}

int label;
double counterX = 0;
double counterY = 0;
vector<double> grid_harmoic_coord ;
MatrixXd GridVertices; 
class Cell{
    public:    
    Cell() {
        label = UNTYPED;
    }
    void find_coord(){}
    void initialize(int n) {}
    void print(int n){}

};
class Derived : public Cell
{
  void print(int n){
    }
  void initialize(int n) {
        for (int i = 0; i<n ; i++ ) {
            grid_harmoic_coord.push_back(0);
        }
    }
    void find_coord(){}
};


typedef vector<Cell> v2d;
typedef vector<v2d> Grid;

Grid grid;
void grid_change(int grid_side_vertices,int total_vertices ,MatrixXd grid1){            
}

void Grid_Create(MatrixXd& grid, int grid_size) {
    int squarecells = (int)(sqrt(pow(2,grid_size)));
    int grid_side_vertices = squarecells+1;
    int total_vertices = grid_side_vertices*grid_side_vertices;
    grid_change(grid_side_vertices,total_vertices,grid);
   // int total_vertices = grid_side_vertices*grid_side_vertices;
    grid = MatrixXd(total_vertices, 2);
    int i =0;
    while(i<total_vertices) {

        for(int j = 0; j<grid_side_vertices; j++) {
            grid.row(i) = RowVector2d(counterX + offsetX,counterY + offsetY);
            counterX += h;
            i++;
        }
        counterX = 0;
        counterY -= h;
    }
    
}

void Re_size(MatrixXd a,MatrixXd b)
{
  a.resizeLike(b);
  a << b;
}

int main(int argc, char *argv[])
{
  
  igl::readOFF("../data/man.off",V,F);
  igl::readOFF("../data/cage.off", Vertices_cage, Faces_cage);
  double rowM=V.rows();
  double rowC=Vertices_cage.rows();
  
  V.conservativeResize(rowM, 2);
  Vertices_cage.conservativeResize(rowC, 2);
  Re_size(main_vertices,Vertices_cage);
  control_vertices.resize(0,2);
  Re_size(original_Vi,control_vertices);
  
 
  calculate_harmonic_function();
  
  MatrixXd starts;
  igl::slice(Vertices_cage, Faces_cage.col(0), 1, starts);
  MatrixXd ends;
  igl::slice(Vertices_cage, Faces_cage.col(1), 1, ends);
  min_dist = (ends - starts).rowwise().norm().minCoeff();
  
  VectorXd max(Vertices_cage.colwise().maxCoeff());
  VectorXd min(Vertices_cage.colwise().minCoeff());
  VectorXd offset = (max-min);
  MatrixXd V_square(Vertices_cage.rows()+4,2);
  int i=0;
  while(i<rowC)
  {
  V_square.row(i) << Vertices_cage.row(i);
  i++; 
  }
  calc_harmonic(V_square,Vertices_cage,max,min,offset);
  setM(E,Vertices_cage);
  // Plot the mesh
  Viewer viewer;
  Grid_Create(GridVertices, grid_size);
  int Rowcells = (int)(sqrt(pow(2,grid_size)));
    int Columncells = Rowcells;  //square grid we are working on
    int Total_cage_vertices = Cage.rows();
    
    Grid grid_(Rowcells, v2d(Columncells));
    grid = grid_;
    //initialize grid cells;
    for(int i = 0; i<Rowcells; i++ ){
        for(int j = 0; j<Columncells; j++ ){
            grid[i][j].initialize(Total_cage_vertices);
    
        }
    }
  RowVector3d  r1=RowVector3d(1, 0, 0);
  RowVector3d  r2=RowVector3d(0, 1, 0);

  viewer.data().add_points(GridVertices, r1);
  viewer.data().point_size = 1; //SIZE of control circles
  model_mesh = viewer.append_mesh(true);
  model_mesh_wei = viewer.append_mesh(true);
 
  viewer.callback_init = [&](Viewer &)
  {
    viewer.core().viewport = Vector4f(0, 0, 1240, 800);
    screen = viewer.core_list[0].id;
    viewer.data(model_mesh_wei).set_visible(false, screen);
  

    return true;
  };
  viewer.callback_post_resize = [&](Viewer &v, int w, int h) 
  {
    v.core(screen).viewport = Vector4f(0, 0, w/2, h);
    return true;
  };
 
  viewer.data(model_mesh).clear();
  viewer.data(model_mesh).set_mesh(V, F);
  //update_edge(viewer,Vertices,Faces_cage,RowVector3d(1,0,0),model_mesh);
  viewer.data(model_mesh).add_points(Vertices_cage, r1);
  int l=0;
  while(l<rowC)
  {
    viewer.data(model_mesh).add_edges(Vertices_cage.row(Faces_cage(l, 0)),Vertices_cage.row(Faces_cage(l, 1)),r1);
    l++;
  }
  viewer.data(model_mesh).add_points(control_vertices, r2);
 
  int j=0;
  while(j<control_faces.rows())
  {
    viewer.data(model_mesh).add_edges(Vertices_cage.row(Faces_cage(j, 0)),Vertices_cage.row(Faces_cage(j, 1)),r2);
    j++;
  }
  viewer.data(model_mesh_wei).set_mesh(triang_vertices, triang_faces);
  viewer.data(model_mesh_wei).set_data(cage_weights.row(current_cage_index));
  viewer.data(model_mesh_wei).add_points(Vertices_cage, r1);
 
  int k=0;
  while(k<rowC)
  {
    viewer.data(model_mesh).add_edges(Vertices_cage.row(Faces_cage(k, 0)),Vertices_cage.row(Faces_cage(k, 1)),r1);
    k++;
  }
  viewer.data(model_mesh_wei).add_points(control_vertices, r2);

  int ij=0;
  while(ij<control_faces.rows())
  {
    viewer.data(model_mesh).add_edges(Vertices_cage.row(Faces_cage(ij, 0)),Vertices_cage.row(Faces_cage(ij, 1)),r1);
    ij++;
  }

  viewer.callback_mouse_down = mouse_down;
  viewer.callback_mouse_move = mouse_move;
  viewer.callback_mouse_up = mouse_up;

  viewer.launch();
}
