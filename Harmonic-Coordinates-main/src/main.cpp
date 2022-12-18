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
#include <float.h>
#include <math.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/triangle/triangulate.h>
#include <igl/cotmatrix.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/unproject_on_plane.h>

using namespace std;
using namespace Eigen;
using namespace igl::opengl::glfw;

MatrixXd V;
MatrixXi F;
// cage mesh
MatrixXd original_Vc;
MatrixXd Vc;
MatrixXi Fc;
MatrixXi H;
// interior control mesh
MatrixXd original_Vi;
MatrixXd Vi;
MatrixXi Fi;
// triangulates inside the cage
MatrixXd Vt;
MatrixXi Ft;
// weight function
MatrixXd weight_in_cage;
MatrixXd weight;

// used for display 
int current_cage_index = 0;
// used for deform
Vector2d previous_mouse_coordinate;
MatrixXi E(4,2);
VectorXi free_vertices, cage_vertices;
int picked_cage_vertex;
int flag = 0;
// double selection_threshold;
double min_dist;
int mesh_2d, weight_mesh;
unsigned int screen;

enum cell_Type{UNTYPED, EXTERIOR, BOUNDARY, INTERIOR};
string to_enum(int label) {
    if(label == UNTYPED) {
        return "UNTYPED";
    }
    if(label == EXTERIOR) {
        return "EXTERIOR";
    }
    if(label == BOUNDARY) {
        return "BOUNDARY";
    }
    if(label == INTERIOR) {
        return "INTERIOR";
    }
    return "Not Found";
}
double h = 7; //jumps between cells of the cage 
int interpolationPrecision = 40; // point on edges to detect cells
int s = 12; //based on paper, s controls size of grid 2^s
double offsetX = -50; //X offset of the grid
double offsetY = 430; //Y offset of the grid
MatrixXd Cage; //Cage vertices
MatrixXi CageEdgesIndices; 
MatrixXd Ec; //Cage edges

int nearest_control_vertex(Vector3d &click_point, bool original_cage)
{
  RowVector2d click_point_2d(click_point(0), click_point(1));
  MatrixXd cage;
  MatrixXd interor; 
  if(original_cage)
  {
  cage=original_Vc;
  interor=original_Vi;
  }
  else
  {
  cage=Vc;
  interor=Vi;
  }
  
  int cage_index;
  double cage_dist = (cage.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&cage_index);
  int interior_index;
  double interior_dist;
  if(interor.rows()>0){ 
  interior_dist = (interor.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&interior_index);
  }
  else
  {
    interior_dist= DBL_MAX;
  }
  double dist = std::min(cage_dist, interior_dist);
  if (dist > min_dist/3)
    return -1;
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

bool callback_mouse_down(Viewer& viewer, int button, int modifier)
{
  if (button == (int) Viewer::MouseButton::Right)
  {
    return false;
  }
  bool click_left = viewer.current_mouse_x < viewer.core(screen).viewport(2);
  Vector3d Z;

    // example from libigl tut 708
    igl::unproject_on_plane(
      Vector2i(viewer.current_mouse_x, viewer.core(screen).viewport(3) - viewer.current_mouse_y),
      viewer.core(screen).proj * viewer.core(screen).view,
      viewer.core(screen).viewport,
      Vector4d(0,0,1,0),
      Z
    );

  int idx = nearest_control_vertex(Z, !click_left);
  if (idx < 0)
    return false;
  current_cage_index = idx;
  if (click_left)
  {
    picked_cage_vertex = idx;
    previous_mouse_coordinate << Z(0), Z(1);
    flag = 1;
  } else
  {
    viewer.data(weight_mesh).set_data(weight_in_cage.row(current_cage_index));
   
    flag = 0; 
    return true;
  }
  if(flag==0)
  return false;
  else
  return true;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
  if (flag==0) 
  {
    return false;
  }
  Vector3d Z;
  igl::unproject_on_plane(
    Vector2i(viewer.current_mouse_x, viewer.core(screen).viewport(3) - viewer.current_mouse_y),
    viewer.core(screen).proj * viewer.core(screen).view,
    viewer.core(screen).viewport,
    Vector4d(0, 0, 1, 0),Z
  );
  Vector2d current_mouse_coordinate(Z(0), Z(1));
  Vector2d translation = current_mouse_coordinate - previous_mouse_coordinate;

  previous_mouse_coordinate = current_mouse_coordinate;
  if (picked_cage_vertex < Vc.rows())
  {
    Vc.row(picked_cage_vertex) += translation;
  } else
  {
    Vi.row(picked_cage_vertex - Vc.rows()) += translation;
  }

   V.setZero();
  int it=0;
  while(it<V.rows())
  {
    int j=0;
    while (j<Vc.rows()+Vi.rows())
    {
      if (j < Vc.rows())
          {
            V.row(it) =V.row(it)+ weight(j, it) * Vc.row(j);
          }
      else
          { 
            V.row(it) = V.row(it)+ weight(j, it) * Vi.row(j-Vc.rows());
          }
      ++j;
    }
    ++it;
  }
  viewer.data(mesh_2d).clear();
  viewer.data(mesh_2d).set_mesh(V, F);
  viewer.data(mesh_2d).add_points(Vc, RowVector3d(1,0,0));

  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(mesh_2d).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      RowVector3d(1,0,0)
    );
  }
  viewer.data(mesh_2d).add_points(Vi, RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(mesh_2d).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      RowVector3d(0,1,0)
    );
  }
  
  viewer.data(weight_mesh).set_data(weight_in_cage.row(current_cage_index));

  return true;
}

void update_edge(Viewer& viewer,MatrixXd Vc,MatrixXd Fc,RowVector3d r,int m)
{
viewer.data(m).add_points(Vc, r);
int in=0;
while(in<Vc.rows())
//  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(m).add_edges(Vc.row(Fc(in, 0)),Vc.row(Fc(in, 1)),r);
    in++;
  }
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier)
{
  if (flag==0)  
  {
    return false;
  }
  flag = 0;
  picked_cage_vertex = -1;
  return true;
}

void calc(MatrixXd x,MatrixXd Vc,VectorXd ma ,VectorXd mi,VectorXd off)
{
  x.row(Vc.rows()) << ma(0)+off(0), ma(1)+off(1);
  x.row(Vc.rows()+1) << ma(0)+off(0), mi(1)-off(1);
  x.row(Vc.rows()+2) << mi(0)-off(0), mi(1)-off(1);
  x.row(Vc.rows()+3) << mi(0)-off(0), ma(1)+off(1);
}
void setM(MatrixXi x,MatrixXd Vc)
{
  x << Vc.rows(),Vc.rows()+1,
       Vc.rows()+1,Vc.rows()+2,
       Vc.rows()+2,Vc.rows()+3,
       Vc.rows()+3,Vc.rows();
}

void calculate_harmonic_function() 
{
  // triangulate the cage
  int control_vertex_num = Vc.rows()+Vi.rows();

  MatrixXd points(Vc.rows() + Vi.rows() + V.rows(), 2);
  points << Vc, Vi, V;
  igl::triangle::triangulate(points, Fc, H, "", Vt, Ft);

  weight_in_cage.resize(Vc.rows()+Vi.rows(), Vt.rows());
  weight.resize(Vc.rows()+Vi.rows(), V.rows());

  // Set up linear solver
  cage_vertices = VectorXi::LinSpaced(control_vertex_num, 0, control_vertex_num-1);
  free_vertices = VectorXi::LinSpaced(Vt.rows()-control_vertex_num, 
                                              control_vertex_num, Vt.rows()-1);
  SparseMatrix<double> L,A;
  SparseMatrix<double> A_ff, A_fc;
  
  
  igl::cotmatrix(Vt, Ft, L);
  A = (-L).eval();

  igl::slice(A, free_vertices, free_vertices, A_ff);
  igl::slice(A, free_vertices, cage_vertices, A_fc);

  SimplicialLLT<SparseMatrix<double>> solver;
  solver.compute(A_ff);
  
  VectorXd phi(control_vertex_num);
  //for (int i = 0; i < control_vertex_num; ++i) 
  int in2=0;
  while(in2<control_vertex_num)
  {
    phi.setZero();
    phi(in2) = 1;
    VectorXd h = solver.solve(-A_fc*phi);
    //for (int j = 0; j < Vt.rows(); ++j)
    int j=0;
    while(j<Vt.rows())
    {
      if (j < control_vertex_num)
      { 
        weight_in_cage(in2, j) = phi(j);
      }
      else
      {
        weight_in_cage(in2, j) = h(j - control_vertex_num);
      }
      j++;
    } 
    in2++;
  }
  weight.setZero();
  // TRIANGULATION
  MatrixXd P1;
  igl::slice(Vt, Ft.col(0), 1, P1);
  MatrixXd P2;
  igl::slice(Vt, Ft.col(1), 1, P2);
  MatrixXd P3;
  igl::slice(Vt, Ft.col(2), 1, P3);
    int in=0;
    while(in<V.rows())
    {
    MatrixXd P = V.row(in).replicate(Ft.rows(), 1);
    MatrixXd bcs;
    igl::barycentric_coordinates(P, P1, P2, P3, bcs);
    int triangle_idx = -1;
    int itr=0;
    while(itr<Ft.rows())
    {
      if (bcs(itr,0)<=1 && bcs(itr,0)>=0)
      {
        if(bcs(itr,1)<=1 && bcs(itr,1)>=0)
          {
            if(bcs(itr,2)<=1 && bcs(itr,2)>=0)
      {
        triangle_idx = itr;
        break;
      }}}
      ++itr;
    }
    RowVector3d bc = bcs.row(triangle_idx);
    int it=0;
    while(it<weight_in_cage.rows())
    {
      double x = bc(0)*weight_in_cage(it, Ft(triangle_idx, 0));
      double y =bc(1)*weight_in_cage(it, Ft(triangle_idx, 1));
      double z =bc(2)*weight_in_cage(it, Ft(triangle_idx, 2));
      weight(it, in) = x+y+z;
    ++it;
    }
    in++;
  }

}

MatrixXd GridVertices; 
struct Cell{
    vector<double> harmonicCoordinates ;
    int label;
    Cell() {
        label = UNTYPED;
    }
    void initialize(int n) {
        for (int i = 0; i<n ; i++ ) {
            harmonicCoordinates.push_back(0);
        }
        label = UNTYPED; 
    }
};

typedef vector<Cell> v2d;
typedef vector<v2d> Grid;

Grid grid;

void createGridVertices(MatrixXd& grid, int s) {
    int squareSideCells = (int)(sqrt(pow(2,s)));
    int verticesPerSide = squareSideCells+1;
    int nbOfVertices = verticesPerSide*verticesPerSide;
    double counterX = 0;
    double counterY = 0;
    grid = MatrixXd(nbOfVertices, 2);
    int i =0;
    while(i<nbOfVertices) {

        for(int j = 0; j<verticesPerSide; j++) {
            grid.row(i) = RowVector2d(counterX + offsetX,counterY + offsetY);
            counterX += h;
            i++;
        }
        counterX = 0;
        counterY -= h;
    }
}

int main(int argc, char *argv[])
{
  igl::readOFF("../data/man.off",V,F);
  igl::readOFF("../data/cage.off", Vc, Fc);
  V.conservativeResize(V.rows(), 2);
  Vc.conservativeResize(Vc.rows(), 2);
  original_Vc.resizeLike(Vc);
  original_Vc << Vc;
  Vi.resize(0,2);
  original_Vi.resizeLike(Vi);
  original_Vi << Vi;
 
  calculate_harmonic_function();

  MatrixXd starts;
  igl::slice(Vc, Fc.col(0), 1, starts);
  MatrixXd ends;
  igl::slice(Vc, Fc.col(1), 1, ends);
  min_dist = (ends - starts).rowwise().norm().minCoeff();
  
  VectorXd max(Vc.colwise().maxCoeff());
  VectorXd min(Vc.colwise().minCoeff());
  VectorXd offset = (max-min);
  MatrixXd V_square(Vc.rows()+4,2);
  // for (int i = 0; i < Vc.rows(); ++i)
  // V_square.row(i) << Vc.row(i);
  int i=0;
  while(i<Vc.rows())
  {
  V_square.row(i) << Vc.row(i);
  i++; 
  }
  calc(V_square,Vc,max,min,offset);
  setM(E,Vc);
  // Plot the mesh
  Viewer viewer;
   createGridVertices(GridVertices, s);
  int numberOrRowCells = (int)(sqrt(pow(2,s)));
    int numberOrColCells = numberOrRowCells;  //square grid we are working on
    int cageVerticesCount = Cage.rows();
    Grid grid_(numberOrRowCells, v2d(numberOrColCells));
    grid = grid_;
    //initialize grid cells;
    for(int i = 0; i<numberOrRowCells; i++ ){
        for(int j = 0; j<numberOrColCells; j++ ){
            grid[i][j].initialize(cageVerticesCount);
    
        }
    }
  viewer.data().add_points(GridVertices, RowVector3d(1, 0, 0));
  viewer.data().point_size = 1; //SIZE of control circles
  mesh_2d = viewer.append_mesh(true);
  weight_mesh = viewer.append_mesh(true);
 
  viewer.callback_init = [&](Viewer &)
  {
    viewer.core().viewport = Vector4f(0, 0, 1240, 800);
    screen = viewer.core_list[0].id;
    viewer.data(weight_mesh).set_visible(false, screen);
  

    return true;
  };
  viewer.callback_post_resize = [&](Viewer &v, int w, int h) 
  {
    v.core(screen).viewport = Vector4f(0, 0, w/2, h);
    return true;
  };
 
  viewer.data(mesh_2d).clear();
  viewer.data(mesh_2d).set_mesh(V, F);
  //update_edge(viewer,Vc,Fc,RowVector3d(1,0,0),mesh_2d);
  viewer.data(mesh_2d).add_points(Vc, RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(mesh_2d).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      RowVector3d(1,0,0)
    );
  }
  viewer.data(mesh_2d).add_points(Vi, RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(mesh_2d).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      RowVector3d(0,1,0)
    );
  }
  viewer.data(weight_mesh).set_mesh(Vt, Ft);
  viewer.data(weight_mesh).set_data(weight_in_cage.row(current_cage_index));
  viewer.data(weight_mesh).add_points(Vc, RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(weight_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      RowVector3d(1,0,0)
    );
  }
  viewer.data(weight_mesh).add_points(Vi, RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(weight_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      RowVector3d(0,1,0)
    );
  }

  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;

  viewer.launch();
}