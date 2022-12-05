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

// #define SIMPLE_TRIANGULATION

using namespace std;

// original mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;
// cage mesh
Eigen::MatrixXd original_Vc;
Eigen::MatrixXd Vc;
Eigen::MatrixXi Fc;
Eigen::MatrixXi H;
// interior control mesh
Eigen::MatrixXd original_Vi;
Eigen::MatrixXd Vi;
Eigen::MatrixXi Fi;
// triangulates inside the cage
Eigen::MatrixXd Vt;
Eigen::MatrixXi Ft;
// weight function
Eigen::MatrixXd h_weight_in_cage;
Eigen::MatrixXd h_weight;
Eigen::MatrixXd m_weight_in_cage;
Eigen::MatrixXd m_weight;
// used for display 
int current_cage_index = 0;
// used for deformation
Eigen::Vector2d previous_mouse_coordinate;
int picked_cage_vertex;
bool doit = false;
double selection_threshold;
int original_mesh, h_weight_mesh, m_weight_mesh;
unsigned int left_view, right_view;
enum CoordinateType {Harmonic};
CoordinateType coordinate_type = Harmonic;


void set_original_mesh(igl::opengl::glfw::Viewer& viewer);

int nearest_control_vertex(Eigen::Vector3d &click_point, bool original_cage)
{
  Eigen::RowVector2d click_point_2d(click_point(0), click_point(1));
  Eigen::MatrixXd& cage = original_cage ? original_Vc : Vc;
  Eigen::MatrixXd& interor = original_cage? original_Vi : Vi;
  int cage_index;
  double cage_dist = (cage.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&cage_index);
  int interior_index;
  double interior_dist = interor.rows()>0&&coordinate_type==Harmonic ? (interor.rowwise() - click_point_2d).rowwise().squaredNorm().minCoeff(&interior_index) : DBL_MAX;
  double dist = std::min(cage_dist, interior_dist);
  if (dist > selection_threshold)
    return -1;
  int index = cage_dist < interior_dist ? cage_index : cage.rows() + interior_index;
  return index;
}

void set_original_mesh(igl::opengl::glfw::Viewer& viewer)
{
  viewer.data(original_mesh).clear();
  viewer.data(original_mesh).set_mesh(V, F);
  viewer.data(original_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(original_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  }
  viewer.data(original_mesh).add_points(Vi, Eigen::RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(original_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      Eigen::RowVector3d(0,1,0)
    );
  }
}
void calculate_harmonic_function() 
{
  // triangulates the cage
  //libigl triangulate
  int control_vertex_num = Vc.rows()+Vi.rows();

  #ifdef SIMPLE_TRIANGULATION
  Eigen::MatrixXd points(Vc.rows() + Vi.rows(), 2);
  points << Vc, Vi;
  igl::triangle::triangulate(points, Fc, H, "q", Vt, Ft);
  #else
  Eigen::MatrixXd points(Vc.rows() + Vi.rows() + V.rows(), 2);
  points << Vc, Vi, V;
  igl::triangle::triangulate(points, Fc, H, "", Vt, Ft);
  #endif

  h_weight_in_cage.resize(Vc.rows()+Vi.rows(), Vt.rows());
  h_weight.resize(Vc.rows()+Vi.rows(), V.rows());


  Eigen::VectorXi free_vertices, cage_vertices;
  cage_vertices = Eigen::VectorXi::LinSpaced(control_vertex_num, 0, control_vertex_num-1);
  free_vertices = Eigen::VectorXi::LinSpaced(Vt.rows()-control_vertex_num, 
                                              control_vertex_num, Vt.rows()-1);
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(Vt, Ft, L);

  
  Eigen::SparseMatrix<double> A;
  
  A = (-L).eval();
 // Solver code from libigl

  Eigen::SparseMatrix<double> A_ff, A_fc;
  igl::slice(A, free_vertices, free_vertices, A_ff);
  igl::slice(A, free_vertices, cage_vertices, A_fc);
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(A_ff);
  assert(solver.info() == Eigen::Success);
  Eigen::VectorXd phi(control_vertex_num);
  for (int i = 0; i < control_vertex_num; ++i) 
  {
    phi.setZero();
    phi(i) = 1;
    Eigen::VectorXd h = solver.solve(-A_fc*phi);
   
    for (int j = 0; j < Vt.rows(); ++j)
    {
      if (j < control_vertex_num)
        h_weight_in_cage(i, j) = phi(j);
      else
        h_weight_in_cage(i, j) = h(j - control_vertex_num);
    } 
    #ifndef SIMPLE_TRIANGULATION
    // if (i < h_weight.rows())
    //   for (int j = 0; j < h_weight.cols(); ++j)
    //     h_weight(i,j) = h(j);
      // h_weight.row(i) << h.transpose();
    #endif
  }
  // #ifdef SIMPLE_TRIANGULATION
  Eigen::MatrixXd P1, P2, P3;
  igl::slice(Vt, Ft.col(0), 1, P1);
  igl::slice(Vt, Ft.col(1), 1, P2);
  igl::slice(Vt, Ft.col(2), 1, P3);
  h_weight.setZero();
  for (int idx = 0; idx < V.rows(); ++idx)
  {
    Eigen::MatrixXd P = V.row(idx).replicate(Ft.rows(), 1);
    Eigen::MatrixXd bcs;
    igl::barycentric_coordinates(P, P1, P2, P3, bcs);
    int triangle_idx = -1;
    for (int i = 0; i < Ft.rows(); ++i)
    {
      if (bcs(i,0)<=1 && bcs(i,0)>=0 &&
          bcs(i,1)<=1 && bcs(i,1)>=0 &&
          bcs(i,2)<=1 && bcs(i,2)>=0)
      {
  
        triangle_idx = i;
        break;
      }
    }
    assert(triangle_idx != -1);
    Eigen::RowVector3d bc = bcs.row(triangle_idx);
   
    for (int i = 0; i < h_weight_in_cage.rows(); ++i)
    { 
      h_weight(i, idx) = bc(0)*h_weight_in_cage(i, Ft(triangle_idx, 0)) + bc(1)*h_weight_in_cage(i, Ft(triangle_idx, 1)) + bc(2)*h_weight_in_cage(i, Ft(triangle_idx, 2));
    }
  }

}
using namespace Eigen;
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
const double h = 7; //jumps between cells of the cage 
const int interpolationPrecision = 40; // point on edges to detect cells
const int s = 12; //based on paper, s controls size of grid 2^s
const double offsetX = -50; //X offset of the grid
const double offsetY = 430; //Y offset of the grid
MatrixXd Cage; //Cage vertices
MatrixXi CageEdgesIndices; 
MatrixXd Ec; //Cage edges

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

    void to_string() {
        cout<<"\nCell Type: "<<to_enum(label)<<endl;
        cout<<"Harmonic Coordinates: [";

        for(auto it = harmonicCoordinates.begin(); it<harmonicCoordinates.end(); it++) {
            cout<< *it;
            if(it != harmonicCoordinates.end()-1) {
                cout<<", ";
            }
        }
        cout<<"]"<<endl;
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
  if (argc < 3)
  {
  
    exit(0);
  }
  igl::readOFF(argv[1],V,F);
  assert(V.rows() > 0);
 
  igl::readOFF(argv[2], Vc, Fc);

  V.conservativeResize(V.rows(), 2);
  Vc.conservativeResize(Vc.rows(), 2);
  original_Vc.resizeLike(Vc);
  original_Vc << Vc;

  if (argc == 4)
  {
    igl::readOFF(argv[3], Vi, Fi);
    Vi.conservativeResize(Vi.rows(), 2);
  
  } else
  {
    Vi.resize(0,2);
  }
  original_Vi.resizeLike(Vi);
  original_Vi << Vi;
 
 
  calculate_harmonic_function();
  

  Eigen::VectorXd max(Vc.colwise().maxCoeff());
  Eigen::VectorXd min(Vc.colwise().minCoeff());
  Eigen::VectorXd offset = (max-min)*0.1;
  Eigen::MatrixXd V_square(Vc.rows()+4,2);
  for (int i = 0; i < Vc.rows(); ++i)
    V_square.row(i) << Vc.row(i);
  V_square.row(Vc.rows()) << max(0)+offset(0), max(1)+offset(1);
  V_square.row(Vc.rows()+1) << max(0)+offset(0), min(1)-offset(1);
  V_square.row(Vc.rows()+2) << min(0)-offset(0), min(1)-offset(1);
  V_square.row(Vc.rows()+3) << min(0)-offset(0), max(1)+offset(1);
  Eigen::MatrixXi E(4,2);
  E << Vc.rows(),Vc.rows()+1,
       Vc.rows()+1,Vc.rows()+2,
       Vc.rows()+2,Vc.rows()+3,
       Vc.rows()+3,Vc.rows();
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;
  igl::triangle::triangulate(V_square, E, H, "a50q", V2, F2);
  m_weight_in_cage.resize(Vc.rows(), V2.rows());
  m_weight_in_cage.setZero();
  Eigen::VectorXi rows = Eigen::VectorXi::LinSpaced(V2.rows()-Vc.rows(), Vc.rows(), V2.rows()-1);
  Eigen::MatrixXd tmp_v, tmp_weight;
  for (int i = 0; i < Vc.rows(); ++i)
    m_weight_in_cage(i,i)=1;
  
  // Plotting the mesh
  igl::opengl::glfw::Viewer viewer;
  original_mesh = viewer.append_mesh(true);
  h_weight_mesh = viewer.append_mesh(true);
  m_weight_mesh = viewer.append_mesh(true);
  viewer.callback_init = [&](igl::opengl::glfw::Viewer &)
  {
    viewer.core().viewport = Eigen::Vector4f(0, 0, 640, 800);
    left_view = viewer.core_list[0].id;
    right_view = viewer.append_core(Eigen::Vector4f(640, 0, 640, 800));
  viewer.data(original_mesh).set_visible(false, right_view);
    viewer.data(m_weight_mesh).set_visible(false, right_view);
    return true;
  };
  viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) 
  {
    v.core(left_view).viewport = Eigen::Vector4f(0, 0, w / 2, h);
    v.core(right_view).viewport = Eigen::Vector4f(w / 2, 0, w - (w / 2), h);
    return true;
  };
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
            //grid[i][j].to_string(); //to print the cell type and harmonic coord
        }
    }
  viewer.data().add_points(GridVertices, RowVector3d(1, 0, 0));
  viewer.data().point_size = 2; //SIZE of control circles
  set_original_mesh(viewer);
  viewer.data(h_weight_mesh).set_mesh(Vt, Ft);
  viewer.data(h_weight_mesh).set_data(h_weight_in_cage.row(current_cage_index));
  viewer.data(h_weight_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(h_weight_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  }
  viewer.data(h_weight_mesh).add_points(Vi, Eigen::RowVector3d(0,1,0));
  for (int i = 0; i < Fi.rows(); ++i)
  {
    viewer.data(h_weight_mesh).add_edges(
      Vi.row(Fi(i, 0)),
      Vi.row(Fi(i, 1)),
      Eigen::RowVector3d(0,1,0)
    );
  }
 
  viewer.data(m_weight_mesh).set_mesh(V2,F2);
  viewer.data(m_weight_mesh).set_data(m_weight_in_cage.row(current_cage_index));
  viewer.data(m_weight_mesh).add_points(Vc, Eigen::RowVector3d(1,0,0));
  for (int i = 0; i < Vc.rows(); ++i)
  {
    viewer.data(m_weight_mesh).add_edges(
      Vc.row(Fc(i, 0)),
      Vc.row(Fc(i, 1)),
      Eigen::RowVector3d(1,0,0)
    );
  } 
  viewer.data(h_weight_mesh).show_lines = false;
  viewer.data(m_weight_mesh).show_lines = false;
  viewer.launch();
}
