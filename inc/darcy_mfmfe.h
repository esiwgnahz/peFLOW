//
// Created by eldar on 6/4/17.
//

#ifndef PEFLOW_DARCY_MFMFE_H
#define PEFLOW_DARCY_MFMFE_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include "utilities.h"
#include <unordered_map>

namespace darcy
{
  using namespace dealii;
  using namespace utilities;

  template <int dim>
  class MultipointMixedDarcyProblem
  {
  public:
    MultipointMixedDarcyProblem (const unsigned int degree);
    void run (const unsigned int refine, const unsigned int grid = 0);
  private:
    const unsigned int  degree;
    Triangulation<dim>  triangulation;
    FESystem<dim>       fe;
    DoFHandler<dim>     dof_handler;
    BlockVector<double> solution;

    void compute_errors (const unsigned int cycle);
    void output_results (const unsigned int cycle,  const unsigned int refine);

    struct VertexAssemblyScratchData
    {
      VertexAssemblyScratchData (const FiniteElement<dim> &fe,
                                 const Triangulation<dim>       &tria,
                                 const Quadrature<dim> &quad,
                                 const Quadrature<dim-1> &f_quad);

      VertexAssemblyScratchData (const VertexAssemblyScratchData &scratch_data);

      FEValues<dim>       fe_values;
      FEFaceValues<dim>   fe_face_values;
      std::vector<int>    n_faces_at_vertex;
      const unsigned long num_cells;
    };

    struct VertexAssemblyCopyData
    {
      MapPointMatrix<dim>                  cell_mat;
      MapPointVector<dim>                  cell_vec;
      MapPointSet<dim>                     local_pres_indices;
      MapPointSet<dim>                     local_vel_indices;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    struct VertexEliminationCopyData
    {
      // Assembly
      FullMatrix<double> vertex_pres_matrix;
      Vector<double>     vertex_pres_rhs;
      FullMatrix<double> Ainverse;
      FullMatrix<double> pressure_matrix;
      Vector<double>     velocity_rhs;
      // Recovery
      Vector<double>     vertex_vel_solution;
      // Indexing
      Point<dim>         p;
    };

    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               VertexAssemblyScratchData                         &scratch_data,
                               VertexAssemblyCopyData                            &copy_data);
    void copy_cell_to_vertex (const VertexAssemblyCopyData &copy_data);
    void vertex_assembly ();
    void vertex_elimination (const typename MapPointMatrix<dim>::iterator &n_it,
                             VertexAssemblyScratchData                    &scratch_data,
                             VertexEliminationCopyData                    &copy_data);
    void copy_vertex_to_system (const VertexEliminationCopyData                            &copy_data);
    void pressure_assembly ();
    void solve_pressure ();
    void velocity_assembly (const typename MapPointMatrix<dim>::iterator &n_it,
                            VertexAssemblyScratchData                  &scratch_data,
                            VertexEliminationCopyData                  &copy_data);
    void copy_vertex_velocity_to_global (const VertexEliminationCopyData &copy_data);
    void velocity_recovery ();
    void make_cell_centered_sp ();
    void reset_data_structures ();

    SparsityPattern cell_centered_sp;
    SparseMatrix<double> pres_system_matrix;
    Vector<double> pres_rhs;

    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> pressure_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> A_inverse;
    std::unordered_map<Point<dim>, Vector<double>, hash_points<dim>, points_equal<dim>> velocity_rhs;

    MapPointMatrix<dim> vertex_matrix;
    MapPointVector<dim> vertex_rhs;

    MapPointSet<dim> pressure_indices;
    MapPointSet<dim> velocity_indices;

    unsigned long n_v, n_p;

    Vector<double> pres_solution;
    Vector<double> vel_solution;

    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;
  };

}

#endif //PEFLOW_DARCY_MFMFE_H
