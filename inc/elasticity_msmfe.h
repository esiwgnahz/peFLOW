//
// Created by eldar on 6/4/17.
//

#ifndef PEFLOW_ELASTICITY_MSMFE_H
#define PEFLOW_ELASTICITY_MSMFE_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include "utilities.h"
#include <unordered_map>

namespace elasticity
{
  using namespace dealii;
  using namespace utilities;

  template <int dim>
  class MultipointMixedElasticityProblem
  {
  public:
    MultipointMixedElasticityProblem (const unsigned int degree);
    void run (const unsigned int refine, const unsigned int grid = 0);
  private:
    const unsigned int  degree;
    const unsigned int  total_dim;
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
      MapPointSet<dim>                     local_displ_indices;
      MapPointSet<dim>                     local_stress_indices;
      MapPointSet<dim>                     local_rotation_indices;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    struct VertexEliminationCopyData
    {
      // Assembly
      FullMatrix<double> vertex_displ_matrix;
      Vector<double>     vertex_displ_rhs;
      FullMatrix<double> Ainverse;
      FullMatrix<double> CACinverse;
      FullMatrix<double> displacement_matrix;
      FullMatrix<double> rotation_matrix;
      Vector<double>     stress_rhs;
      Vector<double>     rotation_rhs;
      // Recovery
      Vector<double>     vertex_stress_solution;
      Vector<double>     vertex_rotation_solution;
      // Indexing
      Point<dim>         p;
    };

    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               VertexAssemblyScratchData                            &scratch_data,
                               VertexAssemblyCopyData                               &copy_data);
    void copy_cell_to_vertex (const VertexAssemblyCopyData &copy_data);
    void vertex_assembly ();
    void vertex_elimination (const typename MapPointMatrix<dim>::iterator &n_it,
                             VertexAssemblyScratchData                    &scratch_data,
                             VertexEliminationCopyData                    &copy_data);
    void copy_vertex_to_system (const VertexEliminationCopyData &copy_data);
    void displacement_assembly ();
    void solve_displacement ();
    void sr_assembly (const typename MapPointMatrix<dim>::iterator &n_it,
                                   VertexAssemblyScratchData                  &scratch_data,
                                   VertexEliminationCopyData                  &copy_data);
    void copy_vertex_sr_to_global (const VertexEliminationCopyData &copy_data);
    void sr_recovery ();
    void make_cell_centered_sp ();
    void reset_data_structures ();

    SparsityPattern cell_centered_sp;
    SparseMatrix<double> displ_system_matrix;
    Vector<double> displ_rhs;

    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> displacement_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> rotation_matrix;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> A_inverse;
    std::unordered_map<Point<dim>, FullMatrix<double>, hash_points<dim>, points_equal<dim>> CAC_inverse;
    std::unordered_map<Point<dim>, Vector<double>, hash_points<dim>, points_equal<dim>> stress_rhs;
    std::unordered_map<Point<dim>, Vector<double>, hash_points<dim>, points_equal<dim>> rotation_rhs;

    MapPointMatrix<dim> vertex_matrix;
    MapPointVector<dim> vertex_rhs;

    MapPointSet<dim> displacement_indices;
    MapPointSet<dim> stress_indices;
    MapPointSet<dim> rotation_indices;

    unsigned long n_s, n_u, n_p;

    Vector<double> displ_solution;
    Vector<double> stress_solution;
    Vector<double> rotation_solution;

    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;
  };

}

#endif //PEFLOW_ELASTICITY_MSMFE_H
