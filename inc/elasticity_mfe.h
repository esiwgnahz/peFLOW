// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_ELASTICITY_MFE_H
#define PEFLOW_ELASTICITY_MFE_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

namespace elasticity
{
  using namespace dealii;

  template <int dim>
  class MixedElasticityProblem
  {
  public:
    MixedElasticityProblem(const unsigned int deg);
    void run(const unsigned int refine, const unsigned int grid = 0);
  private:
    struct CellAssemblyScratchData
    {
      CellAssemblyScratchData (const FiniteElement<dim> &fe,
                               const Quadrature<dim>    &quadrature,
                               const Quadrature<dim-1>  &face_quadrature);
      CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data);
      FEValues<dim>     fe_values;
      FEFaceValues<dim> fe_face_values;
    };

    struct CellAssemblyCopyData
    {
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
    };


    void make_grid_and_dofs();
    void assemble_system();
    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               CellAssemblyScratchData                             &scratch,
                               CellAssemblyCopyData                                &copy_data);
    void copy_local_to_global (const CellAssemblyCopyData &copy_data);
    void solve();
    void compute_errors (const unsigned int cycle);
    void output_results (const unsigned int cycle,  const unsigned int refine);

    const unsigned int degree;
    const int          total_dim;
    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;

    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;
  };
}

#endif //PEFLOW_ELASTICITY_MFE_H
