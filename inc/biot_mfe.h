// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_BIOT_MFE_H
#define PEFLOW_BIOT_MFE_H

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/parsed_function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/sparse_direct.h>

#include "../inc/problem.h"
#include "../inc/utilities.h"
#include "../inc/biot_data.h"

namespace biot
{
  using namespace dealii;

   template <int dim>
  class MixedBiotProblem : public Problem<dim>
  {
  public:
    MixedBiotProblem(const unsigned int degree,
                     ParameterHandler &,
                     const double time_step,
                     const unsigned int num_time_steps);
    void run(const unsigned int refine, const unsigned int grid = 0);
  private:
    ParameterHandler &prm;

    struct CellAssemblyScratchData
    {
      CellAssemblyScratchData (const FiniteElement<dim> &fe,
                               const KInverse<dim> &k_data,
                               const LameCoefficients<dim> &lame,
                               Functions::ParsedFunction<dim> *darcy_bc,
                               Functions::ParsedFunction<dim> *darcy_rhs,
                               Functions::ParsedFunction<dim> *elasticity_bc,
                               Functions::ParsedFunction<dim> *elasticity_rhs,
                               const double c_0, const double alpha);
      CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data);
      FEValues<dim>     fe_values;
      FEFaceValues<dim> fe_face_values;
      KInverse<dim>     K_inv;
      LameCoefficients<dim> lame;
      Functions::ParsedFunction<dim> *darcy_bc;
      Functions::ParsedFunction<dim> *darcy_rhs;
      Functions::ParsedFunction<dim> *elasticity_bc;
      Functions::ParsedFunction<dim> *elasticity_rhs;
      double            c_0;
      double            alpha;
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

    void assemble_rhs();
    void assemble_rhs_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                            CellAssemblyScratchData                             &scratch,
                            CellAssemblyCopyData                                &copy_data);
    void copy_local_mat_to_global (const CellAssemblyCopyData &copy_data);
    void copy_local_rhs_to_global (const CellAssemblyCopyData &copy_data);

    void solve();
    void compute_errors (const unsigned int cycle);
    void output_results (const unsigned int cycle,  const unsigned int refine);
    void set_current_errors_to_zero();

    const unsigned int degree;
    double time;
    const double time_step;
    const unsigned int num_time_steps;

    const int total_dim = dim+1+ dim*dim + dim + static_cast<int>(dim*(dim-1)/2);

    std::vector<double> l2_l2_norms;
    std::vector<double> l2_l2_errors;

    std::vector<double> linf_l2_norms;
    std::vector<double> linf_l2_errors;

    std::vector<double> velocity_stress_l2_div_norms;
    std::vector<double> velocity_stress_l2_div_errors;

    std::vector<double> velocity_stress_linf_div_norms;
    std::vector<double> velocity_stress_linf_div_errors;

    std::vector<double> pressure_disp_l2_midcell_norms;
    std::vector<double> pressure_disp_l2_midcell_errors;

    std::vector<double> pressure_disp_linf_midcell_norms;
    std::vector<double> pressure_disp_linf_midcell_errors;

    // void invert_spd (const FullMatrix<double> &A, FullMatrix<double> &X);

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> solution;
    BlockVector<double> old_solution;
    BlockVector<double> system_rhs;

    ConvergenceTable convergence_table;

    SparseDirectUMFPACK  A_direct;
  };

}

#endif //PEFLOW_BIOT_MFE_H
