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

  // TODO: change Biot to not read parameters from file on every time step

  /*
   * Class implementing the method to solve five-field mixed Biot problem.
   * In this current version it is a coupled MSMFE-MFMFE system with
   * inexact quadrature rule, although the actual elimination
   * procedure is not yet implemented.
   */
   template <int dim>
  class MixedBiotProblem : public Problem<dim>
  {
  public:
    /*
     * Class constructor takes degree and reference to parameter handle
     * as arguments. It also needs time_step and the total number of
     * time steps for simulation
     */
    MixedBiotProblem(const unsigned int degree,
                     ParameterHandler &,
                     const double time_step,
                     const unsigned int num_time_steps);

    /*
     * Main driver function
     */
    void run(const unsigned int refine, const unsigned int grid = 0);
  private:
    /*
     * Reference to a parameter handler object that stores parameters,
     * data and the exact solution
     */
    ParameterHandler &prm;

    /*
     * Data structure holding the information needed by threads
     * during assembly process
     */
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

    /*
     * Structure to copy data from threads to the main
     */
    struct CellAssemblyCopyData
    {
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    /*
     * Make grid, distribute DoFs and create sparsity pattern
     */
    void make_grid_and_dofs();

    /*
     * Assemble cell matrix only as it may stay constant over the time,
     * worker function for each thread
     */
    void assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                               CellAssemblyScratchData                             &scratch,
                               CellAssemblyCopyData                                &copy_data);

    /*
     * Copy data from threads to main
     */
    void copy_local_mat_to_global (const CellAssemblyCopyData &copy_data);

    /*
     * Function to assign each thread to matrix assembly
     */
    void assemble_system();

    /*
     * Assemble cell RHS separately as it will change over the time,
     * worker function for each thread
     */
    void assemble_rhs_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                            CellAssemblyScratchData                             &scratch,
                            CellAssemblyCopyData                                &copy_data);

    /*
     * Copy data from threads to main
     */
    void copy_local_rhs_to_global (const CellAssemblyCopyData &copy_data);

    /*
     * Function to assign each thread to RHS assembly
     */
    void assemble_rhs();

    /*
     * Solve the saddle-point type system (Direct UMFPACK)
     */
    void solve();

    /*
     * Functions that compute errors and output results and
     * convergence rates
     */
    void compute_errors (const unsigned int cycle);
    void output_results (const unsigned int cycle,  const unsigned int refine);

    /*
     * Reset errors at the end of refinement cycle
     */
    void set_current_errors_to_zero();

    /*
     * Total number of components of the solution, it is needed for
     * convenience, as elasticity problem has different dimension
     * for the rotation variable in 2 and 3 dimension
     */
    const int total_dim = dim+1+ dim*dim + dim + static_cast<int>(dim*(dim-1)/2);

    /*
     * We store errors on each time step, as they are needed for L2 in time
     * convergence rates
     */
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

    /*
     * Data structures and internal parameters
     */
    const unsigned int degree;
    double time;
    const double time_step;
    const unsigned int num_time_steps;

    Triangulation<dim> triangulation;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> solution;
    BlockVector<double> old_solution;
    BlockVector<double> system_rhs;

    /*
     * Convergence table and wall-time timer objects
     */
    ConvergenceTable convergence_table;
    TimerOutput      computing_timer;

    /*
     * Store the factorization of a matrix to be reused
     * on each time step (only makes sense if mu and lambda
     * are independent of time)
     */
    SparseDirectUMFPACK  A_direct;
  };

}

#endif //PEFLOW_BIOT_MFE_H
