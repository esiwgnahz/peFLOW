// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/work_stream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "../inc/elasticity_mfe.h"
#include "../inc/elasticity_data.h"
#include "../inc/utilities.h"

namespace elasticity
{
  using namespace dealii;
  using namespace utilities;

  // MixedElasticityProblem: class constructor
  template <int dim>
  MixedElasticityProblem<dim>::MixedElasticityProblem (const unsigned int deg, ParameterHandler &param)
          :
          prm(param),
          degree(deg),
          total_dim(dim*dim + dim + static_cast<int>(0.5*dim*(dim-1))),
          fe(FE_BDM<dim>(deg), dim,
             FE_DGP<dim>(deg-1), dim,
             FE_DGP<dim>(deg-1), static_cast<int>(0.5*dim*(dim-1))),
          dof_handler(triangulation),
          computing_timer(std::cout, TimerOutput::summary,
                          TimerOutput::wall_times)
  {}

  // MixedElasticityProblem: make grid and DoFs
  template <int dim>
  void MixedElasticityProblem<dim>::make_grid_and_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Make grid and DOFs");

    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
    system_matrix.clear();

    dof_handler.distribute_dofs(fe);

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim*dim + dim + rotation_dim);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    unsigned int n_s=0, n_u=0, n_p=0;

    for (unsigned int i=0; i<dim; ++i)
    {
      n_s += dofs_per_component[i*dim];
      n_u += dofs_per_component[dim*dim + i];
      // Rotation is scalar in 2d and vector in 3d, so this:
      if (dim == 2)
        n_p = dofs_per_component[dim*dim + dim];
      else if (dim == 3)
        n_p += dofs_per_component[dim*dim + dim + i];
    }

    const unsigned int n_couplings = dof_handler.max_couplings_between_dofs();

    sparsity_pattern.reinit(3,3);
    sparsity_pattern.block(0,0).reinit (n_s, n_s, n_couplings);
    sparsity_pattern.block(1,0).reinit (n_u, n_s, n_couplings);
    sparsity_pattern.block(2,0).reinit (n_p, n_s, n_couplings);

    sparsity_pattern.block(0,1).reinit (n_s, n_u, n_couplings);
    sparsity_pattern.block(1,1).reinit (n_u, n_u, n_couplings);
    sparsity_pattern.block(2,1).reinit (n_p, n_u, n_couplings);

    sparsity_pattern.block(0,2).reinit (n_s, n_p, n_couplings);
    sparsity_pattern.block(1,2).reinit (n_u, n_p, n_couplings);
    sparsity_pattern.block(2,2).reinit (n_p, n_p, n_couplings);
    sparsity_pattern.collect_sizes();

    DoFTools::make_sparsity_pattern (dof_handler, sparsity_pattern);
    sparsity_pattern.compress();

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(3);
    solution.block(0).reinit(n_s);
    solution.block(1).reinit(n_u);
    solution.block(2).reinit(n_p);
    solution.collect_sizes();

    system_rhs.reinit(3);
    system_rhs.block(0).reinit(n_s);
    system_rhs.block(1).reinit(n_u);
    system_rhs.block(2).reinit(n_p);
    system_rhs.collect_sizes();
  }

  // Scratch data for multithreading
  template <int dim>
  MixedElasticityProblem<dim>::CellAssemblyScratchData::
  CellAssemblyScratchData (const FiniteElement<dim> &fe,
                           const Quadrature<dim>    &quadrature,
                           const Quadrature<dim-1>  &face_quadrature,
                           const LameCoefficients<dim> &lame_data,
                           Functions::ParsedFunction<dim> *bc,
                           Functions::ParsedFunction<dim> *rhs)
          :
          fe_values (fe,
                     quadrature,
                     update_values   | update_gradients |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (fe,
                          face_quadrature,
                          update_values     | update_quadrature_points   |
                          update_JxW_values | update_normal_vectors),
          lame(lame_data),
          bc(bc),
          rhs(rhs)
  {}

  template <int dim>
  MixedElasticityProblem<dim>::CellAssemblyScratchData::
  CellAssemblyScratchData (const CellAssemblyScratchData &scratch_data)
          :
          fe_values (scratch_data.fe_values.get_fe(),
                     scratch_data.fe_values.get_quadrature(),
                     update_values   | update_gradients |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (scratch_data.fe_face_values.get_fe(),
                          scratch_data.fe_face_values.get_quadrature(),
                          update_values     | update_quadrature_points   |
                          update_JxW_values | update_normal_vectors),
          lame(scratch_data.lame),
          bc(scratch_data.bc),
          rhs(scratch_data.rhs)
  {}

  // Copy local contributions to global system
  template <int dim>
  void MixedElasticityProblem<dim>::copy_local_to_global (const CellAssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
    {
      for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
        system_matrix.add (copy_data.local_dof_indices[i],
                           copy_data.local_dof_indices[j],
                           copy_data.cell_matrix(i,j));
      system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
    }
  }

  // Function to assemble on a cell
  template <int dim>
  void MixedElasticityProblem<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                          CellAssemblyScratchData                                   &scratch_data,
                                                          CellAssemblyCopyData                                      &copy_data)
  {
    const unsigned int rotation_dim    = static_cast<int>(0.5*dim*(dim-1));
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

    copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
    copy_data.cell_rhs.reinit (dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);

    scratch_data.fe_values.reinit (cell);

    // Stress and rotation DoFs vectors
    std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
    std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());

    // Displacement DoFs
    const FEValuesExtractors::Vector displacement (dim*dim);

    for (unsigned int d=0; d<dim; ++d)
    {
      const FEValuesExtractors::Vector tmp_stress(d*dim);
      stresses[d].first_vector_component = tmp_stress.first_vector_component;
      if (dim == 2 && d == 0)
      {
        const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim);
        rotations[d].component = tmp_rotation.component;
      } else if (dim == 3) {
        const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + d);
        rotations[d].component = tmp_rotation.component;
      }
    }


    // Stress, divergence and rotation
    std::vector<std::vector<Tensor<1,dim>>> phi_i_s(dofs_per_cell, std::vector<Tensor<1,dim>> (dim));
    std::vector<Tensor<1,dim>> div_phi_i_s(dofs_per_cell);
    std::vector<Tensor<1,dim>> phi_i_u(dofs_per_cell);
    std::vector<Tensor<1,rotation_dim>> phi_i_p(dofs_per_cell);

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        // Evaluate test functions
        for (unsigned int s_i=0; s_i<dim; ++s_i)
        {
          phi_i_s[k][s_i] = scratch_data.fe_values[stresses[s_i]].value (k, q);
          div_phi_i_s[k][s_i] = scratch_data.fe_values[stresses[s_i]].divergence (k, q);
        }
        phi_i_u[k] = scratch_data.fe_values[displacement].value (k, q);
        for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
          phi_i_p[k][r_i] = scratch_data.fe_values[rotations[r_i]].value (k, q);
      }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        Point<dim> point = scratch_data.fe_values.get_quadrature_points()[q];
        const double mu = scratch_data.lame.mu_value(point);
        const double lambda = scratch_data.lame.lambda_value(point);
        Tensor<2,dim> asigma = compliance_tensor_stress<dim>(phi_i_s[i], mu, lambda);
        for (unsigned int j=i; j<dofs_per_cell; ++j)
        {
          Tensor<2,dim> sigma = make_tensor(phi_i_s[j]);
          copy_data.cell_matrix(i,j) += (scalar_product(asigma, sigma)
                                         + scalar_product(phi_i_u[i], div_phi_i_s[j])
                                         + scalar_product(phi_i_u[j], div_phi_i_s[i])
                                         + scalar_product(phi_i_p[i], make_asymmetry_tensor(phi_i_s[j]))
                                         + scalar_product(phi_i_p[j], make_asymmetry_tensor(phi_i_s[i]))) * scratch_data.fe_values.JxW(q);
        }

        for (unsigned d_i=0; d_i<dim; ++d_i)
          copy_data.cell_rhs(i) += -(phi_i_u[i][d_i] * scratch_data.rhs->value(scratch_data.fe_values.get_quadrature_points()[q], d_i)) * scratch_data.fe_values.JxW(q);
      }
    }

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=i+1; j<dofs_per_cell; ++j)
        copy_data.cell_matrix(j,i) = copy_data.cell_matrix(i,j);

    for (unsigned int face_no=0;
         face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if (cell->at_boundary(face_no)) // && (cell->face(face_no)->boundary_id() == 1)
      {
        scratch_data.fe_face_values.reinit (cell, face_no);

        for (unsigned int q=0; q<n_face_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            Tensor<2,dim> sigma;
            for (unsigned int d_i=0; d_i<dim; ++d_i)
              sigma[d_i] = scratch_data.fe_face_values[stresses[d_i]].value (i, q);

            Tensor<1,dim> sigma_n = sigma * scratch_data.fe_face_values.normal_vector(q);
            for (unsigned int d_i=0; d_i<dim; ++d_i)
              copy_data.cell_rhs(i) += ((sigma_n[d_i]*scratch_data.bc->value(scratch_data.fe_face_values.get_quadrature_points()[q],d_i))
                                       *scratch_data.fe_face_values.JxW(q));

          }
      }
    cell->get_dof_indices (copy_data.local_dof_indices);
  }

  template <int dim>
  void MixedElasticityProblem<dim>::assemble_system ()
  {
    Functions::ParsedFunction<dim> *mu                  = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *lambda              = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *bc       = new Functions::ParsedFunction<dim>(dim);
    Functions::ParsedFunction<dim> *rhs      = new Functions::ParsedFunction<dim>(dim);

    prm.enter_subsection(std::string("lambda ") + Utilities::int_to_string(dim)+std::string("D"));
    lambda->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("mu ") + Utilities::int_to_string(dim)+std::string("D"));
    mu->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("BC ")+ Utilities::int_to_string(dim)+std::string("D"));
    bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("RHS ") + Utilities::int_to_string(dim)+std::string("D"));
    rhs->parse_parameters(prm);
    prm.leave_subsection();

    LameCoefficients<dim> lame(prm,mu, lambda);

    TimerOutput::Scope t(computing_timer, "Assemble system");
    QGauss<dim> quad(2*(degree+1)+1);
    QGauss<dim-1> face_quad(2*(degree+1)+1);

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MixedElasticityProblem::assemble_system_cell,
                    &MixedElasticityProblem::copy_local_to_global,
                    CellAssemblyScratchData(fe,quad,face_quad, lame, bc, rhs),
                    CellAssemblyCopyData());

    delete mu;
    delete lambda;
    delete bc;
    delete rhs;
  }


  // MixedElasticityProblem: Solve
  template <int dim>
  void MixedElasticityProblem<dim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "Solve (Direct UMFPACK)");

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
  }

  // MixedElasticityProblem: Compute errors
  template <int dim>
  void MixedElasticityProblem<dim>::compute_errors(const unsigned cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute errors");

    const ComponentSelectFunction<dim> rotation_mask(dim*dim+dim, MixedElasticityProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> displacement_mask(std::make_pair(dim*dim,dim*dim+dim), MixedElasticityProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> stress_mask(std::make_pair(0,dim*dim), MixedElasticityProblem<dim>::total_dim);
    
    ExactSolution<dim> exact_solution(prm);
    prm.enter_subsection(std::string("Exact solution ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_val_data.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Exact gradient ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_grad_val_data.parse_parameters(prm);
    prm.leave_subsection();


    // Vectors to temporarily store cellwise errros
    Vector<double> cellwise_errors (triangulation.n_active_cells());
    Vector<double> cellwise_norms (triangulation.n_active_cells());

    // Vectors to temporarily store cellwise componentwise div errors
    Vector<double> cellwise_div_errors (triangulation.n_active_cells());
    Vector<double> cellwise_div_norms (triangulation.n_active_cells());

    // Define quadrature points to compute errors at
    QTrapez<1>      q_trapez;
    QIterated<dim>  quadrature(q_trapez,degree+2);

    // This is used to show superconvergence at midcells
    QGauss<dim>   quadrature_super(1);

    // Since we want to compute the relative norm
    BlockVector<double> zerozeros(1, solution.size());

    // Rotation error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &rotation_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &rotation_mask);
    const double p_l2_norm = cellwise_norms.l2_norm();

    // Displacement error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_norm = cellwise_norms.l2_norm();

    // Displacement error and norm at midcells
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature_super,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature_super,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double u_l2_mid_norm = cellwise_norms.l2_norm();

    // Stress L2 error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &stress_mask);
    const double s_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &stress_mask);

    const double s_l2_norm = cellwise_norms.l2_norm();

    // Stress Hdiv seminorm
    cellwise_errors = 0;
    cellwise_norms = 0;
    for (int i=0; i<dim; ++i){
      const ComponentSelectFunction<dim> stress_component_mask (std::make_pair(i*dim,(i+1)*dim),MixedElasticityProblem<dim>::total_dim);

      VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                         cellwise_div_errors, quadrature,
                                         VectorTools::Hdiv_seminorm,
                                         &stress_component_mask);
      cellwise_errors += cellwise_div_errors;

      VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                         cellwise_div_norms, quadrature,
                                         VectorTools::Hdiv_seminorm,
                                         &stress_component_mask);
      cellwise_norms += cellwise_div_norms;
    }

    const double s_hd_error = cellwise_errors.l2_norm();
    const double s_hd_norm = cellwise_norms.l2_norm();

    // Assemble convergence table
    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("Stress,L2", s_l2_error/s_l2_norm);
    convergence_table.add_value("Stress,Hdiv", s_hd_error/s_hd_norm); //sx_hd_error/sx_hd_norm
    convergence_table.add_value("Displ,L2", u_l2_error/u_l2_norm);
    convergence_table.add_value("Displ,L2mid", u_l2_mid_error/u_l2_mid_norm);
    convergence_table.add_value("Rotat,L2", p_l2_error/p_l2_norm);
  }


  // MixedElasticityProblem: Output results
  template <int dim>
  void MixedElasticityProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
  {
    TimerOutput::Scope t(computing_timer, "Output results");
    std::vector<std::string> solution_names;
    std::string rhs_name = "rhs";

    switch(dim)
    {
      case 2:
        solution_names.insert(solution_names.end(), {"s11","s12","s21","s22","u","v","p"});
        break;

      case 3:
        solution_names.insert(solution_names.end(),
                              {"s11","s12","s13","s21","s22","s23","s31","s32","s33","u","v","w","p1","p2","p3"});
        break;

      default:
      Assert(false, ExcNotImplemented());
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(total_dim-1, DataComponentInterpretation::component_is_part_of_vector);

    switch (dim)
    {
      case 2:
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_scalar);
        break;

      case 3:
        data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);
        break;

      default:
      Assert(false, ExcNotImplemented());
        break;
    }

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);

    data_out.build_patches ();

    std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk (output);

    convergence_table.set_precision("Stress,L2", 3);
    convergence_table.set_precision("Stress,Hdiv", 3);
    convergence_table.set_precision("Displ,L2", 3);
    convergence_table.set_precision("Displ,L2mid", 3);
    convergence_table.set_precision("Rotat,L2", 3);
    convergence_table.set_scientific("Stress,L2", true);
    convergence_table.set_scientific("Stress,Hdiv", true);
    convergence_table.set_scientific("Displ,L2", true);
    convergence_table.set_scientific("Displ,L2mid", true);
    convergence_table.set_scientific("Rotat,L2", true);
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("Stress,L2", "$ \\|\\sigma - \\sigma_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Stress,Hdiv", "$ \\|\\nabla\\cdot(\\sigma - \\sigma_h)\\|_{L^2} $");
    convergence_table.set_tex_caption("Displ,L2", "$ \\|u - u_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Displ,L2mid", "$ \\|Qu - u_h\\|_{L^2} $");
    convergence_table.set_tex_caption("Rotat,L2", "$ \\|p - p_h\\|_{L^2} $");
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    convergence_table.evaluate_convergence_rates("Stress,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Stress,Hdiv", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Displ,L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Displ,L2mid", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates("Rotat,L2", ConvergenceTable::reduction_rate_log2);

    std::ofstream error_table_file("error" + std::to_string(dim) + "d.tex");

    if (cycle == refine-1){
      convergence_table.write_text(std::cout);
      convergence_table.write_tex(error_table_file);
    }
  }

  // MixedElasticityProblem: run
  template <int dim>
  void MixedElasticityProblem<dim>::run(const unsigned int refine, const unsigned int grid)
  {
    dof_handler.clear();
    triangulation.clear();
    convergence_table.clear();

    for (unsigned int cycle=0; cycle<refine; ++cycle)
    {
      if(cycle == 0){
        if(grid){
          GridIn<dim> grid_in;
          grid_in.attach_triangulation (triangulation);
          std::string mesh_filename ("mesh"+std::to_string(dim)+"d.msh");
          std::ifstream input_file(mesh_filename);

          Assert(input_file.is_open(), ExcFileNotOpen(mesh_filename.c_str()));
          Assert(triangulation.dimension == dim, ExcDimensionMismatch(triangulation.dimension, dim));

          grid_in.read_msh (input_file);
        } else {
          GridGenerator::hyper_cube (triangulation, 0, 1);
          triangulation.refine_global(1);
        }

        typename Triangulation<dim>::cell_iterator
                cell = triangulation.begin (),
                endc = triangulation.end();
        for (; cell!=endc; ++cell)
          for (unsigned int face_number=0;
               face_number<GeometryInfo<dim>::faces_per_cell;
               ++face_number)
            if ((std::fabs(cell->face(face_number)->center()(0) - (1)) < 1e-12)
                ||
                (std::fabs(cell->face(face_number)->center()(1) - (1)) < 1e-12))
              cell->face(face_number)->set_boundary_id (1);
      } else {
        triangulation.refine_global(1);
      }

      make_grid_and_dofs();
      assemble_system ();
      solve ();
      compute_errors(cycle);
      output_results (cycle, refine);
      computing_timer.print_summary();
      computing_timer.reset();
    }
  }

  template class MixedElasticityProblem<2>;
  template class MixedElasticityProblem<3>;
}
