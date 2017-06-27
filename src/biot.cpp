// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_rt_bubbles.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/base/work_stream.h>

#include <fstream>
#include <iostream>
#include <cmath>

#include "../inc/biot_mfe.h"
#include "../inc/biot_data.h"
#include "../inc/biot_parameter_reader.h"
#include "../inc/utilities.h"

namespace biot
{
  using namespace dealii;
  using namespace utilities;


  // MixedBiotProblem: class constructor
  template <int dim>
  MixedBiotProblem<dim>::MixedBiotProblem (const unsigned int degree, ParameterHandler &param,
                                           const double time_step,
                                           const unsigned int num_time_steps)
    :
      prm(param),
      total_dim(dim + 1 + dim*dim + dim + static_cast<int>(dim*(dim-1)/2)),
      l2_l2_norms   (5, 0.0),
      l2_l2_errors  (5, 0.0),
      linf_l2_norms (5, 0.0),
      linf_l2_errors(5, 0.0),
      velocity_stress_l2_div_norms     (2, 0.0),
      velocity_stress_l2_div_errors    (2, 0.0),
      velocity_stress_linf_div_norms   (2, 0.0),
      velocity_stress_linf_div_errors  (2, 0.0),
      pressure_disp_l2_midcell_norms   (2, 0.0),
      pressure_disp_l2_midcell_errors  (2, 0.0),
      pressure_disp_linf_midcell_norms (2, 0.0),
      pressure_disp_linf_midcell_errors(2, 0.0),
      degree(degree),
      time(0.0),
      time_step(time_step),
      num_time_steps(num_time_steps),
      fe(FE_RT_Bubbles<dim>(degree+1), 1,
         FE_DGQ<dim>(degree), 1,
         FE_RT_Bubbles<dim>(degree+1), dim,
         FE_DGQ<dim>(degree), dim,
         FE_Q<dim>(degree+1), static_cast<int>(0.5*dim*(dim-1)) ),
      dof_handler(triangulation),
      computing_timer(std::cout, TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  // MixedBiotProblem: make grid and DoFs
  template <int dim>
  void MixedBiotProblem<dim>::make_grid_and_dofs()
  {
    TimerOutput::Scope t(computing_timer, "Make grid and DOFs");
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));
    system_matrix.clear();

    dof_handler.distribute_dofs(fe);

    DoFRenumbering::component_wise (dof_handler);

    std::vector<types::global_dof_index> dofs_per_component (dim+ 1+dim*dim + dim + rotation_dim);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
    unsigned int n_u = dofs_per_component[0], n_p = dofs_per_component[dim],
        n_s=0, n_d=0, n_r=0;

    for (unsigned int i=0; i<dim; ++i)
      {
        n_s += dofs_per_component[i*dim+dim+1];
        n_d += dofs_per_component[dim*dim + i+dim+1];
        // Rotation is scalar in 2d and vector in 3d, so this:
        if (dim == 2)
          n_r = dofs_per_component[dim*dim + dim+dim+1];
        else if (dim == 3)
          n_r += dofs_per_component[dim*dim + dim + i+dim+1];
      }

    BlockDynamicSparsityPattern dsp(5, 5);
    dsp.block(0, 0).reinit (n_u, n_u);
    dsp.block(1, 0).reinit (n_p, n_u);
    dsp.block(2, 0).reinit (n_s, n_u);
    dsp.block(3, 0).reinit (n_d, n_u);
    dsp.block(4, 0).reinit (n_r, n_u);

    dsp.block(0, 1).reinit (n_u, n_p);
    dsp.block(1, 1).reinit (n_p, n_p);
    dsp.block(2, 1).reinit (n_s, n_p);
    dsp.block(3, 1).reinit (n_d, n_p);
    dsp.block(4, 1).reinit (n_r, n_p);

    dsp.block(0, 2).reinit (n_u, n_s);
    dsp.block(1, 2).reinit (n_p, n_s);
    dsp.block(2, 2).reinit (n_s, n_s);
    dsp.block(3, 2).reinit (n_d, n_s);
    dsp.block(4, 2).reinit (n_r, n_s);

    dsp.block(0, 3).reinit (n_u, n_d);
    dsp.block(1, 3).reinit (n_p, n_d);
    dsp.block(2, 3).reinit (n_s, n_d);
    dsp.block(3, 3).reinit (n_d, n_d);
    dsp.block(4, 3).reinit (n_r, n_d);

    dsp.block(0, 4).reinit (n_u, n_r);
    dsp.block(1, 4).reinit (n_p, n_r);
    dsp.block(2, 4).reinit (n_s, n_r);
    dsp.block(3, 4).reinit (n_d, n_r);
    dsp.block(4, 4).reinit (n_r, n_r);


    dsp.collect_sizes ();
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit (sparsity_pattern);


    solution.reinit (5);
    solution.block(0).reinit (n_u);
    solution.block(1).reinit (n_p);
    solution.block(2).reinit (n_s);
    solution.block(3).reinit (n_d);
    solution.block(4).reinit (n_r);
    solution.collect_sizes ();

    old_solution.reinit (5);
    old_solution.block(0).reinit (n_u);
    old_solution.block(1).reinit (n_p);
    old_solution.block(2).reinit (n_s);
    old_solution.block(3).reinit (n_d);
    old_solution.block(4).reinit (n_r);
    old_solution.collect_sizes ();

    system_rhs.reinit (5);
    system_rhs.block(0).reinit (n_u);
    system_rhs.block(1).reinit (n_p);
    system_rhs.block(2).reinit (n_s);
    system_rhs.block(3).reinit (n_d);
    system_rhs.block(4).reinit (n_r);
    system_rhs.collect_sizes ();


    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if ((cell->face(f)->at_boundary())
              &&
              (cell->face(f)->center()[1]==1.0))
            cell->face(f)->set_all_boundary_ids(1);
        }
  }

  // Scratch data for multithreading
  template <int dim>
  MixedBiotProblem<dim>::CellAssemblyScratchData::
  CellAssemblyScratchData (const FiniteElement<dim> &fe, const KInverse<dim> &k_data, const LameCoefficients<dim> &lame_data,
                           Functions::ParsedFunction<dim> *darcy_bc,
                           Functions::ParsedFunction<dim> *darcy_rhs, Functions::ParsedFunction<dim> *elasticity_bc,
                           Functions::ParsedFunction<dim> *elasticity_rhs,
                           const double c_0, const double alpha)
    :
      fe_values (fe,
                 QTrapez<dim>(),
                 update_values   | update_gradients |
                 update_quadrature_points | update_JxW_values),
      fe_face_values (fe,
                      QGauss<dim-1>(1),
                      update_values     | update_quadrature_points   |
                      update_JxW_values | update_normal_vectors),
      K_inv(k_data),
      lame(lame_data),
      darcy_bc(darcy_bc),
      darcy_rhs(darcy_rhs),
      elasticity_bc(elasticity_bc),
      elasticity_rhs(elasticity_rhs),
      c_0(c_0),
      alpha(alpha)
  {}

  template <int dim>
  MixedBiotProblem<dim>::CellAssemblyScratchData::
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
      K_inv(scratch_data.K_inv),
      lame(scratch_data.lame),
      darcy_bc(scratch_data.darcy_bc),
      darcy_rhs(scratch_data.darcy_rhs),
      elasticity_bc(scratch_data.elasticity_bc),
      elasticity_rhs(scratch_data.elasticity_rhs),
      c_0(scratch_data.c_0),
      alpha(scratch_data.alpha)
  {}


  // Copy local contributions to global system
  template <int dim>
  void MixedBiotProblem<dim>::copy_local_mat_to_global (const CellAssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
      for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j){
          system_matrix.add (copy_data.local_dof_indices[i],
                             copy_data.local_dof_indices[j],
                             copy_data.cell_matrix(i,j));
        }
  }


  // Copy local contributions to global rhs
  template <int dim>
  void MixedBiotProblem<dim>::copy_local_rhs_to_global (const CellAssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
      system_rhs(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
  }


  // Function to assemble on a cell
  template <int dim>
  void MixedBiotProblem<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                    CellAssemblyScratchData                                   &scratch_data,
                                                    CellAssemblyCopyData                                      &copy_data)
  {
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();

    const unsigned int rotation_dim    = static_cast<int>(0.5*dim*(dim-1));

    copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);

    scratch_data.fe_values.reinit (cell);

    std::vector<Tensor<2,dim>>             k_inverse_values (n_q_points);
    scratch_data.K_inv.value_list (scratch_data.fe_values.get_quadrature_points(), k_inverse_values);

    // Velocity and Stress DoFs
    FEValuesExtractors::Vector velocity(0);
    std::vector<FEValuesExtractors::Vector> stress(dim, FEValuesExtractors::Vector());
    // Pressure and Displacement DoFs
    const FEValuesExtractors::Scalar        pressure (dim);
    const FEValuesExtractors::Vector        displacement (dim*dim+dim+1);
    std::vector<FEValuesExtractors::Scalar> rotation(rotation_dim, FEValuesExtractors::Scalar());
    // Stress and Roatation DoFs
    for (unsigned int d=0; d<dim; ++d)
      {
        const FEValuesExtractors::Vector tmp_stress(d*dim+dim+1);
        stress[d].first_vector_component = tmp_stress.first_vector_component;
        if (dim == 2 && d == 0)
          {
            const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + dim + 1);
            rotation[d].component = tmp_rotation.component;
          } else if (dim == 3) {
            const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + d + dim + 1);
            rotation[d].component = tmp_rotation.component;
          }
      }


    // Velocity and pressure
    std::vector<Tensor<1,dim> >               phi_i_u(dofs_per_cell);
    std::vector <double>                      div_phi_i_u(dofs_per_cell);
    std::vector <double>                      phi_i_p(dofs_per_cell);

    // Stress, displacement and rotation
    std::vector<std::vector<Tensor<1,dim> > > phi_i_s(dofs_per_cell, std::vector<Tensor<1,dim> > (dim));
    std::vector<Tensor<1,dim> > div_phi_i_s(dofs_per_cell);
    std::vector<Tensor<1,dim> > phi_i_d(dofs_per_cell);
    std::vector<Tensor<1,rotation_dim> > phi_i_r(dofs_per_cell);

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            // Evaluate test functions
            phi_i_u[k] = scratch_data.fe_values[velocity].value (k, q);
            phi_i_p[k] = scratch_data.fe_values[pressure].value (k, q);
            div_phi_i_u[k] = scratch_data.fe_values[velocity].divergence (k, q);

            for (unsigned int s_i=0; s_i<dim; ++s_i)
              {
                phi_i_s[k][s_i] = scratch_data.fe_values[stress[s_i]].value (k, q);
                div_phi_i_s[k][s_i] = scratch_data.fe_values[stress[s_i]].divergence (k, q);
              }
            phi_i_d[k] = scratch_data.fe_values[displacement].value (k, q);
            for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
              phi_i_r[k][r_i] = scratch_data.fe_values[rotation[r_i]].value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            Point<dim> point = scratch_data.fe_values.get_quadrature_points()[q];
            const double mu = scratch_data.lame.mu_value(point);
            const double lambda = scratch_data.lame.lambda_value(point);
            Tensor<2,dim> asigma = compliance_tensor_stress<dim>(phi_i_s[i], mu, lambda);
            Tensor<2,dim> apId = compliance_tensor_pressure<dim>(phi_i_p[i], mu, lambda);

            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
                Tensor<2,dim> sigma = make_tensor(phi_i_s[j]);

                copy_data.cell_matrix(i,j) += (phi_i_u[i] * k_inverse_values[q] * phi_i_u[j]+ time_step*div_phi_i_u[j] * phi_i_p[i]   // Darcy eq-n
                                               -phi_i_p[j] * div_phi_i_u[i] + scratch_data.c_0*phi_i_p[i]*phi_i_p[j] + scratch_data.alpha*trace(asigma)*phi_i_p[j] // Coupling eq-n
                                               + scratch_data.alpha*scratch_data.alpha*trace(apId)*phi_i_p[j]
                                               + scalar_product(asigma, sigma) + scratch_data.alpha*scalar_product(apId, sigma)  // Mixed elasticity eq-ns
                                               + scalar_product(phi_i_d[i], div_phi_i_s[j])  + scalar_product(phi_i_d[j], div_phi_i_s[i])
                                               + scalar_product(phi_i_r[i], make_asymmetry_tensor(phi_i_s[j]))
                                               + scalar_product(phi_i_r[j], make_asymmetry_tensor(phi_i_s[i])) )
                                               * scratch_data.fe_values.JxW(q);

              }
          }
      }



    cell->get_dof_indices (copy_data.local_dof_indices);
  }


  // Function to assemble on a cell
  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                 CellAssemblyScratchData                                   &scratch_data,
                                                 CellAssemblyCopyData                                      &copy_data)
  {
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();

    const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();


    copy_data.cell_rhs.reinit (dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);


    scratch_data.fe_values.reinit (cell);

    scratch_data.darcy_bc->set_time(time);
    scratch_data.elasticity_bc->set_time(time);

    scratch_data.darcy_rhs->set_time(time);
    scratch_data.elasticity_rhs->set_time(time);

    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));

    // Velocity and Stress DoFs vectors
    FEValuesExtractors::Vector velocity(0);
    std::vector<FEValuesExtractors::Vector> stress(dim, FEValuesExtractors::Vector());
    // Pressure and Displacement DoFs
    const FEValuesExtractors::Scalar        pressure (dim);
    const FEValuesExtractors::Vector        displacement (dim*dim+dim+1);
    std::vector<FEValuesExtractors::Scalar> rotation(rotation_dim, FEValuesExtractors::Scalar());
    // Stress and Rotation DoFs

    for (unsigned int d=0; d<dim; ++d)
      {
        const FEValuesExtractors::Vector tmp_stress(d*dim+dim+1);
        stress[d].first_vector_component = tmp_stress.first_vector_component;
        if (dim == 2 && d == 0)
          {
            const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + dim + 1);
            rotation[d].component = tmp_rotation.component;
          } else if (dim == 3) {
            const FEValuesExtractors::Scalar tmp_rotation(dim*dim + dim + d + dim + 1);
            rotation[d].component = tmp_rotation.component;
          }
      }

    std::vector <double>                      phi_i_p(dofs_per_cell);
    std::vector<Tensor<1,dim> >               phi_i_d(dofs_per_cell);

    std::vector<double> old_pressure_values(n_q_points);
    std::vector<std::vector<Tensor<1, dim>>> old_stress(dim, std::vector<Tensor<1,dim>> (n_q_points));

    scratch_data.fe_values[pressure].get_function_values (old_solution, old_pressure_values);
    for (unsigned int s_i=0; s_i<dim; ++s_i)
      {
        scratch_data.fe_values[stress[s_i]].get_function_values(old_solution, old_stress[s_i]);
      }

    std::vector<std::vector<Tensor<1, dim>>> old_stress_values(n_q_points, std::vector<Tensor<1,dim>> (dim));
    for (unsigned int s_i=0; s_i<dim; ++s_i)
      for (unsigned int q=0; q<n_q_points; ++q)
        old_stress_values[q][s_i] = old_stress[s_i][q];


    for (unsigned int q=0; q<n_q_points; ++q)
      {
        Point<dim> point = scratch_data.fe_values.get_quadrature_points()[q];
        const double mu = scratch_data.lame.mu_value(point);
        const double lambda = scratch_data.lame.lambda_value(point);
        Tensor<2,dim> asigma = compliance_tensor_stress<dim>(old_stress_values[q], mu, lambda);
        Tensor<2,dim> apId = compliance_tensor_pressure<dim>(old_pressure_values[q], mu, lambda);
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            // Evaluate test functions
            phi_i_p[k] = scratch_data.fe_values[pressure].value (k, q);

            phi_i_d[k] = scratch_data.fe_values[displacement].value (k, q);

          }

        for (unsigned int i=0; i<dofs_per_cell; ++i){
            copy_data.cell_rhs(i) += (time_step*phi_i_p[i] * scratch_data.darcy_rhs->value(scratch_data.fe_values.get_quadrature_points()[q])
                                      + scratch_data.c_0*old_pressure_values[q] * phi_i_p[i]
                                      + scratch_data.alpha*scratch_data.alpha * trace(apId) * phi_i_p[i]
                                      + scratch_data.alpha*trace(asigma) * phi_i_p[i] )
                *scratch_data.fe_values.JxW(q);

            for (unsigned d_i=0; d_i<dim; ++d_i)
              copy_data.cell_rhs(i) += (phi_i_d[i][d_i] * scratch_data.elasticity_rhs->value(scratch_data.fe_values.get_quadrature_points()[q], d_i)) *
                  scratch_data.fe_values.JxW(q);
          }
      }


    for (unsigned int face_no=0;
         face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if (cell->at_boundary(face_no) ) // pressure part of the boundary
        {
          scratch_data.fe_face_values.reinit (cell, face_no);

          for (unsigned int q=0; q<n_face_q_points; ++q)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                copy_data.cell_rhs(i) += -(scratch_data.fe_face_values[velocity].value (i, q) *
                                           scratch_data.fe_face_values.normal_vector(q) *
                                           scratch_data.darcy_bc->value(scratch_data.fe_face_values.get_quadrature_points()[q]) *
                                           scratch_data.fe_face_values.JxW(q));
                Tensor<2,dim> sigma;
                for (unsigned int d_i=0; d_i<dim; ++d_i)
                  sigma[d_i] = scratch_data.fe_face_values[stress[d_i]].value (i, q);

                Tensor<1,dim> sigma_n = sigma * scratch_data.fe_face_values.normal_vector(q);
                for (unsigned int d_i=0; d_i<dim; ++d_i)
                  copy_data.cell_rhs(i) += ((sigma_n[d_i]*scratch_data.elasticity_bc->value(scratch_data.fe_face_values.get_quadrature_points()[q],d_i))
                                            *scratch_data.fe_face_values.JxW(q));
              }
        }
    cell->get_dof_indices (copy_data.local_dof_indices);
  }


  template <int dim>
  void MixedBiotProblem<dim>::assemble_system ()
  {
    TimerOutput::Scope t(computing_timer, "Assemble system");

    Functions::ParsedFunction<dim> *k_inv          = new Functions::ParsedFunction<dim>(dim*dim);
    Functions::ParsedFunction<dim> *mu             = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *lambda         = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *darcy_bc       = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *darcy_rhs      = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *elasticity_bc  = new Functions::ParsedFunction<dim>(dim);
    Functions::ParsedFunction<dim> *elasticity_rhs = new Functions::ParsedFunction<dim>(dim);

    const double alpha = prm.get_double("alpha");
    const double c_0 = prm.get_double("Storativity");

    prm.enter_subsection(std::string("permeability ")+ Utilities::int_to_string(dim)+std::string("D"));
    k_inv->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("lambda ")+ Utilities::int_to_string(dim)+std::string("D"));
    lambda->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("mu ")+ Utilities::int_to_string(dim)+std::string("D"));
    mu->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Darcy BC ")+ Utilities::int_to_string(dim)+std::string("D"));
    darcy_bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Elasticity BC ")+ Utilities::int_to_string(dim)+std::string("D"));
    elasticity_bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Darcy RHS ")+ Utilities::int_to_string(dim)+std::string("D"));
    darcy_rhs->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Elasticity RHS ")+ Utilities::int_to_string(dim)+std::string("D"));
    elasticity_rhs->parse_parameters(prm);
    prm.leave_subsection();

    KInverse<dim> k_inverse(prm,k_inv);
    LameCoefficients<dim> lame(prm,mu, lambda);

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MixedBiotProblem::assemble_system_cell,
                    &MixedBiotProblem::copy_local_mat_to_global,
                    CellAssemblyScratchData(fe, k_inverse, lame, darcy_bc, darcy_rhs,
                                            elasticity_bc, elasticity_rhs,
                                            c_0,alpha),
                    CellAssemblyCopyData());

    delete k_inv;
    delete mu;
    delete lambda;
    delete darcy_bc;
    delete darcy_rhs;
    delete elasticity_bc;
    delete elasticity_rhs;
  }

  template <int dim>
  void MixedBiotProblem<dim>::assemble_rhs ()
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    Functions::ParsedFunction<dim> *k_inv = new Functions::ParsedFunction<dim>(dim*dim);
    Functions::ParsedFunction<dim> *mu             = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *lambda         = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *darcy_bc       = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *darcy_rhs      = new Functions::ParsedFunction<dim>(1);
    Functions::ParsedFunction<dim> *elasticity_bc  = new Functions::ParsedFunction<dim>(dim);
    Functions::ParsedFunction<dim> *elasticity_rhs = new Functions::ParsedFunction<dim>(dim);

    const double alpha = prm.get_double("alpha");
    const double c_0 = prm.get_double("Storativity");

    prm.enter_subsection(std::string("permeability ")+ Utilities::int_to_string(dim)+std::string("D"));
    k_inv->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("lambda ")+ Utilities::int_to_string(dim)+std::string("D"));
    lambda->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("mu ")+ Utilities::int_to_string(dim)+std::string("D"));
    mu->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Darcy BC ")+ Utilities::int_to_string(dim)+std::string("D"));
    darcy_bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Elasticity BC ")+ Utilities::int_to_string(dim)+std::string("D"));
    elasticity_bc->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Darcy RHS ")+ Utilities::int_to_string(dim)+std::string("D"));
    darcy_rhs->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Elasticity RHS ")+ Utilities::int_to_string(dim)+std::string("D"));
    elasticity_rhs->parse_parameters(prm);
    prm.leave_subsection();

    KInverse<dim> k_inverse(prm,k_inv);
    LameCoefficients<dim> lame(prm,mu, lambda);

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MixedBiotProblem::assemble_rhs_cell,
                    &MixedBiotProblem::copy_local_rhs_to_global,
                    CellAssemblyScratchData(fe, k_inverse, lame, darcy_bc, darcy_rhs,
                                            elasticity_bc, elasticity_rhs,
                                            c_0,alpha),
                    CellAssemblyCopyData());

    delete k_inv;
    delete mu;
    delete lambda;
    delete darcy_bc;
    delete darcy_rhs;
    delete elasticity_bc;
    delete elasticity_rhs;
  }


  // MixedBiotProblem: Solve
  template <int dim>
  void MixedBiotProblem<dim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "Solve");

    if(time==time_step)
      A_direct.initialize(system_matrix);

    A_direct.vmult (solution, system_rhs);

//    deallog << "Solution norms: " << solution.block(0).l2_norm() << " "
//            << solution.block(1).l2_norm() << " "
//            << solution.block(2).l2_norm() << " "
//            << solution.block(3).l2_norm() << " "
//            << solution.block(4).l2_norm() << " "
//            << std::endl;
  }



  // MixedBiotProblem: Compute errors
  template <int dim>
  void MixedBiotProblem<dim>::compute_errors(const unsigned cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute errors");

    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), MixedBiotProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> pressure_mask(std::make_pair(dim,1+dim), MixedBiotProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> stress_mask(std::make_pair(dim+1,dim+1+dim*dim), MixedBiotProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> displacement_mask(std::make_pair(dim+1+dim*dim,dim+dim+1+dim*dim), MixedBiotProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> rotation_mask(dim+1+dim+dim*dim, MixedBiotProblem<dim>::total_dim);

    ExactSolution<dim> exact_solution(time, prm);
    prm.enter_subsection(std::string("Exact solution ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_val_data.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection(std::string("Exact gradient ")+ Utilities::int_to_string(dim)+std::string("D"));
    exact_solution.exact_solution_grad_val_data.parse_parameters(prm);
    prm.leave_subsection();

    exact_solution.exact_solution_val_data.set_time(time);
    exact_solution.exact_solution_grad_val_data.set_time(time);

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


    // Pressure error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_norm = cellwise_norms.l2_norm();

    // L2 in time error
    l2_l2_errors[1] +=p_l2_error;
    l2_l2_norms[1] += p_l2_norm;
    // Linf in time error
    linf_l2_errors[1] = std::max(linf_l2_errors[1],p_l2_error);
    linf_l2_norms[1] = std::max(linf_l2_norms[1],p_l2_norm);

    // Pressure error and norm at midcells
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature_super,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature_super,
                                       VectorTools::L2_norm,
                                       &pressure_mask);
    const double p_l2_mid_norm = cellwise_norms.l2_norm();

    // L2 in time error
    pressure_disp_l2_midcell_errors[0] +=p_l2_mid_error;
    pressure_disp_l2_midcell_norms[0] += p_l2_mid_norm;
    // Linf in time error
    pressure_disp_linf_midcell_errors[0] = std::max(pressure_disp_linf_midcell_errors[0],p_l2_mid_error);
    pressure_disp_linf_midcell_norms[0] = std::max(pressure_disp_linf_midcell_norms[0],p_l2_mid_norm);

    // Velocity L2 error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);
    const double u_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &velocity_mask);

    const double u_l2_norm = cellwise_norms.l2_norm();

    // L2 in time error
    l2_l2_errors[0] +=u_l2_error;
    l2_l2_norms[0] += u_l2_norm;


    // Velocity Hdiv error and seminorm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_div_errors, quadrature,
                                       VectorTools::Hdiv_seminorm,
                                       &velocity_mask);
    const double u_hd_error = cellwise_div_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_div_norms, quadrature,
                                       VectorTools::Hdiv_seminorm,
                                       &velocity_mask);
    const double u_hd_norm = cellwise_div_norms.l2_norm();

    // L2 in time error
    velocity_stress_l2_div_errors[0] +=u_hd_error;
    velocity_stress_l2_div_norms[0] += u_hd_norm;

    // Rotation error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &rotation_mask);
    const double r_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &rotation_mask);
    const double r_l2_norm = cellwise_norms.l2_norm();

    // Linf in time error
    linf_l2_errors[4] = std::max(linf_l2_errors[4],r_l2_error);
    linf_l2_norms[4] = std::max(linf_l2_norms[4],r_l2_norm);

    // Displacement error and norm
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double d_l2_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double d_l2_norm = cellwise_norms.l2_norm();

    // Linf in time error
    linf_l2_errors[3] = std::max(linf_l2_errors[3],d_l2_error);
    linf_l2_norms[3] = std::max(linf_l2_norms[3],d_l2_norm);

    // Displacement error and norm at midcells
    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature_super,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double d_l2_mid_error = cellwise_errors.l2_norm();

    VectorTools::integrate_difference (dof_handler, zerozeros, exact_solution,
                                       cellwise_norms, quadrature_super,
                                       VectorTools::L2_norm,
                                       &displacement_mask);
    const double d_l2_mid_norm = cellwise_norms.l2_norm();

    // L2 in time error
    pressure_disp_l2_midcell_errors[1] +=d_l2_mid_error;
    pressure_disp_l2_midcell_norms[1] += d_l2_mid_norm;
    // Linf in time error
    pressure_disp_linf_midcell_errors[1] = std::max(pressure_disp_linf_midcell_errors[1],d_l2_mid_error);
    pressure_disp_linf_midcell_norms[1] = std::max(pressure_disp_linf_midcell_norms[1],d_l2_mid_norm);

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

    // Linf in time error
    linf_l2_errors[2] = std::max(linf_l2_errors[2],s_l2_error);
    linf_l2_norms[2] = std::max(linf_l2_norms[2],s_l2_norm);

    // Stress Hdiv seminorm
    cellwise_errors = 0;
    cellwise_norms = 0;
    for (int i=0; i<dim; ++i){
        const ComponentSelectFunction<dim> stress_component_mask (std::make_pair(i*dim,(i+1)*dim),MixedBiotProblem<dim>::total_dim);

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

    // Linf in time error
    velocity_stress_linf_div_errors[1] = std::max(velocity_stress_linf_div_errors[1],s_hd_error);
    velocity_stress_linf_div_norms[1] = std::max(velocity_stress_linf_div_norms[1],s_hd_norm);

    double total_time = time_step*num_time_steps;
    // On the last time step compute actual errors
    if(std::abs(time-total_time)<1.0e-12){
        // Assemble convergence table
        const unsigned int n_active_cells=triangulation.n_active_cells();
        const unsigned int n_dofs=dof_handler.n_dofs();

        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", n_active_cells);
        convergence_table.add_value("dofs", n_dofs);
        convergence_table.add_value("Velocity,L2,L2", l2_l2_errors[0]/l2_l2_norms[0]);
        convergence_table.add_value("Velocity,L2,Hdiv", velocity_stress_l2_div_errors[0]/velocity_stress_l2_div_norms[0]);
        convergence_table.add_value("Pressure,L2,L2", l2_l2_errors[1]/l2_l2_norms[1]);
        convergence_table.add_value("Pressure,L2,L2mid", pressure_disp_l2_midcell_errors[0]/pressure_disp_l2_midcell_norms[0]);
        convergence_table.add_value("Pressure,Linf,L2", linf_l2_errors[1]/linf_l2_norms[1]);
        convergence_table.add_value("Pressure,Linf,L2mid", pressure_disp_linf_midcell_errors[0]/pressure_disp_linf_midcell_norms[0]);
        convergence_table.add_value("Stress,Linf,L2", linf_l2_errors[2]/linf_l2_norms[2]);
        convergence_table.add_value("Stress,Linf,Hdiv", velocity_stress_linf_div_errors[1]/velocity_stress_linf_div_norms[1]); //sx_hd_error/sx_hd_norm
        convergence_table.add_value("Displ,Linf,L2", linf_l2_errors[3]/linf_l2_norms[3]);
        convergence_table.add_value("Displ,Linf,L2mid", pressure_disp_linf_midcell_errors[1]/pressure_disp_linf_midcell_norms[1]);
        convergence_table.add_value("Displ,L2,L2mid", pressure_disp_l2_midcell_errors[1]/pressure_disp_l2_midcell_norms[1]);
        convergence_table.add_value("Rotat,Linf,L2", linf_l2_errors[4]);// /r_l2_norm);

      }
  }

  // Set errors to zero for the next refinement
  template <int dim>
  void MixedBiotProblem<dim>::set_current_errors_to_zero()
  {
    std::fill(l2_l2_errors.begin(), l2_l2_errors.end(), 0.0);
    std::fill(l2_l2_norms.begin(), l2_l2_norms.end(), 0.0);

    std::fill(linf_l2_errors.begin(), linf_l2_errors.end(), 0.0);
    std::fill(linf_l2_norms.begin(), linf_l2_norms.end(), 0.0);

    std::fill(velocity_stress_l2_div_errors.begin(), velocity_stress_l2_div_errors.end(), 0.0);
    std::fill(velocity_stress_l2_div_norms.begin(), velocity_stress_l2_div_norms.end(), 0.0);

    std::fill(velocity_stress_linf_div_errors.begin(), velocity_stress_linf_div_errors.end(), 0.0);
    std::fill(velocity_stress_linf_div_norms.begin(), velocity_stress_linf_div_norms.end(), 0.0);

    std::fill(pressure_disp_l2_midcell_errors.begin(), pressure_disp_l2_midcell_errors.end(), 0.0);
    std::fill(pressure_disp_l2_midcell_norms.begin(), pressure_disp_l2_midcell_norms.end(), 0.0);

    std::fill(pressure_disp_linf_midcell_errors.begin(), pressure_disp_linf_midcell_errors.end(), 0.0);
    std::fill(pressure_disp_linf_midcell_norms.begin(), pressure_disp_linf_midcell_norms.end(), 0.0);
  }


  // MixedBiotProblem: Output results
  template <int dim>
  void MixedBiotProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
  {
    TimerOutput::Scope t(computing_timer, "Output results");

    std::vector<std::string> solution_names;
    std::string rhs_name = "rhs";

    switch(dim)
      {
      case 2:
        solution_names.push_back ("u1");
        solution_names.push_back ("u2");
        solution_names.push_back ("p");
        solution_names.push_back ("s11");
        solution_names.push_back ("s12");
        solution_names.push_back ("s21");
        solution_names.push_back ("s22");
        solution_names.push_back ("d1");
        solution_names.push_back ("d2");
        solution_names.push_back ("r");
        break;

      case 3:
        solution_names.push_back ("u1");
        solution_names.push_back ("u2");
        solution_names.push_back ("u3");
        solution_names.push_back ("p");
        solution_names.push_back ("s11");
        solution_names.push_back ("s12");
        solution_names.push_back ("s13");
        solution_names.push_back ("s21");
        solution_names.push_back ("s22");
        solution_names.push_back ("s23");
        solution_names.push_back ("s31");
        solution_names.push_back ("s32");
        solution_names.push_back ("s33");
        solution_names.push_back ("d1");
        solution_names.push_back ("d2");
        solution_names.push_back ("d3");
        solution_names.push_back ("r1");
        solution_names.push_back ("r2");
        solution_names.push_back ("r3");
        break;

      default:
        Assert(false, ExcNotImplemented());
      }


    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    for(unsigned int i=dim+2;i<total_dim;i++)
      data_component_interpretation.push_back (DataComponentInterpretation::component_is_part_of_vector);

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


    int tmp = time/time_step;
    //  std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(cycle) + ".vtk");
    std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(tmp) + ".vtk");
    data_out.write_vtk (output);
    double total_time = time_step*num_time_steps;
    if (cycle == refine-1 && std::abs(time-total_time)<1.0e-12){
        convergence_table.set_precision("Velocity,L2,L2", 3);
        convergence_table.set_precision("Velocity,L2,Hdiv", 3);
        convergence_table.set_precision("Pressure,L2,L2", 3);
        convergence_table.set_precision("Pressure,L2,L2mid", 3);
        convergence_table.set_precision("Pressure,Linf,L2", 3);
        convergence_table.set_precision("Pressure,Linf,L2mid", 3);
        convergence_table.set_scientific("Velocity,L2,L2", true);
        convergence_table.set_scientific("Velocity,L2,Hdiv", true);
        convergence_table.set_scientific("Pressure,L2,L2", true);
        convergence_table.set_scientific("Pressure,L2,L2mid", true);
        convergence_table.set_scientific("Pressure,Linf,L2", true);
        convergence_table.set_scientific("Pressure,Linf,L2mid", true);
        convergence_table.set_precision("Stress,Linf,L2", 3);
        convergence_table.set_precision("Stress,Linf,Hdiv", 3);
        convergence_table.set_precision("Displ,Linf,L2", 3);
        convergence_table.set_precision("Displ,Linf,L2mid", 3);
        convergence_table.set_precision("Displ,L2,L2mid", 3);
        convergence_table.set_precision("Rotat,Linf,L2", 3);
        convergence_table.set_scientific("Stress,Linf,L2", true);
        convergence_table.set_scientific("Stress,Linf,Hdiv", true);
        convergence_table.set_scientific("Displ,Linf,L2", true);
        convergence_table.set_scientific("Displ,Linf,L2mid", true);
        convergence_table.set_scientific("Displ,L2,L2mid", true);
        convergence_table.set_scientific("Rotat,Linf,L2", true);
        convergence_table.set_tex_caption("cells", "\\# cells");
        convergence_table.set_tex_caption("dofs", "\\# dofs");
        convergence_table.set_tex_caption("Velocity,L2,L2", "$ \\|\\u - \\u_h\\|_{L^2(L^2)} $");
        convergence_table.set_tex_caption("Velocity,L2,Hdiv", "$ \\|\\nabla\\cdot(\\u - \\u_h)\\|_{L^2(L^2)} $");
        convergence_table.set_tex_caption("Pressure,L2,L2", "$ \\|p - p_h\\|_{L^2(L^2)} $");
        convergence_table.set_tex_caption("Pressure,L2,L2mid", "$ \\|Qp - p_h\\|_{L^2(L^2)} $");
        convergence_table.set_tex_caption("Pressure,Linf,L2", "$ \\|p - p_h\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Pressure,Linf,L2mid", "$ \\|Qp - p_h\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Stress,Linf,L2", "$ \\|\\sigma - \\sigma_h\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Stress,Linf,Hdiv", "$ \\|\\nabla\\cdot(\\sigma - \\sigma_h)\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Displ,Linf,L2", "$ \\|\\bbeta - \\bbeta_h\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Displ,Linf,L2mid", "$ \\|Q\\bbeta - \\bbeta_h\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_caption("Displ,L2,L2mid", "$ \\|Q\\bbeta - \\bbeta_h\\|_{L^2(L^2)} $");
        convergence_table.set_tex_caption("Rotat,Linf,L2", "$ \\|r - r_h\\|_{L^{\\infty}(L^2)} $");
        convergence_table.set_tex_format("cells", "r");
        convergence_table.set_tex_format("dofs", "r");

        convergence_table.evaluate_convergence_rates("Velocity,L2,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Velocity,L2,Hdiv", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Pressure,L2,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Pressure,L2,L2mid", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Pressure,Linf,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Pressure,Linf,L2mid", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Stress,Linf,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Stress,Linf,Hdiv", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Displ,Linf,L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Displ,Linf,L2mid", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Displ,L2,L2mid", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates("Rotat,Linf,L2", ConvergenceTable::reduction_rate_log2);

        std::ofstream error_table_file("error" + std::to_string(dim) + "d.tex");


        convergence_table.write_text(std::cout);
        convergence_table.write_tex(error_table_file);
      }
  }


  // MixedBiotProblem: run
  template <int dim>
  void MixedBiotProblem<dim>::run(const unsigned int refine, const unsigned int grid)
  {
    Functions::ParsedFunction<dim> initial_condition_data(10);
    prm.enter_subsection(std::string("Initial conditions ")+ Utilities::int_to_string(dim)+std::string("D"));
    initial_condition_data.parse_parameters(prm);
    prm.leave_subsection();
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
              }
            else {
                GridGenerator::hyper_cube (triangulation, 0, 1);
                triangulation.refine_global(2);
                // GridTools::transform(&grid_transform<dim>, triangulation);
              }

          } else {
            triangulation.refine_global(1);
          }


        make_grid_and_dofs();

        {
          ConstraintMatrix constraints;
          constraints.close();
          VectorTools::project (dof_handler,
                                constraints,
                                QGauss<dim>(degree+4),
                                initial_condition_data,
                                old_solution);

        }

        assemble_system ();
        for(unsigned int i=0;i<num_time_steps;i++){
            time+=time_step;
            assemble_rhs ();
            solve ();

            compute_errors(cycle);

            old_solution = solution;
            system_rhs = 0;
            output_results (cycle, refine);

          }

        set_current_errors_to_zero();
        time = 0.0;
        computing_timer.print_summary();
        computing_timer.reset();
      }
  }

  // Explicit instantiation
  template class MixedBiotProblem<2>;
  template class MixedBiotProblem<3>;

}



