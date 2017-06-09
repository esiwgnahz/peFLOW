// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/iterative_inverse.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/work_stream.h>

#include <fstream>

#include "../inc/darcy_data.h"
#include "../inc/darcy_mfe.h"
#include "../inc/utilities.h"

namespace darcy
{
    using namespace dealii;
    using namespace utilities;

    // MixedDarcyProblem: class constructor
    template <int dim>
    MixedDarcyProblem<dim>::MixedDarcyProblem (const unsigned int degree, ParameterHandler &param)
            :
            degree(degree),
            prm(param),
            fe(FE_RaviartThomas<dim>(degree-1), 1, 
               FE_DGQ<dim>(degree-1), 1), 
            dof_handler(triangulation),
            computing_timer(std::cout, TimerOutput::summary,
                            TimerOutput::wall_times)
    {}


    // MixedDarcyProblem: make grid and DoFs
    template <int dim>
    void MixedDarcyProblem<dim>::make_grid_and_dofs()
    {
        TimerOutput::Scope t(computing_timer, "Make grid and DOFs");
        system_matrix.clear();

        dof_handler.distribute_dofs(fe);

        DoFRenumbering::component_wise (dof_handler);

        std::vector<types::global_dof_index> dofs_per_component (dim+ 1);
        DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
        const unsigned int n_u = dofs_per_component[0],
                n_p = dofs_per_component[dim];


        BlockDynamicSparsityPattern dsp(2, 2);
        dsp.block(0, 0).reinit (n_u, n_u);
        dsp.block(1, 0).reinit (n_p, n_u);
        dsp.block(0, 1).reinit (n_u, n_p);
        dsp.block(1, 1).reinit (n_p, n_p);
        dsp.collect_sizes ();
        DoFTools::make_sparsity_pattern (dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit (sparsity_pattern);
        solution.reinit (2);
        solution.block(0).reinit (n_u);
        solution.block(1).reinit (n_p);
        solution.collect_sizes ();
        system_rhs.reinit (2);
        system_rhs.block(0).reinit (n_u);
        system_rhs.block(1).reinit (n_p);
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

        {
            const FEValuesExtractors::Vector velocity(0);

            std::map<types::global_dof_index,double> boundary_values;

        }
    }


    // Scratch data for multithreading
    template <int dim>
    MixedDarcyProblem<dim>::CellAssemblyScratchData::
    CellAssemblyScratchData (const FiniteElement<dim> &fe,
                             const Quadrature<dim>    &quadrature,
                             const Quadrature<dim-1>  &face_quadrature,
                             const KInverse<dim> &k_data,
                             Functions::ParsedFunction<dim> *bc_data,
                             Functions::ParsedFunction<dim> *rhs_data)
            :
            fe_values (fe,
                       quadrature,
                       update_values   | update_gradients |
                       update_quadrature_points | update_JxW_values),
            fe_face_values (fe,
                            face_quadrature,
                            update_values     | update_quadrature_points   |
                            update_JxW_values | update_normal_vectors),
            K_inv(k_data),
            bc(bc_data),
            rhs(rhs_data)
    {}


    template <int dim>
    MixedDarcyProblem<dim>::CellAssemblyScratchData::
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
            bc(scratch_data.bc),
            rhs(scratch_data.rhs)
    {}


    // Copy local contributions to global system
    template <int dim>
    void MixedDarcyProblem<dim>::copy_local_to_global (const CellAssemblyCopyData &copy_data)
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
    void MixedDarcyProblem<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                       CellAssemblyScratchData                                   &scratch_data,
                                                       CellAssemblyCopyData                                      &copy_data)
    {
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

        copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        copy_data.cell_rhs.reinit (dofs_per_cell);
        copy_data.local_dof_indices.resize(dofs_per_cell);

        scratch_data.fe_values.reinit (cell);

        // Velocity  and Pressure DoFs vectors
        const FEValuesExtractors::Vector velocity(0);
        const FEValuesExtractors::Scalar pressure (dim);


        std::vector<Tensor<2,dim>>             k_inverse_values (n_q_points);
        scratch_data.K_inv.value_list (scratch_data.fe_values.get_quadrature_points(), k_inverse_values);


        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const Tensor<1,dim> phi_i_u     = scratch_data.fe_values[velocity].value (i, q);
                const double        div_phi_i_u = scratch_data.fe_values[velocity].divergence (i, q);
                const double        phi_i_p     = scratch_data.fe_values[pressure].value (i, q);
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const Tensor<1,dim> phi_j_u     = scratch_data.fe_values[velocity].value (j, q);
                    const double        div_phi_j_u = scratch_data.fe_values[velocity].divergence (j, q);
                    const double        phi_j_p     = scratch_data.fe_values[pressure].value (j, q);
                    copy_data.cell_matrix(i,j) += (phi_i_u * k_inverse_values[q] * phi_j_u
                                                   - div_phi_i_u * phi_j_p
                                                   - phi_i_p * div_phi_j_u)
                                                  * scratch_data.fe_values.JxW(q);
                }

                copy_data.cell_rhs(i) += -phi_i_p *scratch_data.rhs->value(scratch_data.fe_values.get_quadrature_points()[q]) 
                                            *scratch_data.fe_values.JxW(q);
            }
        }

        for (unsigned int face_no=0;
             face_no<GeometryInfo<dim>::faces_per_cell;
             ++face_no)
            if ((cell->at_boundary(face_no)) ) // && (cell->face(face_no)->boundary_id() != 1)
            {
                scratch_data.fe_face_values.reinit (cell, face_no);

                for (unsigned int q=0; q<n_face_q_points; ++q)
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                        copy_data.cell_rhs(i) += -(scratch_data.fe_face_values[velocity].value (i, q) *
                                                   scratch_data.fe_face_values.normal_vector(q) *
                                                   scratch_data.bc->value(scratch_data.fe_face_values.get_quadrature_points()[q]) *
                                                   scratch_data.fe_face_values.JxW(q));

                    }
            }
        cell->get_dof_indices (copy_data.local_dof_indices);
    }


    template <int dim>
    void MixedDarcyProblem<dim>::assemble_system ()
    {
      Functions::ParsedFunction<dim> *k_inv    = new Functions::ParsedFunction<dim>(dim*dim);
      Functions::ParsedFunction<dim> *bc       = new Functions::ParsedFunction<dim>(1);
      Functions::ParsedFunction<dim> *rhs      = new Functions::ParsedFunction<dim>(1);

      prm.enter_subsection(std::string("permeability ") + Utilities::int_to_string(dim)+std::string("D"));
      k_inv->parse_parameters(prm);
      prm.leave_subsection();

      prm.enter_subsection("BC " + Utilities::int_to_string(dim)+std::string("D"));
      bc->parse_parameters(prm);
      prm.leave_subsection();

      prm.enter_subsection("RHS " + Utilities::int_to_string(dim)+std::string("D"));
      rhs->parse_parameters(prm);
      prm.leave_subsection();

      TimerOutput::Scope t(computing_timer, "Assemble system");
      QGauss<dim> quad(2*(degree+1)+1);
      QGauss<dim-1> face_quad(2*(degree+1)+1);

      KInverse<dim> k_inverse(prm,k_inv);

      WorkStream::run(dof_handler.begin_active(),
                      dof_handler.end(),
                      *this,
                      &MixedDarcyProblem::assemble_system_cell,
                      &MixedDarcyProblem::copy_local_to_global,
                      CellAssemblyScratchData(fe,quad,face_quad, k_inverse, bc, rhs),
                      CellAssemblyCopyData());
    }


    // Schur complement
    class SchurComplement : public Subscriptor
    {
    public:
        SchurComplement (const BlockSparseMatrix<double> &A,
                         const IterativeInverse<Vector<double> > &Minv);
        void vmult (Vector<double>       &dst,
                    const Vector<double> &src) const;
    private:
        const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
        const SmartPointer<const IterativeInverse<Vector<double> > > m_inverse;
        mutable Vector<double> tmp1, tmp2;
    };
    SchurComplement::SchurComplement (const BlockSparseMatrix<double> &A,
                                      const IterativeInverse<Vector<double> > &Minv)
            :
            system_matrix (&A),
            m_inverse (&Minv),
            tmp1 (A.block(0,0).m()),
            tmp2 (A.block(0,0).m())
    {}
    void SchurComplement::vmult (Vector<double>       &dst,
                                 const Vector<double> &src) const
    {
        system_matrix->block(0,1).vmult (tmp1, src);
        m_inverse->vmult (tmp2, tmp1);
        system_matrix->block(1,0).vmult (dst, tmp2);
    }
    class ApproximateSchurComplement : public Subscriptor
    {
    public:
        ApproximateSchurComplement (const BlockSparseMatrix<double> &A);
        void vmult (Vector<double>       &dst,
                    const Vector<double> &src) const;
        void Tvmult (Vector<double>       &dst,
                     const Vector<double> &src) const;
    private:
        const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
        mutable Vector<double> tmp1, tmp2;
    };
    ApproximateSchurComplement::ApproximateSchurComplement (const BlockSparseMatrix<double> &A)
            :
            system_matrix (&A),
            tmp1 (A.block(0,0).m()),
            tmp2 (A.block(0,0).m())
    {}
    void ApproximateSchurComplement::vmult (Vector<double>       &dst,
                                            const Vector<double> &src) const
    {
        system_matrix->block(0,1).vmult (tmp1, src);
        system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
        system_matrix->block(1,0).vmult (dst, tmp2);
    }
    void ApproximateSchurComplement::Tvmult (Vector<double>       &dst,
                                             const Vector<double> &src) const
    {
        vmult (dst, src);
    }


    // MixedDarcyProblem: Solve
    template <int dim>
    void MixedDarcyProblem<dim>::solve ()
    {
        TimerOutput::Scope t(computing_timer, "Solve (Preconditioned CG)");

        PreconditionIdentity identity;
        IterativeInverse<Vector<double> > m_inverse;
        m_inverse.initialize(system_matrix.block(0,0), identity);
        m_inverse.solver.select("cg");
        static ReductionControl inner_control(10000, 0., 1.e-10);
        m_inverse.solver.set_control(inner_control);
        Vector<double> tmp (solution.block(0).size());
        {
          Vector<double> schur_rhs (solution.block(1).size());
          m_inverse.vmult (tmp, system_rhs.block(0));
          system_matrix.block(1,0).vmult (schur_rhs, tmp);
          schur_rhs -= system_rhs.block(1);
          SchurComplement
          schur_complement (system_matrix, m_inverse);
          ApproximateSchurComplement
          approximate_schur_complement (system_matrix);

          IterativeInverse<Vector<double> >
          preconditioner;
          preconditioner.initialize(approximate_schur_complement, identity);
          preconditioner.solver.select("cg");
          preconditioner.solver.set_control(inner_control);

          SolverControl solver_control (10*solution.block(1).size(),
                                        1e-10*schur_rhs.l2_norm());
          SolverCG<>    cg (solver_control);
          cg.solve (schur_complement, solution.block(1), schur_rhs,
                    preconditioner);
        }
        {
          system_matrix.block(0,1).vmult (tmp, solution.block(1));
          tmp *= -1;
          tmp += system_rhs.block(0);
          m_inverse.vmult (solution.block(0), tmp);
        }
    }


    // MixedDarcyProblem: Compute errors
    template <int dim>
    void MixedDarcyProblem<dim>::compute_errors(const unsigned cycle)
    {
      TimerOutput::Scope t(computing_timer, "Compute errors");

      const ComponentSelectFunction<dim> pressure_mask(dim, dim+1);
      const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);

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
      QGauss<dim>   quadrature_super(degree);

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


      // Assemble convergence table
      const unsigned int n_active_cells=triangulation.n_active_cells();
      const unsigned int n_dofs=dof_handler.n_dofs();

      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", n_active_cells);
      convergence_table.add_value("dofs", n_dofs);
      convergence_table.add_value("Velocity,L2", u_l2_error/u_l2_norm);
      convergence_table.add_value("Velocity,Hdiv", u_hd_error/u_hd_norm);
      convergence_table.add_value("Pressure,L2", p_l2_error/p_l2_norm);
      convergence_table.add_value("Pressure,L2mid", p_l2_mid_error/p_l2_mid_norm);
    }


    // MixedDarcyProblem: Output results
    template <int dim>
    void MixedDarcyProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
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
            break;
          case 3:
            solution_names.push_back ("u1");
            solution_names.push_back ("u2");
            solution_names.push_back ("u3");
            solution_names.push_back ("p");
            break;
          default:
            Assert(false, ExcNotImplemented());
      }


      DataOut<dim> data_out;
      data_out.attach_dof_handler (dof_handler);
      data_out.add_data_vector (solution, solution_names, DataOut<dim>::type_dof_data);

      data_out.build_patches ();

      std::ofstream output ("solution" + std::to_string(dim) + "d-" + std::to_string(cycle) + ".vtk");
      data_out.write_vtk (output);

      convergence_table.set_precision("Velocity,L2", 3);
      convergence_table.set_precision("Velocity,Hdiv", 3);
      convergence_table.set_precision("Pressure,L2", 3);
      convergence_table.set_precision("Pressure,L2mid", 3);
      convergence_table.set_scientific("Velocity,L2", true);
      convergence_table.set_scientific("Velocity,Hdiv", true);
      convergence_table.set_scientific("Pressure,L2", true);
      convergence_table.set_scientific("Pressure,L2mid", true);
      convergence_table.set_tex_caption("cells", "\\# cells");
      convergence_table.set_tex_caption("dofs", "\\# dofs");
      convergence_table.set_tex_caption("Velocity,L2", "$ \\|\\u - \\u_h\\|_{L^2} $");
      convergence_table.set_tex_caption("Velocity,Hdiv", "$ \\|\\nabla\\cdot(\\u - \\u_h)\\|_{L^2} $");
      convergence_table.set_tex_caption("Pressure,L2", "$ \\|p - p_h\\|_{L^2} $");
      convergence_table.set_tex_caption("Pressure,L2mid", "$ \\|Qp - p_h\\|_{L^2} $");
      convergence_table.set_tex_format("cells", "r");
      convergence_table.set_tex_format("dofs", "r");

      convergence_table.evaluate_convergence_rates("Velocity,L2", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates("Velocity,Hdiv", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates("Pressure,L2", ConvergenceTable::reduction_rate_log2);
      convergence_table.evaluate_convergence_rates("Pressure,L2mid", ConvergenceTable::reduction_rate_log2);

      std::ofstream error_table_file("error" + std::to_string(dim) + "d.tex");

      if (cycle == refine-1){
          convergence_table.write_text(std::cout);
          convergence_table.write_tex(error_table_file);
      }
    }


    // MixedDarcyProblem: run
    template <int dim>
    void MixedDarcyProblem<dim>::run(const unsigned int refine, const unsigned int grid)
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

  // Explicit instantiation
  template class MixedDarcyProblem<2>;
  template class MixedDarcyProblem<3>;
}


