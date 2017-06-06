//
// Created by eldar on 6/4/17.
//

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
#include <deal.II/fe/fe_rt_bubbles.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/work_stream.h>

#include <fstream>

#include "../inc/elasticity_data.h"
#include "../inc/elasticity_msmfe.h"
#include "../inc/utilities.h"

namespace elasticity
{
  using namespace dealii;
  using namespace utilities;

  // MultipointMixedElasticityProblem: class constructor
  template <int dim>
  MultipointMixedElasticityProblem<dim>::MultipointMixedElasticityProblem (const unsigned int degree)
          :
          degree(degree),
          total_dim(dim*dim + dim + static_cast<int>(0.5*dim*(dim-1))),
          fe(FE_RT_Bubbles<dim>(degree), dim,
             FE_DGQ<dim>(degree-1), dim,
             FE_Q<dim>(degree), static_cast<int>(0.5*dim*(dim-1))),
          dof_handler(triangulation),
          computing_timer(std::cout, TimerOutput::summary,
                          TimerOutput::wall_times)
  {}


  template <int dim>
  void MultipointMixedElasticityProblem<dim>::reset_data_structures ()
  {
    displacement_indices.clear();
    stress_indices.clear();
    rotation_indices.clear();
    stress_rhs.clear();
    rotation_rhs.clear();
    A_inverse.clear();
    CAC_inverse.clear();
    rotation_matrix.clear();
    displacement_matrix.clear();
    vertex_matrix.clear();
    vertex_rhs.clear();
  }


// Scratch data for multithreading
  template <int dim>
  MultipointMixedElasticityProblem<dim>::VertexAssemblyScratchData::
  VertexAssemblyScratchData (const FiniteElement<dim> &fe,
                             const Triangulation<dim> &tria,
                             const Quadrature<dim> &quad,
                             const Quadrature<dim-1> &f_quad)
          :
          fe_values (fe,
                     quad,
                     update_values   | update_gradients |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (fe,
                          f_quad,
                          update_values     | update_quadrature_points   |
                          update_JxW_values | update_normal_vectors),
          num_cells(tria.n_active_cells())
  {
    n_faces_at_vertex.resize(tria.n_vertices(), 0);
    typename Triangulation<dim>::active_face_iterator face = tria.begin_active_face(), endf = tria.end_face();

    for (; face != endf; ++face)
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
        n_faces_at_vertex[face->vertex_index(v)] += 1;
  }

  template <int dim>
  MultipointMixedElasticityProblem<dim>::VertexAssemblyScratchData::
  VertexAssemblyScratchData (const VertexAssemblyScratchData &scratch_data)
          :
          fe_values (scratch_data.fe_values.get_fe(),
                     scratch_data.fe_values.get_quadrature(),
                     update_values   | update_gradients |
                     update_quadrature_points | update_JxW_values),
          fe_face_values (scratch_data.fe_face_values.get_fe(),
                          scratch_data.fe_face_values.get_quadrature(),
                          update_values     | update_quadrature_points   |
                          update_JxW_values | update_normal_vectors),
          n_faces_at_vertex(scratch_data.n_faces_at_vertex),
          num_cells(scratch_data.num_cells)
  {}


  template <int dim>
  void MultipointMixedElasticityProblem<dim>::copy_cell_to_vertex (const VertexAssemblyCopyData &copy_data)
  {
    for (auto m : copy_data.cell_mat) {
      for (auto p : m.second)
        vertex_matrix[m.first][p.first] += p.second;

      for (auto p : copy_data.cell_vec.at(m.first))
        vertex_rhs[m.first][p.first] += p.second;

      for (auto p : copy_data.local_displ_indices.at(m.first))
        displacement_indices[m.first].insert(p);

      for (auto p : copy_data.local_stress_indices.at(m.first))
        stress_indices[m.first].insert(p);

      for (auto p : copy_data.local_rotation_indices.at(m.first))
        rotation_indices[m.first].insert(p);
    }
  }

// Function to assemble on a cell
  template <int dim>
  void MultipointMixedElasticityProblem<dim>::assemble_system_cell (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                    VertexAssemblyScratchData                         &scratch_data,
                                                                    VertexAssemblyCopyData                            &copy_data)
  {
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));

    copy_data.cell_mat.clear();
    copy_data.cell_vec.clear();
    copy_data.local_stress_indices.clear();
    copy_data.local_displ_indices.clear();
    copy_data.local_rotation_indices.clear();

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_face_q_points = scratch_data.fe_face_values.get_quadrature().size();

    copy_data.local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices (copy_data.local_dof_indices);

    scratch_data.fe_values.reinit (cell);

    const LameFirstParameter<dim> lambda;
    const LameSecondParameter<dim> mu;

    const RightHandSide<dim>              right_hand_side;
    const DisplacementBoundaryValues<dim> displacement_boundary_values;

    std::vector<double> lambda_values (n_q_points);
    std::vector<double> mu_values (n_q_points);

    std::vector<Vector<double>> rhs_values (n_q_points, Vector<double>(dim));
    std::vector<Vector<double>> boundary_values (n_face_q_points, Vector<double>(dim));

    // Stress and rotation DoFs vectors
    std::vector<FEValuesExtractors::Vector> stresses(dim, FEValuesExtractors::Vector());
    std::vector<FEValuesExtractors::Scalar> rotations(rotation_dim, FEValuesExtractors::Scalar());

    // Displacement DoFs
    const FEValuesExtractors::Vector displacements (dim*dim);

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

    lambda.value_list (scratch_data.fe_values.get_quadrature_points(), lambda_values);
    mu.value_list     (scratch_data.fe_values.get_quadrature_points(), mu_values);

    right_hand_side.vector_value_list (scratch_data.fe_values.get_quadrature_points(), rhs_values);

    unsigned int n_stress = dim*dim*pow(degree+1,dim);
    unsigned int n_rotation = rotation_dim*pow(degree+1,dim);
    std::unordered_map<unsigned int, std::unordered_map<unsigned int, double>> div_map;

    std::vector<Tensor<1,dim>> div_phi_i_s(dofs_per_cell);
    std::vector<Tensor<1,dim>> phi_i_u(dofs_per_cell);
    std::vector<std::vector<Tensor<1,dim>>> phi_i_s(dofs_per_cell, std::vector<Tensor<1,dim>> (dim));
    std::vector<Tensor<1,rotation_dim>> phi_i_p(dofs_per_cell);

    // Assemble the div terms
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      Point<dim> p = scratch_data.fe_values.quadrature_point(q);

      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        // Evaluate test functions
        for (unsigned int s_i=0; s_i<dim; ++s_i)
          div_phi_i_s[k][s_i] = scratch_data.fe_values[stresses[s_i]].divergence (k, q);

        phi_i_u[k] = scratch_data.fe_values[displacements].value (k, q);
      }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=n_stress; j<dofs_per_cell; ++j)
        {
          double div_term = (scalar_product(phi_i_u[i], div_phi_i_s[j])
                             + scalar_product(phi_i_u[j], div_phi_i_s[i])) * scratch_data.fe_values.JxW(q);

          if (fabs(div_term) > 1.e-12)
            div_map[i][j] += div_term;
        }

        double source_term = 0;
        bool displ_flag = false;
        for (unsigned d_i=0; d_i<dim; ++d_i) {
          source_term += -(phi_i_u[i][d_i] * rhs_values[q][d_i]) * scratch_data.fe_values.JxW(q);
          if (!displ_flag && (phi_i_u[i][d_i]) > 1.e-12)
            displ_flag = true;
        }

        if (displ_flag || fabs(source_term) > 1.e-12)
          copy_data.cell_vec[p][copy_data.local_dof_indices[i]] += source_term;
      }
    }

    // Assemble coercive and asymmetry terms and incorporate div terms
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

        for (unsigned int r_i=0; r_i<rotation_dim; ++r_i)
          phi_i_p[k][r_i] = scratch_data.fe_values[rotations[r_i]].value (k, q);
      }

      std::set<types::global_dof_index> stress_indices;
      Point<dim> p = scratch_data.fe_values.quadrature_point(q);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        bool rotation_flag = false;
        Tensor<2,dim> asigma = compliance_tensor(phi_i_s[i], lambda_values[q], mu_values[q]);
        for (unsigned int j=i; j<dofs_per_cell; ++j)
        {
          Tensor<2,dim> sigma = make_tensor(phi_i_s[j]);
          double mass_term = (scalar_product(asigma, sigma)) * scratch_data.fe_values.JxW(q);
          double sr_term = (scalar_product(phi_i_p[i], make_asymmetry_tensor(phi_i_s[j]))
                            + scalar_product(phi_i_p[j], make_asymmetry_tensor(phi_i_s[i])))
                           * scratch_data.fe_values.JxW(q);


          if (fabs(mass_term) > 1.e-12)
          {
            copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j])] +=
                    mass_term;
            stress_indices.insert(i);
            copy_data.local_stress_indices[p].insert(copy_data.local_dof_indices[j]);
          }

          if (fabs(scalar_product(phi_i_p[i], phi_i_p[i])) > 1.e-12)
            rotation_flag = true;

          if (fabs(sr_term) > 1.e-12 && copy_data.local_dof_indices[i] > n_s)
          {
            copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[j], copy_data.local_dof_indices[i])] +=
                    sr_term;
            copy_data.local_rotation_indices[p].insert(copy_data.local_dof_indices[i]);
          }
          else if (fabs(sr_term) > 1.e-12 && copy_data.local_dof_indices[j] > n_s)
          {
            copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j])] +=
                    sr_term;
            copy_data.local_rotation_indices[p].insert(copy_data.local_dof_indices[j]);
          }
        }
      }

      for (auto i : stress_indices)
        for (auto el : div_map[i])
          if (fabs(el.second) > 1.e-12) {
            copy_data.cell_mat[p][std::make_pair(copy_data.local_dof_indices[i],
                                                 copy_data.local_dof_indices[el.first])] += el.second;
            copy_data.local_displ_indices[p].insert(copy_data.local_dof_indices[el.first]);
          }
    }

    std::map<types::global_dof_index,double> displ_bc;
    for (unsigned int face_no=0;
         face_no<GeometryInfo<dim>::faces_per_cell;
         ++face_no)
      if ( (cell->at_boundary(face_no)) ) // && (cell->face(face_no)->boundary_id() != 1)
      {
        scratch_data.fe_face_values.reinit (cell, face_no);
        displacement_boundary_values.vector_value_list (scratch_data.fe_face_values.get_quadrature_points(), boundary_values);

        for (unsigned int q=0; q<n_face_q_points; ++q)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            Tensor<2,dim> sigma;
            for (unsigned int d_i=0; d_i<dim; ++d_i)
              sigma[d_i] = scratch_data.fe_face_values[stresses[d_i]].value (i, q);

            Tensor<1,dim> sigma_n = sigma * scratch_data.fe_face_values.normal_vector(q);
            double tmp = 0;
            for (unsigned int d_i=0; d_i<dim; ++d_i)
              tmp += ((sigma_n[d_i]*boundary_values[q][d_i])*scratch_data.fe_face_values.JxW(q));

            if (fabs(tmp) > 1.e-12) {
              displ_bc[copy_data.local_dof_indices[i]] += tmp;
            }
          }
      }

    for (auto m : copy_data.cell_vec)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        if (fabs(displ_bc[copy_data.local_dof_indices[i]]) > 1.e-12)
          copy_data.cell_vec[m.first][copy_data.local_dof_indices[i]] += displ_bc[copy_data.local_dof_indices[i]];

  }


  template <int dim>
  void MultipointMixedElasticityProblem<dim>::vertex_assembly()
  {
    TimerOutput::Scope t(computing_timer, "Vertex assembly");
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise (dof_handler);
    std::vector<types::global_dof_index> dofs_per_component (dim*dim+dim+rotation_dim);
    DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    n_s = 0, n_u = 0, n_p = 0;
    for (int i=0; i<dim; ++i){
      n_s += dofs_per_component[i*dim];
      n_u += dofs_per_component[dim*dim+i];
      if (dim == 2)
        n_p = dofs_per_component[dim*dim+dim];
      else if (dim == 3)
        n_p += dofs_per_component[dim*dim+dim+i];
    }



    displ_rhs.reinit(n_u);
    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    *this,
                    &MultipointMixedElasticityProblem::assemble_system_cell,
                    &MultipointMixedElasticityProblem::copy_cell_to_vertex,
                    VertexAssemblyScratchData(fe, triangulation,quad,face_quad),
                    VertexAssemblyCopyData());
  }

  template <int dim>
  void MultipointMixedElasticityProblem<dim>::make_cell_centered_sp()
  {
    TimerOutput::Scope t(computing_timer, "Make sparsity pattern");
    DynamicSparsityPattern dsp(n_u, n_u);

    std::set<types::global_dof_index>::iterator ui_it, uj_it;

    unsigned int i, j;
    for (auto el : vertex_matrix)
      for (ui_it = displacement_indices[el.first].begin(), i = 0;
           ui_it != displacement_indices[el.first].end();
           ++ui_it, ++i)
        for (uj_it = ui_it, j = 0;
             uj_it != displacement_indices[el.first].end();
             ++uj_it, ++j)
          dsp.add(*ui_it - n_s, *uj_it - n_s);


    dsp.symmetrize();
    cell_centered_sp.copy_from(dsp);
    displ_system_matrix.reinit (cell_centered_sp);
  }


  template <int dim>
  void MultipointMixedElasticityProblem<dim>::vertex_elimination(const typename MapPointMatrix<dim>::iterator &n_it,
                                                            VertexAssemblyScratchData &scratch_data,
                                                            VertexEliminationCopyData &copy_data)
  {
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));

    // Matrix for each component
    unsigned int n_edges = stress_indices.at((*n_it).first).size();
    unsigned int n_cells = displacement_indices.at((*n_it).first).size();

    FullMatrix<double> stress_matrix(n_edges,n_edges);
    copy_data.rotation_matrix.reinit(n_edges,rotation_dim);
    copy_data.displacement_matrix.reinit(n_edges,n_cells);

    //RHS vector for each component
    copy_data.stress_rhs.reinit(n_edges);
    Vector<double> displacement_rhs(n_cells);

    std::set<types::global_dof_index>::iterator si_it, sj_it, u_it, p_it;
    unsigned int i, j, k;
    for(si_it = stress_indices.at((*n_it).first).begin(), i = 0; si_it != stress_indices.at((*n_it).first).end(); ++si_it, ++i)
    {
      for(sj_it = stress_indices.at((*n_it).first).begin(), j = 0; sj_it != stress_indices.at((*n_it).first).end(); ++sj_it, ++j) {
        stress_matrix.add(i, j, vertex_matrix[(*n_it).first][std::make_pair(*si_it, *sj_it)]);
        if (j != i)
          stress_matrix.add(j, i, vertex_matrix[(*n_it).first][std::make_pair(*si_it, *sj_it)]);
      }

      for(u_it = displacement_indices.at((*n_it).first).begin(), j = 0; u_it != displacement_indices.at((*n_it).first).end(); ++u_it, ++j)
        copy_data.displacement_matrix.add(i, j, vertex_matrix[(*n_it).first][std::make_pair(*si_it, *u_it)]);

      for(p_it = rotation_indices.at((*n_it).first).begin(), k = 0; p_it != rotation_indices.at((*n_it).first).end(); ++p_it, ++k)
        copy_data.rotation_matrix.add(i, k, vertex_matrix[(*n_it).first][std::make_pair(*si_it, *p_it)]);

      copy_data.stress_rhs(i) += vertex_rhs.at((*n_it).first)[*si_it];
    }

    for (u_it = displacement_indices.at((*n_it).first).begin(), i = 0; u_it != displacement_indices.at((*n_it).first).end(); ++u_it, ++i)
      displacement_rhs(i) += vertex_rhs.at((*n_it).first)[*u_it];

    // Create and invert A matrix
    copy_data.Ainverse.reinit(n_edges,n_edges);
    invert_spd(stress_matrix, copy_data.Ainverse);

    // Create CtAiC and invert it
    copy_data.CACinverse.reinit(rotation_dim,rotation_dim);
    copy_data.CACinverse.triple_product(copy_data.Ainverse, copy_data.rotation_matrix, copy_data.rotation_matrix, true, false);

    for (unsigned int i=0; i<rotation_dim; ++i)
      copy_data.CACinverse(i,i) = 1.0/copy_data.CACinverse(i,i);

    // Reinit displacement related matrices and vectors
    copy_data.vertex_displ_matrix.reinit(n_cells, n_cells);
    copy_data.vertex_displ_matrix = 0;
    copy_data.vertex_displ_rhs = displacement_rhs;
    copy_data.vertex_displ_rhs *= -1.0;

    Vector<double> local_displacement_solution;

    // Compute BtAiC
    FullMatrix<double> BtAiC(n_cells,rotation_dim);
    BtAiC.triple_product(copy_data.Ainverse, copy_data.displacement_matrix, copy_data.rotation_matrix, true, false);
    // Compute BCACB
    FullMatrix<double> BCACB(n_cells,n_cells);
    BCACB.triple_product(copy_data.CACinverse, BtAiC, BtAiC, false, true);

    // Compute LHS
    copy_data.vertex_displ_matrix.triple_product(copy_data.Ainverse, copy_data.displacement_matrix,
                                                 copy_data.displacement_matrix, true, false);
    copy_data.vertex_displ_matrix.add(-1.0, BCACB);

    // Compute RHS
    Vector<double> tmp_rhs1(n_edges);
    Vector<double> tmp_rhs2(n_cells);
    Vector<double> tmp_rhs3(rotation_dim);
    Vector<double> tmp_rhs4(rotation_dim);
    // Contribution from first elimination step
    copy_data.Ainverse.vmult(tmp_rhs1, copy_data.stress_rhs, false);
    copy_data.displacement_matrix.Tvmult(tmp_rhs2, tmp_rhs1, false);
    copy_data.vertex_displ_rhs.sadd(1.0, tmp_rhs2);
    // Contribution from second elimination step
    copy_data.rotation_matrix.Tvmult(tmp_rhs3, tmp_rhs1, false);
    copy_data.CACinverse.vmult(tmp_rhs4, tmp_rhs3, false);
    copy_data.rotation_rhs = tmp_rhs4;
    BtAiC.vmult(tmp_rhs2, tmp_rhs4, false);
    copy_data.vertex_displ_rhs.sadd(-1.0, tmp_rhs2);

    copy_data.p = (*n_it).first;
  }

  template <int dim>
  void MultipointMixedElasticityProblem<dim>::copy_vertex_to_system (const VertexEliminationCopyData &copy_data)
  {
    A_inverse[copy_data.p] = copy_data.Ainverse;
    CAC_inverse[copy_data.p] = copy_data.CACinverse;
    displacement_matrix[copy_data.p] = copy_data.displacement_matrix;
    rotation_matrix[copy_data.p] = copy_data.rotation_matrix;
    stress_rhs[copy_data.p] = copy_data.stress_rhs;
    rotation_rhs[copy_data.p] = copy_data.rotation_rhs;

    std::set<types::global_dof_index>::iterator ui_it, uj_it;
    unsigned int i, j;
    for (ui_it = displacement_indices[copy_data.p].begin(), i = 0;
         ui_it != displacement_indices[copy_data.p].end();
         ++ui_it, ++i) {
      for (uj_it = displacement_indices[copy_data.p].begin(), j = 0;
           uj_it != displacement_indices[copy_data.p].end();
           ++uj_it, ++j)
        displ_system_matrix.add(*ui_it - n_s, *uj_it - n_s, copy_data.vertex_displ_matrix(i, j));

      displ_rhs(*ui_it - n_s) += -1.0*copy_data.vertex_displ_rhs(i);
    }
  }

  template <int dim>
  void MultipointMixedElasticityProblem<dim>::displacement_assembly()
  {
    TimerOutput::Scope t(computing_timer, "Displacement matrix assembly");

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    displ_rhs.reinit(n_u);

    WorkStream::run(vertex_matrix.begin(),
                    vertex_matrix.end(),
                    *this,
                    &MultipointMixedElasticityProblem::vertex_elimination,
                    &MultipointMixedElasticityProblem::copy_vertex_to_system,
                    VertexAssemblyScratchData(fe, triangulation, quad, face_quad),
                    VertexEliminationCopyData());

  }

//MultipointMixedElasticityProblem: Solving for velocity
  template <int dim>
  void MultipointMixedElasticityProblem<dim>::sr_assembly (const typename MapPointMatrix<dim>::iterator &n_it,
                                                            VertexAssemblyScratchData                         &scratch_data,
                                                            VertexEliminationCopyData                         &copy_data)
  {
    const unsigned int rotation_dim = static_cast<int>(0.5*dim*(dim-1));

    unsigned int n_edges = stress_indices.at((*n_it).first).size();
    unsigned int n_cells = displacement_indices.at((*n_it).first).size();

    Vector<double> local_displacement_solution;
    Vector<double> tmp_rhs1(n_edges);
    Vector<double> tmp_rhs2(n_edges);
    Vector<double> tmp_rhs3(n_edges);
    Vector<double> tmp_rhs4(rotation_dim);
    Vector<double> tmp_rhs5(rotation_dim);

    copy_data.vertex_stress_solution.reinit(n_edges);
    local_displacement_solution.reinit(n_cells);

    std::set<types::global_dof_index>::iterator u_it;
    unsigned int i;

    // RHS computations
    for (u_it = displacement_indices[(*n_it).first].begin(), i = 0; u_it != displacement_indices[(*n_it).first].end(); ++u_it, ++i)
      local_displacement_solution(i) = displ_solution(*u_it - n_s);

    // Compute rotation solution at vertex
    displacement_matrix[(*n_it).first].vmult(tmp_rhs2, local_displacement_solution, false);
    A_inverse[(*n_it).first].vmult(tmp_rhs1, tmp_rhs2, false);
    rotation_matrix[(*n_it).first].Tvmult(tmp_rhs4, tmp_rhs1, false);
    CAC_inverse[(*n_it).first].vmult(tmp_rhs5, tmp_rhs4, false);
    copy_data.vertex_rotation_solution = rotation_rhs[(*n_it).first];
    copy_data.vertex_rotation_solution.sadd(-1.0, tmp_rhs5);

    // Compute stress solution at vertex
    A_inverse[(*n_it).first].vmult(copy_data.vertex_stress_solution,stress_rhs[(*n_it).first],false); // Ai f
    copy_data.vertex_stress_solution.sadd(-1.0, tmp_rhs1); // Ai f - Ai B u
    rotation_matrix[(*n_it).first].vmult(tmp_rhs2, copy_data.vertex_rotation_solution, false); // C gamma
    A_inverse[(*n_it).first].vmult(tmp_rhs1,tmp_rhs2,false); // Ai C gamma
    copy_data.vertex_stress_solution.sadd(-1.0, tmp_rhs1); // Ai f - Ai B u - Ai C g

    // I lost track of negative signs...
    copy_data.vertex_rotation_solution *= -1.0;

    copy_data.p = (*n_it).first;
  }

  template <int dim>
  void MultipointMixedElasticityProblem<dim>::copy_vertex_sr_to_global (const VertexEliminationCopyData &copy_data)
  {
    std::set<types::global_dof_index>::iterator si_it, pi_it;
    unsigned int i;

    for (si_it = stress_indices[copy_data.p].begin(), i = 0;
         si_it != stress_indices[copy_data.p].end();
         ++si_it, ++i)
    {
      stress_solution(*si_it) += copy_data.vertex_stress_solution(i);
    }

    for (pi_it = rotation_indices[copy_data.p].begin(), i = 0;
         pi_it != rotation_indices[copy_data.p].end();
         ++pi_it, ++i)
    {
      rotation_solution(*pi_it-n_s-n_u) += copy_data.vertex_rotation_solution(i);
    }

  }

  template <int dim>
  void MultipointMixedElasticityProblem<dim>::sr_recovery()
  {
    TimerOutput::Scope t(computing_timer, "Stress and rotation solution recovery");

    QGaussLobatto<dim> quad(degree+1);
    QGauss<dim-1> face_quad(degree);

    stress_solution.reinit(n_s);
    rotation_solution.reinit(n_p);

    WorkStream::run(vertex_matrix.begin(),
                    vertex_matrix.end(),
                    *this,
                    &MultipointMixedElasticityProblem::sr_assembly,
                    &MultipointMixedElasticityProblem::copy_vertex_sr_to_global,
                    VertexAssemblyScratchData(fe, triangulation, quad, face_quad),
                    VertexEliminationCopyData());

    solution.reinit(3);
    solution.block(0) = stress_solution;
    solution.block(1) = displ_solution;
    solution.block(2) = rotation_solution;
    solution.collect_sizes();
  }

// MultipointMixedElasticityProblem: Solve SPD cell-centered system for displacement
  template <int dim>
  void MultipointMixedElasticityProblem<dim>::solve_displacement()
  {
    TimerOutput::Scope t(computing_timer, "Displacement CG solve");

    displ_solution.reinit(n_u);

    SolverControl solver_control (10*n_u, 1e-10);
    SolverCG<> solver (solver_control);

    PreconditionIdentity identity;

    solver.solve(displ_system_matrix, displ_solution, displ_rhs, identity);
  }


// MultipointMixedElasticityProblem: Compute errors
  template <int dim>
  void MultipointMixedElasticityProblem<dim>::compute_errors(const unsigned cycle)
  {
    TimerOutput::Scope t(computing_timer, "Compute errors");

    const ComponentSelectFunction<dim> rotation_mask(dim*dim+dim, MultipointMixedElasticityProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> displacement_mask(std::make_pair(dim*dim,dim*dim+dim), MultipointMixedElasticityProblem<dim>::total_dim);
    const ComponentSelectFunction<dim> stress_mask(std::make_pair(0,dim*dim), MultipointMixedElasticityProblem<dim>::total_dim);
    ExactSolution<dim> exact_solution;

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
      const ComponentSelectFunction<dim> stress_component_mask (std::make_pair(i*dim,(i+1)*dim),MultipointMixedElasticityProblem<dim>::total_dim);

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


// MultipointMixedElasticityProblem: Output results
  template <int dim>
  void MultipointMixedElasticityProblem<dim>::output_results(const unsigned int cycle, const unsigned int refine)
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

// MultipointMixedElasticityProblem: run
  template <int dim>
  void MultipointMixedElasticityProblem<dim>::run(const unsigned int refine, const unsigned int grid)
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
          if (dim == 3) {
            triangulation.refine_global(1);
            //GridTools::transform(&grid_transform<dim>, triangulation);
          } else if (dim == 2)
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

      vertex_assembly();
      make_cell_centered_sp();
      displacement_assembly();
      solve_displacement ();
      sr_recovery ();
      compute_errors (cycle);
      output_results (cycle, refine);
      reset_data_structures ();

      computing_timer.print_summary ();
      computing_timer.reset ();
    }
  }

  template class MultipointMixedElasticityProblem<2>;
  template class MultipointMixedElasticityProblem<3>;
}
