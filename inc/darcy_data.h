// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_DARCY_DATA_H
#define PEFLOW_DARCY_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>

#include <deal.II/base/parsed_function.h>


namespace darcy
{
	using namespace dealii;

	
  // Exact solution
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  private:
    ParameterHandler &prm;
  public:
    ExactSolution(ParameterHandler &);
    Functions::ParsedFunction<dim> exact_solution_val_data;
    Functions::ParsedFunction<dim> exact_solution_grad_val_data;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim,double>>  &grads) const;

  };

  template <int dim>
  ExactSolution<dim>::ExactSolution( ParameterHandler &param)
    :
      Function<dim>(dim+1),
      prm(param),
      exact_solution_val_data(dim+1),
      exact_solution_grad_val_data(dim*(dim+1))
  {}

  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {

    switch (dim)
      {
      case 2:
        //Velocity
        values(0) = exact_solution_val_data.value(p,0);
        values(1) = exact_solution_val_data.value(p,1);
        //Pressure
        values(2) = exact_solution_val_data.value(p,2);
        break;
      case 3:
        // Velocity:
        values(0) = exact_solution_val_data.value(p,0);
        values(1) = exact_solution_val_data.value(p,1);
        values(2) = exact_solution_val_data.value(p,2);
        // Pressure:
        values(3) = exact_solution_val_data.value(p,3);
        break;
      default:
        Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  void
  ExactSolution<dim>::vector_gradient (const Point<dim> &p,
                                       std::vector<Tensor<1,dim,double>> &grads) const
  {
    Tensor<1,dim> tmp;
    switch (dim)
      {
      case 2:
        // Velocity
        tmp[0] = exact_solution_grad_val_data.value(p,0);
        tmp[1] = exact_solution_grad_val_data.value(p,1);
        grads[0] = tmp;
        tmp[0] = exact_solution_grad_val_data.value(p,2);
        tmp[1] = exact_solution_grad_val_data.value(p,3);
        grads[1] = tmp;
        // Pressure (not used)
        tmp[0] = 0;
        tmp[1] = 0;
        grads[2] = tmp;
        break;
      case 3:
        // Velocity
        tmp[0] = exact_solution_grad_val_data.value(p,0);
        tmp[1] = exact_solution_grad_val_data.value(p,1);
        tmp[2] = exact_solution_grad_val_data.value(p,2);
        grads[0] = tmp;
        tmp[0] = exact_solution_grad_val_data.value(p,3);
        tmp[1] = exact_solution_grad_val_data.value(p,4);
        tmp[2] = exact_solution_grad_val_data.value(p,5);
        grads[1] = tmp;
        tmp[0] = exact_solution_grad_val_data.value(p,6);
        tmp[1] = exact_solution_grad_val_data.value(p,7);
        tmp[2] = exact_solution_grad_val_data.value(p,8);
        grads[2] = tmp;
        // Pressure (not used)
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        grads[3] = tmp;
        break;
      default:
        Assert(false, ExcNotImplemented());
      }
  }


   // Permeability (inverse of the tensor)
  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse (ParameterHandler &, const Functions::ParsedFunction<dim> *k_inv_data);
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
    const Functions::ParsedFunction<dim> *k_inv_data;
  private:
    ParameterHandler &prm;
  };

  template <int dim>
  KInverse<dim>::KInverse(ParameterHandler &param, const Functions::ParsedFunction<dim> *k_inv_data)
    :
      TensorFunction<2,dim>(),
      k_inv_data(k_inv_data),
      prm(param)
  {}

  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));
    for (unsigned int p=0; p<points.size(); ++p)
      {
        switch(dim)
          {
          case 2:
            for (unsigned int p=0; p<points.size(); ++p)
              {
                values[p].clear ();
                values[p][0][0] = k_inv_data->value(points[p],0);
                values[p][0][1] = k_inv_data->value(points[p],1);
                values[p][1][0] = k_inv_data->value(points[p],2);
                values[p][1][1] = k_inv_data->value(points[p],3);
              }
            break;
          case 3:
            for (unsigned int p=0; p<points.size(); ++p)
              {
                values[p].clear ();
                values[p][0][0] = k_inv_data->value(points[p],0);
                values[p][0][1] = k_inv_data->value(points[p],1);
                values[p][0][2] = k_inv_data->value(points[p],2);
                values[p][1][0] = k_inv_data->value(points[p],3);
                values[p][1][1] = k_inv_data->value(points[p],4);
                values[p][1][2] = k_inv_data->value(points[p],5);
                values[p][2][0] = k_inv_data->value(points[p],6);
                values[p][2][1] = k_inv_data->value(points[p],7);
                values[p][2][2] = k_inv_data->value(points[p],8);
              }
            break;
          default:
            Assert(false, ExcNotImplemented());
          }
      }
  }
}

#endif /*PEFLOW_DARCY_DATA_H*/
