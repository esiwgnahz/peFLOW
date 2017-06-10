// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_ELASTICITY_DATA_H
#define PEFLOW_ELASTICITY_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>

#include <deal.II/base/parsed_function.h>

namespace elasticity
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
  ExactSolution<dim>::ExactSolution(ParameterHandler &param)
    :
      Function<dim>(dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim),
      prm(param),
      exact_solution_val_data(dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim),
      exact_solution_grad_val_data(dim*(dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim))
  {}

  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {

    switch (dim)
      {
      case 2:
        //Stress
        values(0) = exact_solution_val_data.value(p,0);
        values(1) = exact_solution_val_data.value(p,1);
        values(2) = exact_solution_val_data.value(p,2);
        values(3) = exact_solution_val_data.value(p,3);
        // Displacement
        values(4) = exact_solution_val_data.value(p,4);
        values(5) = exact_solution_val_data.value(p,5);
        // Rotation
        values(6) = exact_solution_val_data.value(p,6);
        break;
      case 3:
        // Stress:
        values(0) = exact_solution_val_data.value(p,0);
        values(1) = exact_solution_val_data.value(p,1);
        values(2) = exact_solution_val_data.value(p,2);
        values(3) = exact_solution_val_data.value(p,3);
        values(4) = exact_solution_val_data.value(p,4);
        values(5) = exact_solution_val_data.value(p,5);
        values(6) = exact_solution_val_data.value(p,6);
        values(7) = exact_solution_val_data.value(p,7);
        values(8) = exact_solution_val_data.value(p,8);
        // Displacement
        values(9)  = exact_solution_val_data.value(p,9);
        values(10) = exact_solution_val_data.value(p,10);
        values(11) = exact_solution_val_data.value(p,11);
        // Rotation
        values(12) = exact_solution_val_data.value(p,12);
        values(13) = exact_solution_val_data.value(p,13);
        values(14) = exact_solution_val_data.value(p,14);
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
    int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));


    switch (dim)
      {
      case 2:
        // Stress
        //sigma 11
        tmp[0] = exact_solution_grad_val_data.value(p,0);
        tmp[1] = exact_solution_grad_val_data.value(p,1);
        grads[0] = tmp;
        //sigma 12
        tmp[0] = exact_solution_grad_val_data.value(p,2);
        tmp[1] = exact_solution_grad_val_data.value(p,3);
        grads[1] = tmp;
        //sigma 21
        tmp[0] = exact_solution_grad_val_data.value(p,4);
        tmp[1] = exact_solution_grad_val_data.value(p,5);
        grads[2] = tmp;
        // sigma 22
        tmp[0] = exact_solution_grad_val_data.value(p,6);
        tmp[1] = exact_solution_grad_val_data.value(p,7);
        grads[3] = tmp;
        // Rest (not used)
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        for (int k=dim*dim;k<total_dim;++k)
          grads[k] = tmp;
        break;
      case 3:
        // Stress
        //sigma 11
        tmp[0] = exact_solution_grad_val_data.value(p,0);
        tmp[1] = exact_solution_grad_val_data.value(p,1);
        tmp[2] = exact_solution_grad_val_data.value(p,2);
        grads[0] = tmp;
        //sigma 12
        tmp[0] = exact_solution_grad_val_data.value(p,3);
        tmp[1] = exact_solution_grad_val_data.value(p,4);
        tmp[2] = exact_solution_grad_val_data.value(p,5);
        grads[1] = tmp;
        //sigma 13
        tmp[0] = exact_solution_grad_val_data.value(p,6);
        tmp[1] = exact_solution_grad_val_data.value(p,7);
        tmp[2] = exact_solution_grad_val_data.value(p,8);
        grads[2] = tmp;
        //sigma 21
        tmp[0] = exact_solution_grad_val_data.value(p,9);
        tmp[1] = exact_solution_grad_val_data.value(p,10);
        tmp[2] = exact_solution_grad_val_data.value(p,11);
        grads[3] = tmp;
        // sigma 22
        tmp[0] = exact_solution_grad_val_data.value(p,12);
        tmp[1] = exact_solution_grad_val_data.value(p,13);
        tmp[2] = exact_solution_grad_val_data.value(p,14);
        grads[4] = tmp;
        // sigma 23
        tmp[0] = exact_solution_grad_val_data.value(p,15);
        tmp[1] = exact_solution_grad_val_data.value(p,16);
        tmp[2] = exact_solution_grad_val_data.value(p,17);
        grads[5] = tmp;
        // sigma 31
        tmp[0] = exact_solution_grad_val_data.value(p,18);
        tmp[1] = exact_solution_grad_val_data.value(p,19);
        tmp[2] = exact_solution_grad_val_data.value(p,20);
        grads[6] = tmp;
        // sigma 32
        tmp[0] = exact_solution_grad_val_data.value(p,21);
        tmp[1] = exact_solution_grad_val_data.value(p,22);
        tmp[2] = exact_solution_grad_val_data.value(p,23);
        grads[7] = tmp;
        // sigma 33
        tmp[0] = exact_solution_grad_val_data.value(p,24);
        tmp[1] = exact_solution_grad_val_data.value(p,25);
        tmp[2] = exact_solution_grad_val_data.value(p,26);
        grads[8] = tmp;
        // Rest (not used)
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        for (int k=dim*dim;k<total_dim;++k)
          grads[k] = tmp;
        break;
      default:
        Assert(false, ExcNotImplemented());
      }
  }

  // Lame coefficients
  template <int dim>
  class LameCoefficients : public Function<dim>
  {
  public:
    LameCoefficients<dim> (ParameterHandler &, const Functions::ParsedFunction<dim> *mu_data,
                           const Functions::ParsedFunction<dim> *lambda_data);
    const double mu_value (const Point<dim> &p) const   {return mu->value(p);}
    const double lambda_value (const Point<dim> &p) const   {return lambda->value(p);}
    const Functions::ParsedFunction<dim> *mu;
    const Functions::ParsedFunction<dim> *lambda;
  private:
    ParameterHandler &prm;
  };


  template <int dim>
  LameCoefficients<dim>::LameCoefficients(ParameterHandler &param, const Functions::ParsedFunction<dim> *mu_data,
                                          const Functions::ParsedFunction<dim> *lambda_data)
    :
      Function<dim>(dim),
      prm(param),
      mu(mu_data),
      lambda(lambda_data)
  {} 
  

  
}

#endif //PEFLOW_ELASTICITY_DATA_H
