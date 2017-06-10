// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_BIOT_DATA_H
#define PEFLOW_BIOT_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>

namespace biot
{
	using namespace dealii;

  // Exact solution
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  private:
    const double current_time;
    ParameterHandler &prm;
  public:
    ExactSolution(const double cur_time, ParameterHandler &);
    Functions::ParsedFunction<dim> exact_solution_val_data;
    Functions::ParsedFunction<dim> exact_solution_grad_val_data;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim,double>>  &grads) const;
    //inline double get_time() {return current_time;}

  };

  template <int dim>
  ExactSolution<dim>::ExactSolution(const double cur_time, ParameterHandler &param)
    :
      Function<dim>(dim+1+dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim),
      prm(param),
      exact_solution_val_data(dim+1+dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim),
      exact_solution_grad_val_data( dim*(dim+1+dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim)),
      current_time(cur_time)
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
        // Stress
        values(3) = exact_solution_val_data.value(p,3);
        values(4) = exact_solution_val_data.value(p,4);
        values(5) = exact_solution_val_data.value(p,5);
        values(6) = exact_solution_val_data.value(p,6);
        // Displacement
        values(7) = exact_solution_val_data.value(p,7);
        values(8) = exact_solution_val_data.value(p,8);
        // Rotation
        values(9) = exact_solution_val_data.value(p,9);
        break;
      case 3:
        // Velocity:
        values(0) = exact_solution_val_data.value(p,0);
        values(1) = exact_solution_val_data.value(p,1);
        values(2) = exact_solution_val_data.value(p,2);
        // Pressure:
        values(3) = exact_solution_val_data.value(p,3);
        // Stress
        values(4) = exact_solution_val_data.value(p,4);
        values(5) = exact_solution_val_data.value(p,5);
        values(6) = exact_solution_val_data.value(p,6);
        values(7) = exact_solution_val_data.value(p,7);
        values(8) = exact_solution_val_data.value(p,8);
        values(9) = exact_solution_val_data.value(p,9);
        values(10) = exact_solution_val_data.value(p,10);
        values(11) = exact_solution_val_data.value(p,11);
        values(12) = exact_solution_val_data.value(p,12);
        // Displacement
        values(13) = exact_solution_val_data.value(p,13);
        values(14) = exact_solution_val_data.value(p,14);
        values(15) = exact_solution_val_data.value(p,15);
        // Rotation
        values(16) = exact_solution_val_data.value(p,16);
        values(17) = exact_solution_val_data.value(p,17);
        values(18) = exact_solution_val_data.value(p,18);
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
        // Stress
        //sigma 11
        tmp[0] = exact_solution_grad_val_data.value(p,6);
        tmp[1] = exact_solution_grad_val_data.value(p,7);
        grads[3] = tmp;
        //sigma 12
        tmp[0] = exact_solution_grad_val_data.value(p,8);
        tmp[1] = exact_solution_grad_val_data.value(p,9);
        grads[4] = tmp;
        //sigma 12
        tmp[0] = exact_solution_grad_val_data.value(p,10);
        tmp[1] = exact_solution_grad_val_data.value(p,11);
        grads[5] = tmp;
        // sigma 22
        tmp[0] = exact_solution_grad_val_data.value(p,12);
        tmp[1] = exact_solution_grad_val_data.value(p,13);
        grads[6] = tmp;
        // Rest (not used)
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        for (int k=1+dim+dim*dim;k<total_dim;++k)
          grads[k] = tmp;
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
        // Stress
        //sigma 11
        tmp[0] = exact_solution_grad_val_data.value(p,12);
        tmp[1] = exact_solution_grad_val_data.value(p,13);
        tmp[2] = exact_solution_grad_val_data.value(p,14);
        grads[4] = tmp;
        //sigma 12
        tmp[0] = exact_solution_grad_val_data.value(p,15);
        tmp[1] = exact_solution_grad_val_data.value(p,16);
        tmp[2] = exact_solution_grad_val_data.value(p,17);
        grads[5] = tmp;
        //sigma 13
        tmp[0] = exact_solution_grad_val_data.value(p,18);
        tmp[1] = exact_solution_grad_val_data.value(p,19);
        tmp[2] = exact_solution_grad_val_data.value(p,20);
        grads[6] = tmp;
        //sigma 21
        tmp[0] = exact_solution_grad_val_data.value(p,21);
        tmp[1] = exact_solution_grad_val_data.value(p,22);
        tmp[2] = exact_solution_grad_val_data.value(p,23);
        grads[7] = tmp;
        // sigma 22
        tmp[0] = exact_solution_grad_val_data.value(p,24);
        tmp[1] = exact_solution_grad_val_data.value(p,25);
        tmp[2] = exact_solution_grad_val_data.value(p,26);
        grads[8] = tmp;
        // sigma 23
        tmp[0] = exact_solution_grad_val_data.value(p,27);
        tmp[1] = exact_solution_grad_val_data.value(p,28);
        tmp[2] = exact_solution_grad_val_data.value(p,29);
        grads[9] = tmp;
        // sigma 31
        tmp[0] = exact_solution_grad_val_data.value(p,30);
        tmp[1] = exact_solution_grad_val_data.value(p,31);
        tmp[2] = exact_solution_grad_val_data.value(p,32);
        grads[10] = tmp;
        // sigma 32
        tmp[0] = exact_solution_grad_val_data.value(p,33);
        tmp[1] = exact_solution_grad_val_data.value(p,34);
        tmp[2] = exact_solution_grad_val_data.value(p,35);
        grads[11] = tmp;
        // sigma 33
        tmp[0] = exact_solution_grad_val_data.value(p,33);
        tmp[1] = exact_solution_grad_val_data.value(p,34);
        tmp[2] = exact_solution_grad_val_data.value(p,35);
        grads[12] = tmp;
        // Rest (not used)
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        for (int k=1+dim+dim*dim;k<total_dim;++k)
          grads[k] = tmp;
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
      prm(param),
      k_inv_data(k_inv_data)
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

#endif /*PEFLOW_BIOT_DATA_H*/
