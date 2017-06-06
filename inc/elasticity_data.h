//
// Created by eldar on 6/4/17.
//

#ifndef PEFLOW_ELASTICITY_DATA_H
#define PEFLOW_ELASTICITY_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>

namespace elasticity
{
  using namespace dealii;

  // First Lame parameter
  template <int dim>
  class LameFirstParameter : public Function<dim>
  {
  public:
    LameFirstParameter () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    virtual void value_list (const std::vector<Point<dim>> &points,
                               std::vector<double> &value_list,
                               const unsigned int c = 0) const;

  };

  // First Lame parameter: value at point
  template <int dim>
  double
  LameFirstParameter<dim>::value(const Point<dim> &p, const unsigned int component) const
  {
    switch (dim)
    {
      case 2:
        return 123.0;
        break;
      case 3:
        return 123.0;
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // First Lame parameter: value evaluated at several points at once
  template <int dim>
  void
  LameFirstParameter<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &value_list,
                                      const unsigned int) const
  {
    Assert(value_list.size() == points.size(),
           ExcDimensionMismatch(value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      value_list[p] = LameFirstParameter<dim>::value(points[p]);
  }


  // Second Lame parameter
  template <int dim>
  class LameSecondParameter : public Function<dim>
  {
  public:
    LameSecondParameter () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    virtual void value_list (const std::vector<Point<dim>> &points,
                             std::vector<double> &value_list,
                             const unsigned int c = 0) const;

  };

  // Second Lame parameter: value at point
  template <int dim>
  double
  LameSecondParameter<dim>::value(const Point<dim> &p, const unsigned int component) const
  {
    switch (dim)
    {
      case 2:
        return 79.3;
        break;
      case 3:
        return 79.3;
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  }

  // Second Lame parameter: value evaluated at several points at once
  template <int dim>
  void
  LameSecondParameter<dim>::value_list(const std::vector<Point<dim>> &points, std::vector<double> &value_list,
                                      const unsigned int) const
  {
    Assert(value_list.size() == points.size(),
           ExcDimensionMismatch(value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      value_list[p] = LameSecondParameter<dim>::value(points[p]);
  }

  
  // Right hand side function
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(dim) {}

    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> >   &points,
                                    std::vector<Vector<double> > &value_list) const;
  };

  // RHS: vector values at one point
  template <int dim>
  inline
  void RightHandSide<dim>::vector_value(const Point<dim> &p, Vector<double>   &values) const
  {
    Assert(values.size() == dim,
           ExcDimensionMismatch(values.size(),dim));
    Assert(dim != 1, ExcNotImplemented());

    const LameFirstParameter<dim> lambda_function;
    const LameSecondParameter<dim> mu_function;
    
    double lambda = lambda_function.value(p);
    double mu = mu_function.value(p);
    
    switch (dim)
    {
      case 2:
        values(0) = -(-M_PI*M_PI*cos(M_PI*p[0])*(lambda*sin(M_PI*p[1]) + lambda*sin(2*M_PI*p[1]) + mu*sin(M_PI*p[1])
                                                 + 6*mu*sin(2*M_PI*p[1])));
        values(1) = -(-M_PI*M_PI*sin(M_PI*p[0])*(lambda*cos(M_PI*p[1]) + 3*mu*cos(M_PI*p[1]) +
                2*lambda*(2*cos(M_PI*p[1])*cos(M_PI*p[1]) - 1) + 2*mu*(2*cos(M_PI*p[1])*cos(M_PI*p[1]) - 1)));
        break;
      case 3:
        values(0) = -2.0*exp(p[0])*(lambda + mu)*(cos(M_PI/12.0) - 1.0);
        values(1) = mu*exp(p[0])*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        values(2) = -mu*exp(p[0])*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        break;
      default:
      Assert(false, ExcNotImplemented());
    }
  }

  // RHS: vector values evaluated at several points at once
  template <int dim>
  void RightHandSide<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> >   &value_list) const
  {
    Assert(value_list.size() == points.size(),
           ExcDimensionMismatch(value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide<dim>::vector_value(points[p], value_list[p]);
  }


  // Boundary conditions (displacement) class
  template <int dim>
  class DisplacementBoundaryValues : public Function<dim>
  {
  public:
    DisplacementBoundaryValues() : Function<dim>(dim) {}

    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> >   &points,
                                    std::vector<Vector<double> > &value_list) const;
  };

  // Boundary conditions: vector values at one point
  template <int dim>
  void DisplacementBoundaryValues<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
  {
    const LameFirstParameter<dim> lambda_function;
    const LameSecondParameter<dim> mu_function;

    double lambda = lambda_function.value(p);
    double mu = mu_function.value(p);
    
    switch (dim)
    {
      case 2:
        values(0) = cos(M_PI*p[0])*sin(2*M_PI*p[1]);
        values(1) = sin(M_PI*p[0])*cos(M_PI*p[1]);
        break;
      case 3:
        values(0) = 0.0;
        values(1) = -(exp(p[0]) - 1.0)*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        values(2) = (exp(p[0]) - 1.0)*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        break;
      default:
      Assert(false, ExcNotImplemented());
    }
  }

  // Boundary conditions: vector values evaluated at several points at once
  template <int dim>
  void DisplacementBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                                          std::vector<Vector<double> >   &value_list) const
  {
    Assert(value_list.size() == points.size(),
           ExcDimensionMismatch(value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      DisplacementBoundaryValues<dim>::vector_value(points[p], value_list[p]);
  }


  // Exact solution class
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution() : Function<dim>(dim*dim + dim + (3-dim)*(dim-1) + (dim-2)*dim) {}

    virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim,double> > &grads) const;
  private:
    int dummy;
  };

  // Exact solution: values of stress, displacement and rotation
  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
  {
    const LameFirstParameter<dim> lambda_function;
    const LameSecondParameter<dim> mu_function;

    double lambda = lambda_function.value(p);
    double mu = mu_function.value(p);
    
    switch (dim)
    {
      case 2:
        // Stress:
        values(0) = -M_PI*sin(M_PI*p[0])*(lambda*sin(M_PI*p[1]) + lambda*sin(2*M_PI*p[1]) + 2*mu*sin(2*M_PI*p[1]));
        values(1) = mu*(M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]) + 2*M_PI*cos(M_PI*p[0])*cos(2*M_PI*p[1]));
        values(2) = mu*(M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]) + 2*M_PI*cos(M_PI*p[0])*cos(2*M_PI*p[1]));
        values(3) = -M_PI*sin(M_PI*p[0])*(lambda*sin(M_PI*p[1]) + lambda*sin(2*M_PI*p[1]) + 2*mu*sin(M_PI*p[1]));
        // Displacement:
        values(4) = cos(M_PI*p[0])*sin(2*M_PI*p[1]);
        values(5) = sin(M_PI*p[0])*cos(M_PI*p[1]);
        // Rotation:
        values(6) = 0.5*(2*M_PI*cos(M_PI*p[0])*cos(2*M_PI*p[1]) - M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]));
        break;
      case 3:
        // Stress:
        values(0) = 2.0*lambda*(exp(p[0]) - 1.0)*(cos(M_PI/12.0) - 1.0);
        values(1) = -mu*exp(p[0])*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        values(2) = mu*exp(p[0])*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        values(3) = -mu*exp(p[0])*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        values(4) = 2*(lambda + mu)*(exp(p[0]) - 1.0)*(cos(M_PI/12.0) - 1.0);
        values(5) = 0.0;
        values(6) = mu*exp(p[0])*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        values(7) = 0.0;
        values(8) = 2*(lambda + mu)*(exp(p[0]) - 1.0)*(cos(M_PI/12.0) - 1.0);
        // Displacement:
        values(9) = 0.0;
        values(10) = -(exp(p[0]) - 1.0)*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        values(11) = (exp(p[0]) - 1.0)*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        // Rotation:
        values(12) = sin(M_PI/12.0)*(exp(p[0]) - 1.0);
        values(13) = -(exp(p[0])*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5))/2.0;
        values(14) = -(exp(p[0])*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5))/2.0;
        break;
      default:
      Assert(false, ExcNotImplemented());
    }
  }

  // Exact solution: gradient of stress, displacement and rotation
  template <int dim>
  void
  ExactSolution<dim>::vector_gradient (const Point<dim> &p, std::vector<Tensor<1,dim,double> > &grads) const
  {
    const LameFirstParameter<dim> lambda_function;
    const LameSecondParameter<dim> mu_function;

    double lambda = lambda_function.value(p);
    double mu = mu_function.value(p);
    
    int total_dim = dim*dim + dim + static_cast<int>(0.5*dim*(dim-1));
    Tensor<1,dim> tmp;
    switch (dim)
    {
      case 2:
        // Gradient of stress (2x2x2 tensor)
        // Gradient of sigma_11
        tmp[0] = -M_PI*M_PI*cos(M_PI*p[0])*(lambda*sin(M_PI*p[1]) + lambda*sin(2*M_PI*p[1]) + 2*mu*sin(2*M_PI*p[1]));
        tmp[1] = -M_PI*M_PI*sin(M_PI*p[0])*(lambda*cos(M_PI*p[1]) + 2*lambda*(2*cos(M_PI*p[1])*cos(M_PI*p[1]) - 1) + 4*mu*(2*cos(M_PI*p[1])*cos(M_PI*p[1]) - 1));
        grads[0] = tmp;
        // Gradient of sigma_12
        tmp[0] = -M_PI*M_PI*mu*sin(M_PI*p[0])*(cos(M_PI*p[1]) + 4*cos(M_PI*p[1])*cos(M_PI*p[1]) - 2);
        tmp[1] = -M_PI*M_PI*mu*cos(M_PI*p[0])*(sin(M_PI*p[1]) + 4*sin(2*M_PI*p[1]));
        grads[1] = tmp;
        // Gradient of sigma_21
        tmp[0] = -M_PI*M_PI*mu*sin(M_PI*p[0])*(cos(M_PI*p[1]) + 4*cos(M_PI*p[1])*cos(M_PI*p[1]) - 2);
        tmp[1] = -M_PI*M_PI*mu*cos(M_PI*p[0])*(sin(M_PI*p[1]) + 4*sin(2*M_PI*p[1]));
        grads[2] = tmp;
        // Gradient of sigma_22
        tmp[0] = -M_PI*M_PI*cos(M_PI*p[0])*(lambda*sin(M_PI*p[1]) + lambda*sin(2*M_PI*p[1]) + 2*mu*sin(M_PI*p[1]));
        tmp[1] = -M_PI*M_PI*sin(M_PI*p[0])*(lambda*cos(M_PI*p[1]) - 2*lambda + 2*mu*cos(M_PI*p[1]) + 4*lambda*cos(M_PI*p[1])*cos(M_PI*p[1]));
        grads[3] = tmp;
        // Rest is meaningles, displacement and rotation are L2
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        for (int k=dim*dim;k<total_dim;++k)
          grads[k] = tmp;

        break;
      case 3:
        // Gradient of stress (3x3x3 tensor)
        // Gradient of sigma_11
        tmp[0] = 2.0*lambda*exp(p[0])*(cos(M_PI/12.0) - 1.0);
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        grads[0] = tmp;
        // Gradient of sigma_12
        tmp[0] = -mu*exp(p[0])*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        tmp[1] = mu*exp(p[0])*(cos(M_PI/12.0) - 1.0);
        tmp[2] = -mu*sin(M_PI/12.0)*exp(p[0]);
        grads[1] = tmp;
        // Gradient of sigma_13
        tmp[0] = mu*exp(p[0])*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        tmp[1] = mu*sin(M_PI/12.0)*exp(p[0]);
        tmp[2] = mu*exp(p[0])*(cos(M_PI/12.0) - 1.0);
        grads[2] = tmp;
        // Gradient of sigma_21
        tmp[0] = -mu*exp(p[0])*(p[1] - cos(M_PI/12.0)*(p[1] - 0.5) + sin(M_PI/12.0)*(p[2] - 0.5) - 0.5);
        tmp[1] = mu*exp(p[0])*(cos(M_PI/12.0) - 1.0);
        tmp[2] = -mu*sin(M_PI/12.0)*exp(p[0]);
        grads[3] = tmp;
        // Gradient of sigma_22
        tmp[0] = 2.0*exp(p[0])*(lambda + mu)*(cos(M_PI/12.0) - 1.0);
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        grads[4] = tmp;
        // Gradient of sigma_23
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        grads[5] = tmp;
        // Gradient of sigma_31
        tmp[0] = mu*exp(p[0])*(cos(M_PI/12.0)*(p[2] - 0.5) - p[2] + sin(M_PI/12.0)*(p[1] - 0.5) + 0.5);
        tmp[1] = mu*sin(M_PI/12.0)*exp(p[0]);
        tmp[2] = mu*exp(p[0])*(cos(M_PI/12.0) - 1.0);
        grads[6] = tmp;
        // Gradient of sigma_32
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        grads[7] = tmp;
        // Gradient of sigma_33
        tmp[0] = 2.0*exp(p[0])*(lambda + mu)*(cos(M_PI/12.0) - 1.0);
        tmp[1] = 0.0;
        tmp[2] = 0.0;
        grads[8] = tmp;
        // Rest is meaningles, displacement and rotation are L2
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
}

#endif //PEFLOW_ELASTICITY_DATA_H
