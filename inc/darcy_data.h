#ifndef PEFLOW_DARCY_DATA_H
#define PEFLOW_DARCY_DATA_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <cmath>

namespace darcy
{
	using namespace dealii;

	template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>(1) {}

    virtual double value (const Point<dim> &p, const unsigned int component = 0) const;
  };

  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    switch (dim)
    {
      case 2:
        return -(p[0]*2.0+2.0)*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]))+pow(p[0]+1.0,2.0)*((p[0]*p[0]*p[0])*(p[1]*p[1])*-1.2E1+sin(p[0]*p[1])*cos(p[1])+p[0]*cos(p[0]*p[1])*sin(p[1])*2.0+(p[0]*p[0])*sin(p[0]*p[1])*cos(p[1]))-(pow(p[0]+1.0,2.0)+p[1]*p[1])*(p[0]*(p[1]*p[1]*p[1]*p[1])*6.0-(p[1]*p[1])*sin(p[0]*p[1])*cos(p[1])+2.0)-sin(p[0]*p[1])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1+cos(p[0]*p[1])*cos(p[1])-p[1]*cos(p[0]*p[1])*sin(p[1])-p[0]*p[1]*sin(p[0]*p[1])*cos(p[1]))*2.0-p[0]*cos(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]))-p[1]*cos(p[0]*p[1])*((p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-sin(p[0]*p[1])*sin(p[1])+p[0]*cos(p[0]*p[1])*cos(p[1]));
        break;
      case 3:
        return -(p[1]*2.0-sin(p[2]))*pow(p[1]+3.0,2.0)-(pow(p[1]+2.0,2.0)+p[0]*p[0])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1-(p[1]*p[1])*cos(p[0]*p[1])+2.0)-(p[2]*p[2]+2.0)*((p[0]*p[0]*p[0]*p[0])*p[1]*6.0-(p[0]*p[0])*cos(p[0]*p[1]))-p[2]*sin(p[1]*p[2])*4.0-p[0]*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]))*2.0-p[2]*cos(p[1]*p[2])*(cos(p[2])+p[1]*p[2]*2.0)+p[1]*sin(p[0]*p[1])*(cos(p[2])+p[1]*p[2]*2.0)-p[1]*cos(p[1]*p[2])*((p[0]*p[0]*p[0]*p[0])*(p[1]*p[1])*3.0-p[0]*sin(p[0]*p[1])+p[2]*p[2]);
        break;
      default:
      Assert (false, ExcNotImplemented());
    }
  }


  // Boundary conditions (pressure)
  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double
  PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                      const unsigned int /*component*/) const
  {
    switch (dim)
    {
      case 2:
        return (p[0]*p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])+p[0]*p[0]+sin(p[0]*p[1])*cos(p[1]);
        break;
      case 3:
        return cos(p[0]*p[1])+sin(p[2])+(p[0]*p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])+p[1]*(p[2]*p[2])+p[0]*p[0];
        break;
      default:
      Assert(false, ExcNotImplemented());
    }

  }


  // Boundary conditions (velocity)
  template <int dim>
  class VelocityBoundaryValues : public Function<dim>
  {
  public:
    VelocityBoundaryValues() : Function<dim>(dim) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
    virtual void vector_value_list (const std::vector<Point<dim> > 	 &points,
                                    std::vector<Vector<double> > &value_list) const;
  };

  template <int dim>
  void VelocityBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                                  Vector<double>   &values) const
  {
    switch (dim)
    {
      case 2:
        values(0) = 2*std::sin(2*p[0]+1)*std::exp(p[1]);
        values(1) = -std::cos(2*p[0]+1)*std::exp(p[1]);
        break;
      case 3:
        values(0) = 0.0;
        values(1) = 0.0;
        values(2) = 0.0;
        break;
      default:
      Assert(false, ExcNotImplemented());
    }
  }

  template <int dim>
  void VelocityBoundaryValues<dim>::vector_value_list(const std::vector<Point<dim> > &points,
                                                      std::vector<Vector<double> >   &value_list) const
  {
    Assert(value_list.size() == points.size(),
           ExcDimensionMismatch(value_list.size(), points.size()));

    const unsigned int n_points = points.size();

    for (unsigned int p=0; p<n_points; ++p)
      VelocityBoundaryValues<dim>::vector_value(points[p], value_list[p]);
  }


  // Permeability (inverse of the tensor)
  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };

  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));


    for (unsigned int p=0; p<points.size(); ++p)
    {
      values[p].clear ();

      switch(dim)
      {
        case 2:
          values[p][0][0] = pow(points[p][0]+1.0,2.0)/(points[p][0]*4.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])-pow(sin(points[p][0]*points[p][1]),2.0)+points[p][0]*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*6.0+(points[p][0]*points[p][0]*points[p][0])*4.0+points[p][0]*points[p][0]*points[p][0]*points[p][0]+points[p][1]*points[p][1]+1.0);
          values[p][0][1] = -sin(points[p][0]*points[p][1])/(points[p][0]*4.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])-pow(sin(points[p][0]*points[p][1]),2.0)+points[p][0]*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*6.0+(points[p][0]*points[p][0]*points[p][0])*4.0+points[p][0]*points[p][0]*points[p][0]*points[p][0]+points[p][1]*points[p][1]+1.0);
          values[p][1][0] = -sin(points[p][0]*points[p][1])/(points[p][0]*4.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])-pow(sin(points[p][0]*points[p][1]),2.0)+points[p][0]*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*6.0+(points[p][0]*points[p][0]*points[p][0])*4.0+points[p][0]*points[p][0]*points[p][0]*points[p][0]+points[p][1]*points[p][1]+1.0);
          values[p][1][1] = (points[p][0]*2.0+points[p][0]*points[p][0]+points[p][1]*points[p][1]+1.0)/(points[p][0]*4.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])-pow(sin(points[p][0]*points[p][1]),2.0)+points[p][0]*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*6.0+(points[p][0]*points[p][0]*points[p][0])*4.0+points[p][0]*points[p][0]*points[p][0]*points[p][0]+points[p][1]*points[p][1]+1.0);
          break;
        case 3:
          values[p][0][0] = (points[p][1]*1.2E1+cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*9.0+3.5E1/2.0)/(points[p][1]*1.18E2-cos(points[p][0]*points[p][1]*2.0)+cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*(3.5E1/2.0)+(points[p][1]*points[p][1])*(1.47E2/2.0)+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*(7.1E1/2.0)+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)-(points[p][2]*points[p][2])*cos(points[p][0]*points[p][1]*2.0)*(1.0/2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+6.9E1);
          values[p][0][1] = (cos(points[p][0]*points[p][1])*sin(points[p][1]*points[p][2]))/(points[p][1]*1.2E2+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])-pow(cos(points[p][0]*points[p][1]),2.0)*2.0-pow(sin(points[p][1]*points[p][2]),2.0)*4.0-(points[p][2]*points[p][2])*pow(cos(points[p][0]*points[p][1]),2.0)-(points[p][0]*points[p][0])*pow(sin(points[p][1]*points[p][2]),2.0)-(points[p][1]*points[p][1])*pow(sin(points[p][1]*points[p][2]),2.0)+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*1.8E1+(points[p][1]*points[p][1])*7.4E1+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*3.6E1-points[p][1]*pow(sin(points[p][1]*points[p][2]),2.0)*4.0+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+7.2E1);
          values[p][0][2] = (cos(points[p][0]*points[p][1])*(points[p][2]*points[p][2]+2.0)*-2.0)/(points[p][1]*2.36E2+cos(points[p][1]*points[p][2]*2.0)*4.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*4.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*1.8E1+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*7.4E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*2.0-pow(cos(points[p][0]*points[p][1]),2.0)*4.0-(points[p][2]*points[p][2])*pow(cos(points[p][0]*points[p][1]),2.0)*2.0+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*4.0+(points[p][0]*points[p][0])*points[p][1]*2.4E1+points[p][1]*(points[p][2]*points[p][2])*1.2E2+(points[p][0]*points[p][0])*3.5E1+(points[p][1]*points[p][1])*1.47E2+(points[p][1]*points[p][1]*points[p][1])*4.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*4.0+(points[p][2]*points[p][2])*7.2E1+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*1.2E1+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*2.0+1.4E2);
          values[p][1][0] = (cos(points[p][0]*points[p][1])*sin(points[p][1]*points[p][2]))/(points[p][1]*1.2E2+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])-pow(cos(points[p][0]*points[p][1]),2.0)*2.0-pow(sin(points[p][1]*points[p][2]),2.0)*4.0-(points[p][2]*points[p][2])*pow(cos(points[p][0]*points[p][1]),2.0)-(points[p][0]*points[p][0])*pow(sin(points[p][1]*points[p][2]),2.0)-(points[p][1]*points[p][1])*pow(sin(points[p][1]*points[p][2]),2.0)+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*1.8E1+(points[p][1]*points[p][1])*7.4E1+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*3.6E1-points[p][1]*pow(sin(points[p][1]*points[p][2]),2.0)*4.0+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+7.2E1);
          values[p][1][1] = (points[p][1]*6.0E1-cos(points[p][0]*points[p][1]*2.0)*(1.0/2.0)+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])+(points[p][0]*points[p][0])*points[p][1]*6.0+(points[p][0]*points[p][0])*9.0+(points[p][1]*points[p][1])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*1.0E1+points[p][1]*points[p][1]*points[p][1]*points[p][1]+7.1E1/2.0)/(points[p][1]*1.18E2-cos(points[p][0]*points[p][1]*2.0)+cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*(3.5E1/2.0)+(points[p][1]*points[p][1])*(1.47E2/2.0)+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*(7.1E1/2.0)+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)-(points[p][2]*points[p][2])*cos(points[p][0]*points[p][1]*2.0)*(1.0/2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+6.9E1);
          values[p][1][2] = -(sin(points[p][1]*points[p][2])*(points[p][1]*4.0+points[p][0]*points[p][0]+points[p][1]*points[p][1]+4.0))/(points[p][1]*1.18E2-cos(points[p][0]*points[p][1]*2.0)+cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*(3.5E1/2.0)+(points[p][1]*points[p][1])*(1.47E2/2.0)+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*(7.1E1/2.0)+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)-(points[p][2]*points[p][2])*cos(points[p][0]*points[p][1]*2.0)*(1.0/2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+6.9E1);
          values[p][2][0] = (cos(points[p][0]*points[p][1])*(points[p][2]*points[p][2]+2.0)*-2.0)/(points[p][1]*2.36E2+cos(points[p][1]*points[p][2]*2.0)*4.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*4.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*1.8E1+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*7.4E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*2.0-pow(cos(points[p][0]*points[p][1]),2.0)*4.0-(points[p][2]*points[p][2])*pow(cos(points[p][0]*points[p][1]),2.0)*2.0+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*4.0+(points[p][0]*points[p][0])*points[p][1]*2.4E1+points[p][1]*(points[p][2]*points[p][2])*1.2E2+(points[p][0]*points[p][0])*3.5E1+(points[p][1]*points[p][1])*1.47E2+(points[p][1]*points[p][1]*points[p][1])*4.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*4.0+(points[p][2]*points[p][2])*7.2E1+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*1.2E1+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*2.0+1.4E2);
          values[p][2][1] = -(sin(points[p][1]*points[p][2])*(points[p][1]*4.0+points[p][0]*points[p][0]+points[p][1]*points[p][1]+4.0))/(points[p][1]*1.18E2-cos(points[p][0]*points[p][1]*2.0)+cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*(3.5E1/2.0)+(points[p][1]*points[p][1])*(1.47E2/2.0)+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*(7.1E1/2.0)+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)-(points[p][2]*points[p][2])*cos(points[p][0]*points[p][1]*2.0)*(1.0/2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+6.9E1);
          values[p][2][2] = ((points[p][2]*points[p][2]+2.0)*(points[p][1]*4.0+points[p][0]*points[p][0]+points[p][1]*points[p][1]+4.0))/(points[p][1]*1.18E2-cos(points[p][0]*points[p][1]*2.0)+cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*2.0+(points[p][0]*points[p][0])*(points[p][2]*points[p][2])*9.0+(points[p][1]*points[p][1])*(points[p][2]*points[p][2])*3.7E1+(points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])*1.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*(points[p][2]*points[p][2])+points[p][1]*cos(points[p][1]*points[p][2]*2.0)*2.0+(points[p][0]*points[p][0])*points[p][1]*1.2E1+points[p][1]*(points[p][2]*points[p][2])*6.0E1+(points[p][0]*points[p][0])*(3.5E1/2.0)+(points[p][1]*points[p][1])*(1.47E2/2.0)+(points[p][1]*points[p][1]*points[p][1])*2.0E1+(points[p][1]*points[p][1]*points[p][1]*points[p][1])*2.0+(points[p][2]*points[p][2])*(7.1E1/2.0)+(points[p][0]*points[p][0])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)-(points[p][2]*points[p][2])*cos(points[p][0]*points[p][1]*2.0)*(1.0/2.0)+(points[p][1]*points[p][1])*cos(points[p][1]*points[p][2]*2.0)*(1.0/2.0)+(points[p][0]*points[p][0])*points[p][1]*(points[p][2]*points[p][2])*6.0+(points[p][0]*points[p][0])*(points[p][1]*points[p][1])*(points[p][2]*points[p][2])+6.9E1);
          break;
        default:
        Assert(false, ExcNotImplemented());
      }

    }
  }


  // Exact solution
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution() : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
    virtual void vector_gradient (const Point<dim> &p,
                                  std::vector<Tensor<1,dim,double>>  &grads) const;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    switch (dim)
    {
      case 2:
      	// Velocity
        values(0) = -sin(p[0]*p[1])*((p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-sin(p[0]*p[1])*sin(p[1])+p[0]*cos(p[0]*p[1])*cos(p[1]))-(pow(p[0]+1.0,2.0)+p[1]*p[1])*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]));
        values(1) = -pow(p[0]+1.0,2.0)*((p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-sin(p[0]*p[1])*sin(p[1])+p[0]*cos(p[0]*p[1])*cos(p[1]))-sin(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]));
        // Pressure
        values(2) = (p[0]*p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])+p[0]*p[0]+sin(p[0]*p[1])*cos(p[1]);
        break;
      case 3:
        // Velocity:
        values(0) = -cos(p[0]*p[1])*(cos(p[2])+p[1]*p[2]*2.0)-(pow(p[1]+2.0,2.0)+p[0]*p[0])*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]));
        values(1) = -(p[2]*p[2]+2.0)*((p[0]*p[0]*p[0]*p[0])*(p[1]*p[1])*3.0-p[0]*sin(p[0]*p[1])+p[2]*p[2])-sin(p[1]*p[2])*(cos(p[2])+p[1]*p[2]*2.0);
        values(2) = -sin(p[1]*p[2])*((p[0]*p[0]*p[0]*p[0])*(p[1]*p[1])*3.0-p[0]*sin(p[0]*p[1])+p[2]*p[2])-(cos(p[2])+p[1]*p[2]*2.0)*pow(p[1]+3.0,2.0)-cos(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]));
        // Pressure:
        values(3) = cos(p[0]*p[1])+sin(p[2])+(p[0]*p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])+p[1]*(p[2]*p[2])+p[0]*p[0];;
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
    switch (dim)
    {
      case 2:
        // Gradient of the velocity (2x2 matrix)
        grads[0][0] = -(p[0]*2.0+2.0)*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]))-(pow(p[0]+1.0,2.0)+p[1]*p[1])*(p[0]*(p[1]*p[1]*p[1]*p[1])*6.0-(p[1]*p[1])*sin(p[0]*p[1])*cos(p[1])+2.0)-sin(p[0]*p[1])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1+cos(p[0]*p[1])*cos(p[1])-p[1]*cos(p[0]*p[1])*sin(p[1])-p[0]*p[1]*sin(p[0]*p[1])*cos(p[1]))-p[1]*cos(p[0]*p[1])*((p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-sin(p[0]*p[1])*sin(p[1])+p[0]*cos(p[0]*p[1])*cos(p[1]));
        grads[0][1] = p[1]*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]))*-2.0-(pow(p[0]+1.0,2.0)+p[1]*p[1])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1+cos(p[0]*p[1])*cos(p[1])-p[1]*cos(p[0]*p[1])*sin(p[1])-p[0]*p[1]*sin(p[0]*p[1])*cos(p[1]))+sin(p[0]*p[1])*((p[0]*p[0]*p[0])*(p[1]*p[1])*-1.2E1+sin(p[0]*p[1])*cos(p[1])+p[0]*cos(p[0]*p[1])*sin(p[1])*2.0+(p[0]*p[0])*sin(p[0]*p[1])*cos(p[1]))-p[0]*cos(p[0]*p[1])*((p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-sin(p[0]*p[1])*sin(p[1])+p[0]*cos(p[0]*p[1])*cos(p[1]));
        grads[1][0] = -sin(p[0]*p[1])*(p[0]*(p[1]*p[1]*p[1]*p[1])*6.0-(p[1]*p[1])*sin(p[0]*p[1])*cos(p[1])+2.0)-pow(p[0]+1.0,2.0)*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1+cos(p[0]*p[1])*cos(p[1])-p[1]*cos(p[0]*p[1])*sin(p[1])-p[0]*p[1]*sin(p[0]*p[1])*cos(p[1]))-(p[0]*2.0+2.0)*((p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-sin(p[0]*p[1])*sin(p[1])+p[0]*cos(p[0]*p[1])*cos(p[1]))-p[1]*cos(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]));
        grads[1][1] = pow(p[0]+1.0,2.0)*((p[0]*p[0]*p[0])*(p[1]*p[1])*-1.2E1+sin(p[0]*p[1])*cos(p[1])+p[0]*cos(p[0]*p[1])*sin(p[1])*2.0+(p[0]*p[0])*sin(p[0]*p[1])*cos(p[1]))-sin(p[0]*p[1])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1+cos(p[0]*p[1])*cos(p[1])-p[1]*cos(p[0]*p[1])*sin(p[1])-p[0]*p[1]*sin(p[0]*p[1])*cos(p[1]))-p[0]*cos(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0])*(p[1]*p[1]*p[1]*p[1])*3.0+p[1]*cos(p[0]*p[1])*cos(p[1]));
        // Rest is meaningles, pressure is L2
        grads[2][0] = 0;
        grads[2][1] = 0;
        break;
      case 3:
        // Gradient of the velocity (3x3 matrix)
        grads[0][0] = -(pow(p[1]+2.0,2.0)+p[0]*p[0])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1-(p[1]*p[1])*cos(p[0]*p[1])+2.0)-p[0]*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]))*2.0+p[1]*sin(p[0]*p[1])*(cos(p[2])+p[1]*p[2]*2.0);
        grads[0][1] = (pow(p[1]+2.0,2.0)+p[0]*p[0])*(sin(p[0]*p[1])-(p[0]*p[0]*p[0])*(p[1]*p[1])*1.2E1+p[0]*p[1]*cos(p[0]*p[1]))-(p[1]*2.0+4.0)*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]))-p[2]*cos(p[0]*p[1])*2.0+p[0]*sin(p[0]*p[1])*(cos(p[2])+p[1]*p[2]*2.0);
        grads[0][2] = -cos(p[0]*p[1])*(p[1]*2.0-sin(p[2]));
        grads[1][0] = (p[2]*p[2]+2.0)*(sin(p[0]*p[1])-(p[0]*p[0]*p[0])*(p[1]*p[1])*1.2E1+p[0]*p[1]*cos(p[0]*p[1]));
        grads[1][1] = -(p[2]*p[2]+2.0)*((p[0]*p[0]*p[0]*p[0])*p[1]*6.0-(p[0]*p[0])*cos(p[0]*p[1]))-p[2]*sin(p[1]*p[2])*2.0-p[2]*cos(p[1]*p[2])*(cos(p[2])+p[1]*p[2]*2.0);
        grads[1][2] = p[2]*(p[2]*p[2]+2.0)*-2.0-sin(p[1]*p[2])*(p[1]*2.0-sin(p[2]))-p[2]*((p[0]*p[0]*p[0]*p[0])*(p[1]*p[1])*3.0-p[0]*sin(p[0]*p[1])+p[2]*p[2])*2.0-p[1]*cos(p[1]*p[2])*(cos(p[2])+p[1]*p[2]*2.0);
        grads[2][0] = sin(p[1]*p[2])*(sin(p[0]*p[1])-(p[0]*p[0]*p[0])*(p[1]*p[1])*1.2E1+p[0]*p[1]*cos(p[0]*p[1]))-cos(p[0]*p[1])*((p[0]*p[0])*(p[1]*p[1]*p[1])*1.2E1-(p[1]*p[1])*cos(p[0]*p[1])+2.0)+p[1]*sin(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]));
        grads[2][1] = p[2]*pow(p[1]+3.0,2.0)*-2.0-sin(p[1]*p[2])*((p[0]*p[0]*p[0]*p[0])*p[1]*6.0-(p[0]*p[0])*cos(p[0]*p[1]))-(p[1]*2.0+6.0)*(cos(p[2])+p[1]*p[2]*2.0)+cos(p[0]*p[1])*(sin(p[0]*p[1])-(p[0]*p[0]*p[0])*(p[1]*p[1])*1.2E1+p[0]*p[1]*cos(p[0]*p[1]))-p[2]*cos(p[1]*p[2])*((p[0]*p[0]*p[0]*p[0])*(p[1]*p[1])*3.0-p[0]*sin(p[0]*p[1])+p[2]*p[2])+p[0]*sin(p[0]*p[1])*(p[0]*2.0+(p[0]*p[0]*p[0])*(p[1]*p[1]*p[1])*4.0-p[1]*sin(p[0]*p[1]));
        grads[2][2] = -(p[1]*2.0-sin(p[2]))*pow(p[1]+3.0,2.0)-p[2]*sin(p[1]*p[2])*2.0-p[1]*cos(p[1]*p[2])*((p[0]*p[0]*p[0]*p[0])*(p[1]*p[1])*3.0-p[0]*sin(p[0]*p[1])+p[2]*p[2]);
        // Rest is meaningles, pressure is L2
        grads[3][0] = 0.0;
        grads[3][1] = 0.0;
        grads[3][2] = 0.0;
        break;
      default:
      Assert(false, ExcNotImplemented());
    }
  }
}

#endif /*PEFLOW_DARCY_DATA_H*/
