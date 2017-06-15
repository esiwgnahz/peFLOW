// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_BIOT_PARAMETER_READER_H
#define PEFLOW_BIOT_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

namespace biot
{
  using namespace dealii;

class BiotParameterReader : public Subscriptor
  {
  public:
    BiotParameterReader(ParameterHandler &paramhandler) : prm(paramhandler) {}
    inline void read_parameters(const std::string);
  private:
    inline void declare_parameters();
    ParameterHandler &prm;
  };

  inline void BiotParameterReader::declare_parameters()
  {
    prm.declare_entry("degree", "0",
                      Patterns::Integer());
    prm.declare_entry("refinements", "1",
                      Patterns::Integer());
    prm.declare_entry("grid_flag", "0",
                      Patterns::Integer());
    prm.declare_entry("time_step", "0.1",
                      Patterns::Double());
    prm.declare_entry("num_time_steps", "1",
                      Patterns::Integer());
    prm.declare_entry("alpha", "1.0",
                      Patterns::Double());
    prm.declare_entry("Storativity", "1.0",
                      Patterns::Double());

    prm.enter_subsection("permeability 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm, 4);
      prm.set("Function expression", "1; 0; 0; 1");
    }
    prm.leave_subsection();

    prm.enter_subsection("permeability 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm, 9);
      prm.set("Function expression", "1; 0; 0; 0; 1; 0; 0; 0; 1");
    }
    prm.leave_subsection();

    prm.enter_subsection("lambda 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "123");
    }
    prm.leave_subsection();

    prm.enter_subsection("lambda 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "1");
    }
    prm.leave_subsection();

    prm.enter_subsection("mu 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "79.3");
    }
    prm.leave_subsection();

    prm.enter_subsection("mu 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "1");
    }
    prm.leave_subsection();

    prm.enter_subsection("Darcy RHS 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Darcy RHS 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Elasticity RHS 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Elasticity RHS 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Darcy BC 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Darcy BC 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Elasticity BC 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Elasticity BC 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Initial conditions 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,10);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Initial conditions 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,19);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,10);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,19);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,20);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,57);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();
  }
  inline void BiotParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.parse_input (parameter_file);
  }
}

#endif //PEFLOW_BIOT_PARAMETER_READER_H
