// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_ELASTICITY_PARAMETER_READER_H
#define PEFLOW_ELASTICITY_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

namespace elasticity
{
  using namespace dealii;

class ElasticityParameterReader : public Subscriptor
  {
  public:
    ElasticityParameterReader(ParameterHandler &paramhandler) : prm(paramhandler) {}
    inline void read_parameters(const std::string);
  private:
    inline void declare_parameters();
    ParameterHandler &prm;
  };

  inline void ElasticityParameterReader::declare_parameters()
  {
    prm.declare_entry("degree", "0",
                      Patterns::Integer());
    prm.declare_entry("refinements", "1",
                      Patterns::Integer());
    prm.declare_entry("grid_flag", "0",
                      Patterns::Integer());

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

    prm.enter_subsection("RHS 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("RHS 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("BC 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,2);
      prm.set("Function expression", "0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("BC 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,7);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,15);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,14);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,45);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();
  }
  inline void ElasticityParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.parse_input (parameter_file);
  }
}

#endif //PEFLOW_ELASTICITY_PARAMETER_READER_H
