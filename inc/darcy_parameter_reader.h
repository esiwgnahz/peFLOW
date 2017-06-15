// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#ifndef PEFLOW_DARCY_PARAMETER_READER_H
#define PEFLOW_DARCY_PARAMETER_READER_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

namespace darcy
{
  using namespace dealii;

class DarcyParameterReader : public Subscriptor
  {
  public:
    DarcyParameterReader(ParameterHandler &paramhandler) : prm(paramhandler) {}
    inline void read_parameters(const std::string);
  private:
    inline void declare_parameters();
    ParameterHandler &prm;
  };

  inline void DarcyParameterReader::declare_parameters()
  {
    prm.declare_entry("degree", "0",
                      Patterns::Integer());
    prm.declare_entry("refinements", "1",
                      Patterns::Integer());
    prm.declare_entry("grid_flag", "0",
                      Patterns::Integer());

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

    prm.enter_subsection("RHS 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("RHS 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("BC 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("BC 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm);
      prm.set("Function expression", "0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,3);
      prm.set("Function expression", "0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact solution 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,4);
      prm.set("Function expression", "0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient 2D");
    {
      Functions::ParsedFunction<2>::declare_parameters(prm,6);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();

    prm.enter_subsection("Exact gradient 3D");
    {
      Functions::ParsedFunction<3>::declare_parameters(prm,12);
      prm.set("Function expression", "0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0");
    }
    prm.leave_subsection();
  }
  inline void DarcyParameterReader::read_parameters (const std::string parameter_file)
  {
    declare_parameters();
    prm.parse_input (parameter_file);
  }
}

#endif //PEFLOW_DARCY_PARAMETER_READER_H
