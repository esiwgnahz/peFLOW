// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumya, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/multithread_info.h>
#include "../inc/biot_mfe.h"
#include "../inc/biot_parameter_reader.h"
#include "../inc/darcy_mfe.h"
#include "../inc/darcy_mfmfe.h"
#include "../inc/elasticity_mfe.h"
#include "../inc/elasticity_msmfe.h"

//#include <deal.II/base/parameter_handler.h>

// Main function
int main()
{
  try {
    using namespace dealii;
    using namespace darcy;
    using namespace elasticity;
    using namespace biot;

    MultithreadInfo::set_thread_limit();

    std::ofstream log_file("log_file.txt");
    deallog.attach(log_file);
    deallog.depth_file(2);

    //MultipointMixedDarcyProblem<2> mixed_darcy_problem_2d(3);
    //mixed_darcy_problem_2d.run(3,0);
    //mixed_darcy_problem_2d.run(7,0);

    //MultipointMixedElasticityProblem<2> mixed_elasticity_problem_2d(2);
    //MixedElasticityProblem<2> mixed_elasticity_problem_2d(2);
    //mixed_elasticity_problem_2d.run(5);

    ParameterHandler  prm;
    ParameterReader   param(prm);

    param.read_parameters("parameters_new.prm");
    // Get degree, grid and time discretization parameters
    const unsigned int degree = prm.get_integer("degree");
    const unsigned int grid  = prm.get_integer("grid_flag");
    const unsigned int refinements = prm.get_integer("refinements");
    const double time_step = prm.get_double("time_step");
    const unsigned int num_time_steps = prm.get_integer("num_time_steps");

    std::cout << "2D case: " << "\n";
    MixedBiotProblem<2> mixed_biot_problem_2d(degree,prm, time_step, num_time_steps);

    mixed_biot_problem_2d.run(refinements,grid);


  } catch (std::exception &exc) {
    std::cerr << std::endl << std:: endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch(...) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }

  return 0;
}
