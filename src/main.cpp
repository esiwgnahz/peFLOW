// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------

#include <deal.II/base/multithread_info.h>
#include "../inc/biot_mfe.h"
#include "../inc/biot_parameter_reader.h"
#include "../inc/darcy_parameter_reader.h"
#include "../inc/elasticity_parameter_reader.h"
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

    unsigned int flag;
    std::cout << " Choose the model to run: " << std::endl;
    std::cout << "1: Mixed Darcy Problem, 2D" <<std::endl;
    std::cout << "2: Mixed Darcy Problem, 3D" <<std::endl;
    std::cout << "3: Multipoint Mixed Darcy Problem, 2D" <<std::endl;
    std::cout << "4: Multipoint Mixed Darcy Problem, 3D" <<std::endl;
    std::cout << "5: Mixed Elasticity Problem, 2D" <<std::endl;
    std::cout << "6: Mixed Elasticity Problem, 3D" <<std::endl;
    std::cout << "7: Multipoint Mixed Elasticity Problem, 2D" <<std::endl;
    std::cout << "8: Multipoint Mixed Elasticity Problem, 3D" <<std::endl;
    std::cout << "9: Mixed Biot Problem, 2D" <<std::endl;

   // std::cin >> flag;
    flag = 6;

    std::cout << "============================================" <<std::endl;

    ParameterHandler  prm;

    if(flag == 1){
        std::cout << "Mixed Darcy, 2D case: " << std::endl;

        DarcyParameterReader   param(prm);
        param.read_parameters("parameters_darcy.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MixedDarcyProblem<2> mixed_darcy_problem_2d(degree, prm);
        mixed_darcy_problem_2d.run(refinements,grid);
      }
    else if(flag == 2){
        std::cout << "Mixed Darcy, 3D case: " << std::endl;

        DarcyParameterReader   param(prm);
        param.read_parameters("parameters_darcy.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MixedDarcyProblem<3> mixed_darcy_problem_3d(degree, prm);
        mixed_darcy_problem_3d.run(refinements,grid);
      }
    else if(flag == 3){
        std::cout << "Multipoint Mixed Darcy, 2D case: " << std::endl;

        DarcyParameterReader   param(prm);
        param.read_parameters("parameters_darcy.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MultipointMixedDarcyProblem<2> multipoint_mixed_darcy_problem_2d(degree, prm);
        multipoint_mixed_darcy_problem_2d.run(refinements,grid);
      }
    else if(flag == 4){
        std::cout << "Multipoint Mixed Darcy, 3D case: " << std::endl;

        DarcyParameterReader   param(prm);
        param.read_parameters("parameters_darcy.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MultipointMixedDarcyProblem<3> multipoint_mixed_darcy_problem_3d(degree, prm);
        multipoint_mixed_darcy_problem_3d.run(refinements,grid);
      }
    else if(flag == 5){
        std::cout << "Mixed Elasticity, 2D case: " << std::endl;

        ElasticityParameterReader   param(prm);
        param.read_parameters("parameters_elasticity.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MixedElasticityProblem<2> mixed_elasticity_problem_2d(degree, prm);
        mixed_elasticity_problem_2d.run(refinements,grid);
      }
    else if(flag == 6){
        std::cout << "Mixed Elasticity, 3D case: " << std::endl;

        ElasticityParameterReader   param(prm);
        param.read_parameters("parameters_elasticity.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MixedElasticityProblem<3> mixed_elasticity_problem_3d(degree, prm);
        mixed_elasticity_problem_3d.run(refinements,grid);
      }
    else if(flag == 7){
        std::cout << "Multipoint Mixed Elasticity, 2D case: " << std::endl;

        ElasticityParameterReader   param(prm);
        param.read_parameters("parameters_elasticity.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MultipointMixedElasticityProblem<2> multipoint_mixed_elasticity_problem_2d(degree, prm);
        multipoint_mixed_elasticity_problem_2d.run(refinements,grid);
      }
    else if(flag == 8){
        std::cout << "Multipoint Mixed Elasticity, 3D case: " << std::endl;

        ElasticityParameterReader   param(prm);
        param.read_parameters("parameters_elasticity.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");

        MultipointMixedElasticityProblem<3> multipoint_mixed_elasticity_problem_3d(degree, prm);
        multipoint_mixed_elasticity_problem_3d.run(refinements,grid);
      }
    else if(flag == 9){
        std::cout << "Mixed Biot, 2D case: " << std::endl;

        BiotParameterReader   param(prm);
        param.read_parameters("parameters_biot.prm");

        // Get parameters
        const unsigned int degree = prm.get_integer("degree");
        const unsigned int grid  = prm.get_integer("grid_flag");
        const unsigned int refinements = prm.get_integer("refinements");
        const double time_step = prm.get_double("time_step");
        const unsigned int num_time_steps = prm.get_integer("num_time_steps");

        MixedBiotProblem<2> mixed_biot_problem_2d(degree,prm, time_step, num_time_steps);
        mixed_biot_problem_2d.run(refinements,grid);
      }



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
