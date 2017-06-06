//
// Created by eldar on 6/4/17.
//

#include <deal.II/base/multithread_info.h>
#include "../inc/darcy_mfe.h"
#include "../inc/darcy_mfmfe.h"
#include "../inc/elasticity_mfe.h"
#include "../inc/elasticity_msmfe.h"

// Main function
int main()
{
  try {
    using namespace dealii;
    using namespace darcy;
    using namespace elasticity;

    MultithreadInfo::set_thread_limit();

    std::ofstream log_file("log_file.txt");
    deallog.attach(log_file);
    deallog.depth_file(2);

    //MultipointMixedDarcyProblem<2> mixed_darcy_problem_2d(3);
    //mixed_darcy_problem_2d.run(3,0);
    //mixed_darcy_problem_2d.run(7,0);

    MultipointMixedElasticityProblem<2> mixed_elasticity_problem_2d(2);
    //MixedElasticityProblem<2> mixed_elasticity_problem_2d(2);
    mixed_elasticity_problem_2d.run(5);


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