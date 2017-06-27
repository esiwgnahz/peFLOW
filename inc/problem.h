// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 Ilona Ambartsumyan, Eldar Khattatov
//
// This file is part of peFLOW.
//
// ---------------------------------------------------------------------
#ifndef PEFLOW_PROBLEM_H
#define PEFLOW_PROBLEM_H

/*
 * Interface class with only one pure virtual function void run().
 */
template <int dim>
class Problem
{
public:
  virtual ~Problem() {};
  virtual void run (const unsigned int refine, const unsigned int grid) = 0;
};

#endif //PEFLOW_PROBLEM_H
