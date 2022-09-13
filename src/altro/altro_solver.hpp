//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "solver_options.hpp"
#include "typedefs.hpp"

namespace altro {

class SolverImpl;  // Forward-declare the implementation. Note this is public, but allows for the
                   // implementation to be defined in a separate static library

class ALTROSolver {
 public:
  ALTROSolver(int horizon_length);                     // Constructor
  ALTROSolver(const ALTROSolver& other);               // Copy constructor
  ALTROSolver(ALTROSolver&& other);                    // Move constructor
  ALTROSolver& operator=(const ALTROSolver& other);    // Copy assignment
  ALTROSolver& operator=(ALTROSolver&& other);         // Move assignment
  ~ALTROSolver();                                      // Destructor

  /**********************************************
   * Problem definition
   **********************************************/

  /**
   * @brief Set the number of states and inputs over a range of time steps
   *
   * If only one time step is specified, sets the dimensions that time step.
   * If two are specified, the dimensions are set for all time steps from
   * `k_start` up to, but not including, `k_stop`.
   *
   * @param num_states Number of states (size of state vector).
   * @param num_inputs Number of inputs (size of input / control vector).
   * @param k_start Knot point index.
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   *                          If specified, sets all of the knot points in range `[k_start, k_stop)`
   */
  void SetDimension(int num_states, int num_inputs, int k_start, int k_stop = 0);

  /**
   * @brief Specify the dynamics function and dynamics Jacobian at a knot point (or range or knot
   * points)
   *
   * The discrete dynamics are assumed to be of the form:
   *
   * \f[
   * x_{k+1} = f(x_k, u_k, t, h)
   * \f]
   *
   * where \f$x_{k+1} \in \mathbb{R}^{n_2}, x_k \in \mathbb{R}^{n_1}, u_k \in \mathbb{R}^{m_1}\f$.
   *
   * The Jacobian is a `(n2,n1 + m1)` matrix, where the first `n1` columns are the derivatives
   * with respect to `x`, and the last `m1` columns are the derivatives with respect to `u`.
   *
   * The functions can be specified as any callable object (anything that be represented as
   * `std::function`).
   *
   * @param dynamics_function A function pointer of the form
   *        `void(double *xnext, const double* x, const double* u, float t, float h)`
   * @param dynamics_jacobian
   *        `void(double *jac, const double* x, const double* u, float t, float h)`
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   */
  void SetExplicitDynamics(ExplicitDynamicsFunction dynamics_function,
                           ExplicitDynamicsJacobian dynamics_jacobian, int k_start, int k_stop = 0);

  /**
   * @brief Specify the dynamics function and dynamics Jacobian at a knot point (or range or knot
   * points)
   *
   * The discrete dynamics are assumed to be of the form:
   *
   * \f[
   * f(x_k, u_k, x_{k+1}, u_{k+1}, t, h)
   * \f]
   *
   * where \f$x_{k+1} \in \mathbb{R}^{n_2}, u_{k+1} \in \mathbb{R}^{m_2}, x_k \in \mathbb{R}^{n_1},
   * u_k \in \mathbb{R}^{m_1}\f$.
   *
   * The Jacobian is a `(n2,n1 + m1)` matrix, where the first `n1` columns are the derivatives
   * with respect to `x`, and the last `m1` columns are the derivatives with respect to `u`.
   *
   * The functions can be specified as any callable object (anything that be represented as
   * `std::function`).
   *
   * @param dynamics_function A function pointer of the form
   *        `void(double* xnext, const double* x1, const double* x2, const double* x2, const double*
   *              u2, float t, float h)`
   * @param dynamics_jacobian
   *        `void(double* jac, const double* x1, const double* x2, const double* x2, const double*
   *              u2, float t, float h)`
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   */
  void SetImplicitDynamics(ImplicitDynamicsFunction dynamics_function,
                           ImplicitDynamicsJacobian dynamics_jacobian, int k_start, int k_stop = 0);

  /**
   * @brief Set the cost function at a knot point (or range of knot points)
   *
   * Cost function is of the form:
   * \f[
   *    \ell(x,u)
   * \f]
   *
   * The function can be specified as any callable object (anything that be represented as
   * `std::function`).
   *
   * @param cost_function A function pointer of the form
   * @param k_start
   * @param k_stop
   */
  void SetCostFunction(CostFunction cost_function, int k_start, int k_stop = 0);

  /**
   * @brief Set cost function properties
   *
   * @param is_quadratic Is the cost function a convex quadratic?
   * @param Q_is_diagonal Is the Hessian wrt x a diagonal matrix?
   * @param R_is_diagonal Is the Hessian wrt u a diagonal matrix?
   * @param H_is_zero Are the states and inputs independent?
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   */
  void SetCostFunctionProperties(bool is_quadratic, bool Q_is_diagonal, bool R_is_diagonal,
                                 bool H_is_zero, int k_start, int k_stop = 0);

  /**
   * @brief Set constraint function at a knot point (or range of knot points)
   *
   * Assumes constraint is of the form
   * \f[
   *    c(x,u) \in \mathcal{K}
   * \f]
   *
   * The functions can be specified as any callable object (anything that be represented as
   * `std::function`).
   *
   * @param constraint_function Constraint function of the form
   *    `void(a_float* val, const a_float* x, const a_float* u)`.
   * @param constraint_jacobian Constraint jacobian of the form
   *    `void(a_float* jac, const a_float* x, const a_float* u)`.
   * @param dim Length of the constraint
   * @param constraint_type One of the valid constraint types.
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   * @return ConstraintIndex An opaque class that uniquely represents the constraint. Should be
   * saved if information about the constraint needs to be queried later.
   */
  ConstraintIndex SetConstraint(ConstraintFunction constraint_function,
                                ConstraintJacobian constraint_jacobian, int dim,
                                ConstraintType constraint_type, int k_start, int k_stop = 0);

  /**
   * @brief Set the upper bound on the states at an index (or a range of knot point indices)
   *
   * The elements can be set to positive infinity to leave them unbounded.
   *
   * @param x_max Upper bound on the states
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetStateUpperBound(a_float* x_max, int k_start, int k_stop = 0);

  /**
   * @brief Set the lower bound on the states at an index (or a range of knot point indices)
   *
   * The elements can be set to negative infinity to leave them unbounded.
   *
   * @param x_min Lower bound on the states
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetStateLowerBound(a_float* x_min, int k_start, int k_stop = 0);

  /**
   * @brief Set the upper bound on the inputs at an index (or a range of knot point indices)
   *
   * The elements can be set to positive infinity to leave them unbounded.
   *
   * @param u_max Upper bound on the inputs
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetInputUpperBound(a_float* u_max, int k_start, int k_stop = 0);

  /**
   * @brief Set the lower bound on the inputs at an index (or a range of knot point indices)
   *
   * The elements can be set to negative infinity to leave them unbounded.
   *
   * @param u_min Lower bound on the inputs
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetInputLowerBound(a_float* u_min, int k_start, int k_stop = 0);

  /**
   * @brief Check if solver has been initialized.
   *
   * Changes to the problem definition (number of knot points, cost, dynamics, constraints
   * dimensions and definitions) can only be made before the solver initialized.
   *
   * @return
   */
  bool InInitialized() const;

  /**********************************************
   * Initialization
   **********************************************/
  void Initialize();

  /**
   * @brief Set the state at a time step (or range of time steps). Used as the initial guess for the
   * trajectory.
   *
   * @pre Solver must be initialized.
   * @param x State vector
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetState(const a_float* x, int k_start, int k_stop = 0);

  /**
   * @brief Set the input at a time step (or range of time steps). Used as the initial guess for the
   * trajectory.
   *
   * @pre Solver must be initialized.
   * @param u Input vector
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetInput(const a_float* u, int k_start, int k_stop = 0);

  /**
   * @brief Set the dual variable associated with dynamics constraint at a time step (or range of
   * time steps). Used as the initial guess for the trajectory.
   *
   * @pre Solver must be initialized.
   * @param y Dynamics dual
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  void SetDualDynamics(const a_float* y, int k_start, int k_stop = 0);

  /**
   * @brief Set the dual variable for a general constraint
   *
   * @pre Solver must be initialized.
   * @param z Dual variable for the constraint
   * @param constraint_index Constraint index returned from `SetConstraint`
   */
  void SetDualGeneric(const a_float* z, const ConstraintIndex& constraint_index);

  void SetDualStateUpperBound(const a_float* b_x_upper, int k_start, int k_stop = 0);
  void SetDualStateLowerBound(const a_float* b_x_lower, int k_start, int k_stop = 0);
  void SetDualInputUpperBound(const a_float* b_u_upper, int k_start, int k_stop = 0);
  void SetDualInputLowerBound(const a_float* b_u_lower, int k_start, int k_stop = 0);

  /**********************************************
   * Options
   **********************************************/
  void SetOptions(SolverOptions opts);
  SolverOptions& GetOptions();
  const SolverOptions& GetOptions() const;

  /**
   * @brief Set a callback function that is call each iteration (of iLQR).
   *
   * Specify a callback function that is called each iteration of the solver. The function takes
   * a const pointer to the solver. The function can implement the pointer always points to a valid
   * and initialized instance of the solver.
   *
   * @param callback Callback function with the following signature:
   *    `void(const ALTROSolver*)
   */
  void SetCallback(CallbackFunction callback);

  /**********************************************
   * Solve
   **********************************************/
  SolveStatus solve();

  SolveStatus GetStatus() const;
  int GetIterations() const;
  int GetSolveTime() const;
  double GetPrimalFeasibility() const;
  double GetFinalObjective() const;

  /**********************************************
   * Getters
   **********************************************/
  void GetFeedbackGain(a_float* K, int k);
  void GetFeedforwardGain(a_float* d, int k);
  void GetState(a_float* x, int k);
  void GetInput(a_float* u, int k);
  void GetDualDynamics(a_float* y, int k);
  void GetDualGeneral(a_float* z, int ConstraintIndex);
  void GetDualStateUpperBound(a_float* b_x_upper, int ConstraintIndex);
  void GetDualStateLowerBound(a_float* b_x_lower, int ConstraintIndex);
  void GetDualInputUpperBound(a_float* b_u_upper, int ConstraintIndex);
  void GetDualInputLowerBound(a_float* b_u_lower, int ConstraintIndex);

 private:
  std::unique_ptr<SolverImpl> solver_;
//  SolverImpl *solver_;
};

}  // namespace altro
