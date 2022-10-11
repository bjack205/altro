//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "altro/solver/exceptions.hpp"
#include "altro/solver/solver_options.hpp"
#include "altro/solver/typedefs.hpp"

namespace altro {

class SolverImpl;  // Forward-declare the implementation. Note this is public, but allows for the
                   // implementation to be defined in a separate static library

class ALTROSolver {
 public:
  explicit ALTROSolver(int horizon_length);          // Constructor
  ALTROSolver(const ALTROSolver& other);             // Copy constructor
  ALTROSolver(ALTROSolver&& other);                  // Move constructor
  ALTROSolver& operator=(const ALTROSolver& other);  // Copy assignment
  ALTROSolver& operator=(ALTROSolver&& other);       // Move assignment
  ~ALTROSolver();                                    // Destructor

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
  ErrorCodes SetDimension(int num_states, int num_inputs, int k_start = AllIndices, int k_stop = 0);

  /**
   * @brief Set the time step between knot point index `k` and `k + 1`.
   * @param h Time step
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetTimeStep(float h, int k_start = AllIndices, int k_stop = 0);

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
  ErrorCodes SetExplicitDynamics(ExplicitDynamicsFunction dynamics_function,
                                 ExplicitDynamicsJacobian dynamics_jacobian,
                                 int k_start = AllIndices, int k_stop = 0);

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
  ErrorCodes SetImplicitDynamics(ImplicitDynamicsFunction dynamics_function,
                                 ImplicitDynamicsJacobian dynamics_jacobian, int k_start,
                                 int k_stop = 0);

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
   *           `a_float(const a_float *x, const a_float *u)`
   * @param cost_gradient A function pointer of the form
   *           `void(a_float *dx, a_float *du, const a_float *x, const a_float *u)`
   * @param cost_hessian A function pointer of the form
   *           `void(a_float *ddx, a_float *ddu, a_float *dxdu, const a_float *x, const a_float *u)`
   * @param k_start
   * @param k_stop
   */
  ErrorCodes SetCostFunction(CostFunction cost_function, CostGradient cost_gradient,
                             CostHessian cost_hessian, int k_start = AllIndices, int k_stop = 0);

  //  /**
  //   * @brief Set cost function properties
  //   *
  //   * @param is_quadratic Is the cost function a convex quadratic?
  //   * @param Q_is_diagonal Is the Hessian wrt x a diagonal matrix?
  //   * @param R_is_diagonal Is the Hessian wrt u a diagonal matrix?
  //   * @param H_is_zero Are the states and inputs independent?
  //   * @param k_start Knot point index
  //   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
  //   */
  //  ErrorCodes SetCostFunctionProperties(bool is_quadratic, bool Q_is_diagonal, bool
  //  R_is_diagonal,
  //                                       bool H_is_zero, int k_start, int k_stop = 0);

  /**
   * @brief Define a quadratic cost with diagonal cost matrices
   *
   * Cost of the form:
   * \f[
   * \frac{1}{2} x^T Q x + q^T x + \frac{1}{2} u^T R u + r^T u + c
   * \f]
   * where \f$Q\f$ and \f$R\f$ are diagonal positive semi-definite matrices.
   *
   * @param Q_diag (n,) diagonal of quadratic state penalty matrix
   * @param R_diag (m,) diagonal of quadratic input penalty matrix
   * @param q (n,) linear state term
   * @param r (m,) linear input term
   * @param c constant term
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   */
  ErrorCodes SetDiagonalCost(int num_states, int num_inputs, const a_float* Q_diag,
                             const a_float* R_diag, const a_float* q, const a_float* r, a_float c,
                             int k_start = AllIndices, int k_stop = 0);

  /**
   * @brief Set a general (dense) quadratic cost
   *
   * Cost of the form:
   * \f[
   * \frac{1}{2} x^T Q x + q^T x + \frac{1}{2} u^T R u + r^T u + u^T H x + c
   * \f]
   * where \f$Q\f$ and \f$R\f$ are symmetric positive semi-definite matrices.
   *
   * All matrices are assumed to be stored in *column-major* format.
   *
   * @param Q (n,n) quadratic state penalty matrix
   * @param R (m,m) quadratic input penalty matrix
   * @param H (m,n) quadratic cross-term penalty matrix
   * @param q (n,) linear state term
   * @param r (m,) linear input term
   * @param c constant term
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   */
  ErrorCodes SetQuadraticCost(int num_states, int num_inputs, const a_float* Q, const a_float* R,
                              const a_float* H, const a_float* q, const a_float* r, a_float c,
                              int k_start = AllIndices, int k_stop = 0);

  /**
   * @brief Set an LQR tracking objective
   *
   * Cost of the form:
   * \f[
   * \frac{1}{2} (x - x_\text{ref})^T Q (x - x_\text{ref})
   * + \frac{1}{2} (u - u_\text{ref})^T R (u - u_\text{ref})
   * \f]
   *
   * @param Q_diag (n,) diagonal of quadratic state penalty matrix
   * @param R_diag (m,) diagonal of quadratic input penalty matrix
   * @param x_ref (n,) state reference
   * @param u_ref (m,) input reference
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal (non-inclusive) knot point index.
   */
  ErrorCodes SetLQRCost(int num_states, int num_inputs, const a_float* Q_diag,
                        const a_float* R_diag, const a_float* x_ref, const a_float* u_ref,
                        int k_start, int k_stop = 0);

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
   * @param label A short descriptive label for the constraint
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   * @return ConstraintIndex An opaque class that uniquely represents the constraint. Should be
   * saved if information about the constraint needs to be queried later.
   */
  ErrorCodes SetConstraint(ConstraintFunction constraint_function,
                           ConstraintJacobian constraint_jacobian, int dim,
                           ConstraintType constraint_type, std::string label, int k_start,
                           int k_stop = 0, std::vector<ConstraintIndex>* con_inds = nullptr);

  /**
   * @brief Set the upper bound on the states at an index (or a range of knot point indices)
   *
   * The elements can be set to positive infinity to leave them unbounded.
   *
   * @param x_max Upper bound on the states
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetStateUpperBound(a_float* x_max, int k_start, int k_stop = 0);

  /**
   * @brief Set the lower bound on the states at an index (or a range of knot point indices)
   *
   * The elements can be set to negative infinity to leave them unbounded.
   *
   * @param x_min Lower bound on the states
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetStateLowerBound(a_float* x_min, int k_start, int k_stop = 0);

  /**
   * @brief Set the upper bound on the inputs at an index (or a range of knot point indices)
   *
   * The elements can be set to positive infinity to leave them unbounded.
   *
   * @param u_max Upper bound on the inputs
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetInputUpperBound(a_float* u_max, int k_start, int k_stop = 0);

  /**
   * @brief Set the lower bound on the inputs at an index (or a range of knot point indices)
   *
   * The elements can be set to negative infinity to leave them unbounded.
   *
   * @param u_min Lower bound on the inputs
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetInputLowerBound(a_float* u_min, int k_start, int k_stop = 0);

  /**
   * @brief Check if solver has been initialized.
   *
   * Changes to the problem definition (number of knot points, cost, dynamics, constraints
   * dimensions and definitions) can only be made before the solver initialized.
   *
   * @return
   */
  bool IsInitialized() const;

  /**********************************************
   * Initialization
   **********************************************/
  ErrorCodes Initialize();

  /**
   * @brief Set the initial state
   *
   * @pre The first dimension must be specified with ::SetDimension
   * @param x0 Initial state
   */
  ErrorCodes SetInitialState(const double* x0, int n);

  /**
   * @brief Set the state at a time step (or range of time steps). Used as the initial guess for the
   * trajectory.
   *
   * @pre Solver must be initialized.
   * @param x State vector
   * @param n Length of state vector
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetState(const a_float* x, int n, int k_start = AllIndices, int k_stop = 0);

  /**
   * @brief Set the input at a time step (or range of time steps). Used as the initial guess for the
   * trajectory.
   *
   * @pre Solver must be initialized.
   * @param u Input vector
   * @param m Length of input vector
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetInput(const a_float* u, int m, int k_start = AllIndices, int k_stop = 0);

  /**
   * @brief Set the dual variable associated with dynamics constraint at a time step (or range of
   * time steps). Used as the initial guess for the trajectory.
   *
   * @pre Solver must be initialized.
   * @param y Dynamics dual
   * @param k_start Knot point index
   * @param k_stop (optional) Terminal knot point index (non-inclusive).
   */
  ErrorCodes SetDualDynamics(const a_float* y, int k_start, int k_stop = 0);

  /**
   * @brief Set the dual variable for a general constraint
   *
   * @pre Solver must be initialized.
   * @param z Dual variable for the constraint
   * @param constraint_index Constraint index returned from `SetConstraint`
   */
  ErrorCodes SetDualGeneric(const a_float* z, const ConstraintIndex& constraint_index);

  ErrorCodes SetDualStateUpperBound(const a_float* b_x_upper, int k_start, int k_stop = 0);
  ErrorCodes SetDualStateLowerBound(const a_float* b_x_lower, int k_start, int k_stop = 0);
  ErrorCodes SetDualInputUpperBound(const a_float* b_u_upper, int k_start, int k_stop = 0);
  ErrorCodes SetDualInputLowerBound(const a_float* b_u_lower, int k_start, int k_stop = 0);

  ErrorCodes OpenLoopRollout();

  /**********************************************
   * MPC Methods
   **********************************************/
  ErrorCodes UpdateLinearCosts(const a_float* q, const a_float* r, a_float c,
                               int k_start = AllIndices, int k_stop = 0);

  ErrorCodes ShiftTrajectory();

  /**********************************************
   * Options
   **********************************************/
  void SetOptions(const AltroOptions& opts);
  AltroOptions& GetOptions();
  const AltroOptions& GetOptions() const;

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
  ErrorCodes SetCallback(CallbackFunction callback);

  /**********************************************
   * Solve
   **********************************************/
  SolveStatus Solve();

  SolveStatus GetStatus() const;
  int GetIterations() const;
  a_float GetSolveTimeMs() const;
  a_float GetPrimalFeasibility() const;
  a_float GetFinalObjective() const;
  a_float CalcCost();

  /**********************************************
   * Getters
   **********************************************/
  int GetHorizonLength() const;
  int GetStateDim(int k) const;
  int GetInputDim(int k) const;
  float GetFinalTime() const;
  float GetTimeStep(int k) const;
  ErrorCodes GetState(a_float* x, int k) const;
  ErrorCodes GetInput(a_float* u, int k) const;
  ErrorCodes GetDualDynamics(a_float* y, int k) const;
  ErrorCodes GetDualGeneral(a_float* z, int ConstraintIndex) const;
  ErrorCodes GetDualStateUpperBound(a_float* b_x_upper, int ConstraintIndex) const;
  ErrorCodes GetDualStateLowerBound(a_float* b_x_lower, int ConstraintIndex) const;
  ErrorCodes GetDualInputUpperBound(a_float* b_u_upper, int ConstraintIndex) const;
  ErrorCodes GetDualInputLowerBound(a_float* b_u_lower, int ConstraintIndex) const;
  ErrorCodes GetFeedbackGain(a_float* K, int k) const;
  ErrorCodes GetFeedforwardGain(a_float* d, int k) const;

  /**********************************************
   * Printers
   **********************************************/
  void PrintStateTrajectory() const;
  void PrintInputTrajectory() const;

  std::unique_ptr<SolverImpl> solver_;

 private:
  enum class LastIndexMode { Inclusive, Exclusive };
  ErrorCodes CheckKnotPointIndices(int& k_start, int& k_stop, LastIndexMode last_index) const;
  ErrorCodes AssertInitialized() const;
  ErrorCodes AssertDimensionsAreSet(int k_start, int k_stop, std::string msg = "") const;
  ErrorCodes AssertStateDim(int k, int n) const;
  ErrorCodes AssertInputDim(int k, int m) const;
  ErrorCodes AssertTimestepsArePositive(std::string msg = "") const;

  //  SolverImpl *solver_;
};

}  // namespace altro
