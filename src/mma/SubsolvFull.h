////////////////////////////////////////////////////////////////////////////////
// Full primal-dual MMA subproblem solver.
//
// A formula-preserving C++ port of the archived official Svanberg v1.5
// `subsolv.m` (GCMMA-MMA-code, Dec 2006). It replaces the reduced dual
// formulation that the vendored Dumas/Aage `MMASolver::SolveDIP` implements.
//
// WHY THIS EXISTS
// ---------------
// The reduced solver keeps only (lam, mu) as state and slaves x, y, z to lam.
// That system's Newton step is not the Newton step of the barrier system it is
// measuring, so the reference's residual-decrease backtracking cannot be
// transplanted onto it: measured on the m7 matrix, doing so refused the full
// feasible step on 80.5% of all inner iterations and created 3007 new
// cap-saturating solves (see /mnt/Local/Results/m8_mma_dual_repair). The
// reference globalization is correct precisely on the full system, so the full
// system is what this file implements.
//
// The variable names, dimensions and equations below follow `subsolv.m`
// verbatim; every block cites the reference lines it ports.
//
// THE SUBPROBLEM (subsolv.m lines 31-37)
// --------------------------------------
//   minimize   SUM_j [ p0_j/(upp_j - x_j) + q0_j/(x_j - low_j) ]
//              + a0*z + SUM_i [ c_i*y_i + 0.5*d_i*y_i^2 ]
//   subject to SUM_j [ p_ij/(upp_j - x_j) + q_ij/(x_j - low_j) ] - a_i*z - y_i
//              <= b_i
//              alfa_j <= x_j <= beta_j,  y_i >= 0,  z >= 0.
//
// DIMENSIONS
// ----------
//   n   design variables
//   m   constraints
//
// P and Q use the REFERENCE layout, which is the transpose of the production
// `pij`/`qij` arrays: P[i*n + j] is MATLAB's P(i+1, j+1) -- row i is the
// constraint, column j is the design variable. The caller converts once.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

namespace mma {

/**
 * Full KKT / barrier residual of the subproblem, using the reference's own
 * nine equation groups (subsolv.m lines 72-85, evaluated at any point and
 * barrier parameter). Every field is the component-wise maximum of |.| over
 * that group's vector.
 */
struct SubsolvResidual {
    /// x stationarity: dpsidx - xsi + eta                          (subsolv 72)
    double rex = 0.0;
    /// y stationarity: c + d.*y - mu - lam                         (subsolv 73)
    double rey = 0.0;
    /// z stationarity: a0 - zet - a'*lam                           (subsolv 74)
    double rez = 0.0;
    /// primal constraints: gvec - a*z - y + s - b                  (subsolv 75)
    double relam = 0.0;
    /// lower-bound complementarity: xsi.*(x-alfa) - epsi           (subsolv 76)
    double rexsi = 0.0;
    /// upper-bound complementarity: eta.*(beta-x) - epsi           (subsolv 77)
    double reeta = 0.0;
    /// y/mu complementarity: mu.*y - epsi                          (subsolv 78)
    double remu = 0.0;
    /// z/zet complementarity: zet*z - epsi                         (subsolv 79)
    double rezet = 0.0;
    /// lambda/s complementarity: lam.*s - epsi                     (subsolv 80)
    double res = 0.0;

    /// max(abs(residu))  -- the inner loop's own stopping measure (subsolv 85)
    double max_norm = 0.0;
    /// sqrt(residu'*residu) -- the reference's backtracking measure (subsolv 84)
    double norm2 = 0.0;

    /// Largest of the nine component maxima.
    double worst_component() const;
};

/// The subproblem exactly as `subsolv.m` receives it.
struct SubsolvProblem {
    int n = 0;
    int m = 0;
    double epsimin = 1.0e-7;
    /// n
    std::vector<double> low, upp, alfa, beta;
    /// n
    std::vector<double> p0, q0;
    /// m*n, reference layout: P[i*n + j] == MATLAB P(i+1, j+1)
    std::vector<double> P, Q;
    double a0 = 1.0;
    /// m
    std::vector<double> a, b, c, d;
};

/// Everything `subsolv.m` returns, plus the diagnostics this project needs.
struct SubsolvResult {
    /// n
    std::vector<double> x;
    /// m
    std::vector<double> y;
    double z = 0.0;
    /// m
    std::vector<double> lam;
    /// n
    std::vector<double> xsi;
    /// n
    std::vector<double> eta;
    /// m
    std::vector<double> mu;
    double zet = 0.0;
    /// m
    std::vector<double> s;

    /// Full KKT residual at the RETURNED point, at `epsi_final`.
    SubsolvResidual residual;

    double epsi_final = 0.0;
    int barrier_levels = 0;
    int inner_newton_iterations = 0;
    int capped_barrier_levels = 0;

    int backtracking_iterations = 0;   ///< Newton iterations needing >=1 halving
    int backtracking_reductions = 0;   ///< halvings summed over the solve
    int backtracking_max_reductions = 0;
    int backtracking_exhausted = 0;    ///< line searches that gave up
    int full_step_iterations = 0;      ///< Newton iterations accepted at once

    /// Every returned state entry is finite and inside its own domain
    /// (alfa <= x <= beta, y >= 0, z >= 0, lam >= 0, mu >= 0, s >= 0,
    /// xsi > 0, eta > 0, zet >= 0).
    bool domain_ok = false;
    bool all_finite = false;

    /// Set when the `m < n` / `m >= n` branch could not be taken as the
    /// reference takes it (see the note on `kDenseBranchLimit` in the .cpp).
    bool unsupported_branch = false;
};

/**
 * Solve the subproblem with the full primal-dual interior-point method of
 * `subsolv.m`. Pure function of the problem: no global state, no I/O.
 */
SubsolvResult SolveSubsolvFull(const SubsolvProblem& problem);

/// The full KKT/barrier residual of a point, evaluated with the reference's
/// equations. Exposed so that a returned solution can be scored independently
/// of the solver that produced it.
SubsolvResidual EvaluateSubsolvResidual(const SubsolvProblem& problem,
                                        const SubsolvResult& point,
                                        double epsi);

/// Reference constants that are part of `subsolv.m`'s control logic.
struct SubsolvConstants {
    /// Inner Newton cap, `ittt < 200` (subsolv.m line 87).
    static int InnerIterationCap() noexcept { return 200; }
    /// Backtracking budget, `itto < 50` (subsolv.m line 175).
    static int BacktrackingReductionLimit() noexcept { return 50; }
    /// Barrier reduction, `epsi = 0.1*epsi` (subsolv.m line 219).
    static double BarrierReduction() noexcept { return 0.1; }
    /// Initial barrier, `epsi = 1` (subsolv.m line 44).
    static double InitialBarrier() noexcept { return 1.0; }
    /// Inner-loop acceptance factor, `residumax > 0.9*epsi` (subsolv.m line 87).
    static double InnerResidualFactor() noexcept { return 0.9; }
    /// Step-limiting safety factor, `-1.01*dxx./xx` (subsolv.m line 152).
    static double StepSafety() noexcept { return 1.01; }
};

} // namespace mma
