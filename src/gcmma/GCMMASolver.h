////////////////////////////////////////////////////////////////////////////////
// GCMMA solver using the reference-faithful full primal-dual subsolver.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../mma/SubsolvFull.h"

#include <vector>

namespace mma {

/** Outcome of the complete GCMMA outer-update transaction. */
enum class GCMMAStatus {
    NotRun,
    CandidatePending,
    Success,
    SubproblemFailure,
    NonFiniteEvaluation,
    NonConservativeAtInnerLimit,
};

/** Diagnostics for the most recently generated GCMMA subproblem. */
struct GCMMASolveDiagnostics {
    GCMMAStatus status = GCMMAStatus::NotRun;
    SubsolvResult subsolv;
    double f0app = 0.0;
    std::vector<double> fapp;
    int inner_iterations = 0;
    int candidate_evaluations = 0;
    std::vector<bool> conservativity_history;
    std::vector<double> raa0_history;
    std::vector<std::vector<double>> raa_history;
    std::vector<SubsolvResult> subsolv_history;
};

class GCMMASolver {
  public:
    GCMMASolver(int n, int m, double a = 0.0, double c = 1000.0,
                double d = 0.0);

    void SetAsymptotes(double init, double decrease, double increase);

    /** Generate and solve one GCMMA subproblem. */
    GCMMAStatus OuterUpdate(double* xmma, const double* xval, double f0x,
                            const double* df0dx, const double* fx,
                            const double* dfdx, const double* xmin,
                            const double* xmax);

    /** Update curvature, generate, and solve the next inner subproblem. */
    GCMMAStatus InnerUpdate(double* xmma, double f0xnew,
                            const double* fxnew, const double* xval,
                            double f0x, const double* df0dx,
                            const double* fx, const double* dfdx,
                            const double* xmin, const double* xmax);

    /** Check the current candidate against the generated approximation. */
    bool ConCheck(double f0xnew, const double* fxnew);
    bool ConCheck(double f0xnew, const double* fxnew, const double* xnew);

    /** Commit a conservative candidate and close the pending transaction. */
    void CommitUpdate();

    /** Reject a pending candidate and restore every persistent field. */
    void RejectUpdate(double* xmma,
                     GCMMAStatus reason = GCMMAStatus::SubproblemFailure);

    void Reset();

    GCMMAStatus GetStatus() const noexcept { return m_diagnostics.status; }
    const GCMMASolveDiagnostics& GetDiagnostics() const noexcept {
        return m_diagnostics;
    }
    const SubsolvResult& GetSubsolvResult() const noexcept {
        return m_diagnostics.subsolv;
    }
    double ApproximateObjective() const noexcept { return f0app; }
    const std::vector<double>& ApproximateConstraints() const noexcept {
        return fapp;
    }

    // Read-only trajectory accessors for the qualification harness.
    int OuterIteration() const noexcept { return outeriter; }
    double Raa0() const noexcept { return raa0; }
    const std::vector<double>& Raa() const noexcept { return raa; }
    const std::vector<double>& Low() const noexcept { return low; }
    const std::vector<double>& Upp() const noexcept { return upp; }
    const std::vector<double>& Alpha() const noexcept { return alpha; }
    const std::vector<double>& Beta() const noexcept { return beta; }
    const std::vector<double>& P0() const noexcept { return p0; }
    const std::vector<double>& Q0() const noexcept { return q0; }
    const std::vector<double>& P() const noexcept { return pij; }
    const std::vector<double>& Q() const noexcept { return qij; }
    const std::vector<double>& B() const noexcept { return b; }

  private:
    int n, m, outeriter;
    const double raa0eps;
    const double raaeps;
    const double xmamieps;
    const double epsimin;
    const double move, albefa;
    double asyminit, asymdec, asyminc;

    double raa0;
    std::vector<double> raa;
    std::vector<double> a, c, d;
    std::vector<double> low, upp, alpha, beta, p0, q0, pij, qij, b;
    double r0 = 0.0, f0app = 0.0;
    std::vector<double> r, fapp;
    std::vector<double> xold1, xold2;

    GCMMASolveDiagnostics m_diagnostics;
    bool m_transaction_pending = false;

    struct Snapshot {
        int outeriter = 0;
        double raa0 = 0.0;
        std::vector<double> raa, low, upp, alpha, beta, p0, q0, pij, qij, b;
        double r0 = 0.0, f0app = 0.0;
        std::vector<double> r, fapp, xold1, xold2;
        std::vector<double> input_x;
        GCMMASolveDiagnostics diagnostics;
    } m_snapshot;

    void SnapshotState(const double* xval);
    void RestoreState(double* xmma);
    bool InputsFinite(const double* xval, double f0x, const double* df0dx,
                      const double* fx, const double* dfdx,
                      const double* xmin, const double* xmax) const;
    bool CandidateFinite(double f0xnew, const double* fxnew) const;
    bool StateFinite() const;
    void SetFailure(GCMMAStatus status, double* xmma);
    bool SolveCurrentSubproblem(double* xmma);

    void Asymp(const double* xval, const double* df0dx, const double* dfdx,
               const double* xmin, const double* xmax);
    void RaaUpdate(const double* xmma, const double* xval, double f0xnew,
                   const double* fxnew, const double* xmin,
                   const double* xmax);
    void GenSub(const double* xval, double f0x, const double* df0dx,
                const double* fx, const double* dfdx, const double* xmin,
                const double* xmax);
    void ComputeApprox(const double* xmma);
};

} // namespace mma
