#include "GCMMASolver.h"
#include "../mma/MMASolver.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mma {
namespace {

bool Finite(const double* values, const int count) {
    if (count == 0) return true;
    if (values == nullptr) return false;
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

} // namespace

GCMMASolver::GCMMASolver(const int nn, const int mm, const double ai,
                         const double ci, const double di)
    : n(nn), m(mm), outeriter(0), raa0eps(1e-6), raaeps(raa0eps),
      xmamieps(1e-5), epsimin(std::sqrt(nn + mm) * 1e-9), move(0.5),
      albefa(0.1), asyminit(0.5), asymdec(0.7), asyminc(1.2), raa0(1.0),
      raa(mm, 1.0), a(mm, ai), c(mm, ci), d(mm, di), low(nn), upp(nn),
      alpha(nn), beta(nn), p0(nn), q0(nn), pij(nn * mm), qij(nn * mm),
      b(mm), r(mm), fapp(mm), xold1(nn), xold2(nn) {
    if (nn <= 0 || mm < 0) {
        throw std::invalid_argument("GCMMA dimensions must satisfy n > 0, m >= 0");
    }
}

void GCMMASolver::SetAsymptotes(const double init, const double decrease,
                                const double increase) {
    asyminit = init;
    asymdec = decrease;
    asyminc = increase;
}

void GCMMASolver::Reset() {
    outeriter = 0;
    raa0 = 1.0;
    std::fill(raa.begin(), raa.end(), 1.0);
    std::fill(xold1.begin(), xold1.end(), 0.0);
    std::fill(xold2.begin(), xold2.end(), 0.0);
    m_transaction_pending = false;
    m_diagnostics = GCMMASolveDiagnostics{};
}

bool GCMMASolver::InputsFinite(const double* xval, const double f0x,
                               const double* df0dx, const double* fx,
                               const double* dfdx, const double* xmin,
                               const double* xmax) const {
    return Finite(xval, n) && std::isfinite(f0x) && Finite(df0dx, n) &&
           Finite(fx, m) && Finite(dfdx, n * m) && Finite(xmin, n) &&
           Finite(xmax, n);
}

bool GCMMASolver::CandidateFinite(const double f0xnew,
                                  const double* fxnew) const {
    return std::isfinite(f0xnew) && Finite(fxnew, m);
}

bool GCMMASolver::StateFinite() const {
    return std::isfinite(raa0) && std::isfinite(f0app) && Finite(raa.data(), m) &&
           Finite(low.data(), n) && Finite(upp.data(), n) &&
           Finite(alpha.data(), n) && Finite(beta.data(), n) &&
           Finite(p0.data(), n) && Finite(q0.data(), n) &&
           Finite(pij.data(), n * m) && Finite(qij.data(), n * m) &&
           Finite(b.data(), m) && Finite(r.data(), m) && Finite(fapp.data(), m);
}

void GCMMASolver::SnapshotState(const double* xval) {
    m_snapshot.outeriter = outeriter;
    m_snapshot.raa0 = raa0;
    m_snapshot.raa = raa;
    m_snapshot.low = low;
    m_snapshot.upp = upp;
    m_snapshot.alpha = alpha;
    m_snapshot.beta = beta;
    m_snapshot.p0 = p0;
    m_snapshot.q0 = q0;
    m_snapshot.pij = pij;
    m_snapshot.qij = qij;
    m_snapshot.b = b;
    m_snapshot.r0 = r0;
    m_snapshot.f0app = f0app;
    m_snapshot.r = r;
    m_snapshot.fapp = fapp;
    m_snapshot.xold1 = xold1;
    m_snapshot.xold2 = xold2;
    m_snapshot.input_x.assign(xval, xval + n);
    m_snapshot.diagnostics = m_diagnostics;
}

void GCMMASolver::RestoreState(double* xmma) {
    outeriter = m_snapshot.outeriter;
    raa0 = m_snapshot.raa0;
    raa = m_snapshot.raa;
    low = m_snapshot.low;
    upp = m_snapshot.upp;
    alpha = m_snapshot.alpha;
    beta = m_snapshot.beta;
    p0 = m_snapshot.p0;
    q0 = m_snapshot.q0;
    pij = m_snapshot.pij;
    qij = m_snapshot.qij;
    b = m_snapshot.b;
    r0 = m_snapshot.r0;
    f0app = m_snapshot.f0app;
    r = m_snapshot.r;
    fapp = m_snapshot.fapp;
    xold1 = m_snapshot.xold1;
    xold2 = m_snapshot.xold2;
    if (xmma != nullptr && m_snapshot.input_x.size() == static_cast<std::size_t>(n)) {
        std::copy(m_snapshot.input_x.begin(), m_snapshot.input_x.end(), xmma);
    }
}

void GCMMASolver::SetFailure(const GCMMAStatus status, double* xmma) {
    if (m_transaction_pending) RestoreState(xmma);
    m_transaction_pending = false;
    m_diagnostics.status = status;
}

bool GCMMASolver::SolveCurrentSubproblem(double* xmma) {
    SubsolvProblem problem;
    problem.n = n;
    problem.m = m;
    problem.epsimin = epsimin;
    problem.low = low;
    problem.upp = upp;
    problem.alfa = alpha;
    problem.beta = beta;
    problem.p0 = p0;
    problem.q0 = q0;
    problem.P.assign(static_cast<std::size_t>(m) * n, 0.0);
    problem.Q.assign(static_cast<std::size_t>(m) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            problem.P[static_cast<std::size_t>(j) * n + i] = pij[i * m + j];
            problem.Q[static_cast<std::size_t>(j) * n + i] = qij[i * m + j];
        }
    }
    problem.a0 = 1.0;
    problem.a = a;
    problem.b = b;
    problem.c = c;
    problem.d = d;

    m_diagnostics.subsolv = SolveSubsolvFull(problem);
    const SubsolvResult& result = m_diagnostics.subsolv;
    m_diagnostics.subsolv_history.push_back(result);
    const bool inner_converged =
        result.status == SubsolvSolveStatus::ConvergedWithinSoftCap ||
        result.status == SubsolvSolveStatus::ConvergedAfterSoftCapExtension;
    if (!inner_converged || !result.all_finite || !result.domain_ok ||
        result.unsupported_branch ||
        !std::isfinite(result.residual.max_norm) ||
        result.residual.max_norm >
            MMASolver::KktResidualToleranceFactor() * epsimin) {
        return false;
    }
    if (result.x.size() != static_cast<std::size_t>(n)) return false;
    std::copy(result.x.begin(), result.x.end(), xmma);
    return StateFinite();
}

GCMMAStatus GCMMASolver::OuterUpdate(
    double* xmma, const double* xval, const double f0x, const double* df0dx,
    const double* fx, const double* dfdx, const double* xmin,
    const double* xmax) {
    if (xmma == nullptr || !InputsFinite(xval, f0x, df0dx, fx, dfdx, xmin, xmax)) {
        if (m_transaction_pending) RestoreState(xmma);
        m_transaction_pending = false;
        m_diagnostics.status = GCMMAStatus::NonFiniteEvaluation;
        return m_diagnostics.status;
    }
    SnapshotState(xval);
    m_transaction_pending = true;
    m_diagnostics = GCMMASolveDiagnostics{};
    m_diagnostics.fapp.assign(static_cast<std::size_t>(m), 0.0);
    Asymp(xval, df0dx, dfdx, xmin, xmax);
    GenSub(xval, f0x, df0dx, fx, dfdx, xmin, xmax);
    xold2 = xold1;
    std::copy_n(xval, n, xold1.data());
    if (!SolveCurrentSubproblem(xmma)) {
        SetFailure(GCMMAStatus::SubproblemFailure, xmma);
        return m_diagnostics.status;
    }
    ComputeApprox(xmma);
    if (!StateFinite() || !std::isfinite(m_diagnostics.subsolv.residual.max_norm)) {
        SetFailure(GCMMAStatus::SubproblemFailure, xmma);
        return m_diagnostics.status;
    }
    m_diagnostics.status = GCMMAStatus::CandidatePending;
    m_diagnostics.raa0_history.push_back(raa0);
    m_diagnostics.raa_history.push_back(raa);
    return m_diagnostics.status;
}

GCMMAStatus GCMMASolver::InnerUpdate(
    double* xmma, const double f0xnew, const double* fxnew,
    const double* xval, const double f0x, const double* df0dx,
    const double* fx, const double* dfdx, const double* xmin,
    const double* xmax) {
    if (!m_transaction_pending || xmma == nullptr || xval == nullptr ||
        !CandidateFinite(f0xnew, fxnew) ||
        !InputsFinite(xval, f0x, df0dx, fx, dfdx, xmin, xmax)) {
        SetFailure(GCMMAStatus::NonFiniteEvaluation, xmma);
        return m_diagnostics.status;
    }
    RaaUpdate(xmma, xval, f0xnew, fxnew, xmin, xmax);
    GenSub(xval, f0x, df0dx, fx, dfdx, xmin, xmax);
    std::copy_n(xval, n, xmma);
    if (!SolveCurrentSubproblem(xmma)) {
        SetFailure(GCMMAStatus::SubproblemFailure, xmma);
        return m_diagnostics.status;
    }
    ComputeApprox(xmma);
    if (!StateFinite()) {
        SetFailure(GCMMAStatus::SubproblemFailure, xmma);
        return m_diagnostics.status;
    }
    ++m_diagnostics.inner_iterations;
    m_diagnostics.raa0_history.push_back(raa0);
    m_diagnostics.raa_history.push_back(raa);
    m_diagnostics.status = GCMMAStatus::CandidatePending;
    return m_diagnostics.status;
}

bool GCMMASolver::ConCheck(const double f0xnew, const double* fxnew) {
    return ConCheck(f0xnew, fxnew, nullptr);
}

bool GCMMASolver::ConCheck(const double f0xnew, const double* fxnew,
                           const double* xnew) {
    if (!m_transaction_pending || !CandidateFinite(f0xnew, fxnew) ||
        (xnew != nullptr && !Finite(xnew, n)) ||
        !StateFinite() || !std::isfinite(f0app) || !Finite(fapp.data(), m)) {
        m_diagnostics.status = GCMMAStatus::NonFiniteEvaluation;
        return false;
    }
    bool conservative = f0app + epsimin >= f0xnew;
    for (int j = 0; j < m; ++j) {
        if (fapp[j] + epsimin < fxnew[j]) conservative = false;
    }
    m_diagnostics.conservativity_history.push_back(conservative);
    m_diagnostics.candidate_evaluations += 1;
    m_diagnostics.status = GCMMAStatus::CandidatePending;
    return conservative;
}

void GCMMASolver::CommitUpdate() {
    if (m_transaction_pending) {
        m_transaction_pending = false;
        m_diagnostics.status = GCMMAStatus::Success;
    }
}

void GCMMASolver::RejectUpdate(double* xmma, const GCMMAStatus reason) {
    if (m_transaction_pending) RestoreState(xmma);
    m_transaction_pending = false;
    m_diagnostics.status = reason;
}

void GCMMASolver::Asymp(const double* xval, const double* df0dx,
                        const double* dfdx, const double* xmin,
                        const double* xmax) {
    ++outeriter;
    std::vector<double> xmami(n);
    double objective_scale = 0.0;
    std::fill(raa.begin(), raa.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        xmami[i] = std::max(xmax[i] - xmin[i], xmamieps);
        objective_scale += std::abs(df0dx[i]) * xmami[i];
        for (int j = 0; j < m; ++j) {
            raa[j] += std::abs(dfdx[i * m + j]) * xmami[i];
        }
    }
    raa0 = std::max(raa0eps, (0.1 / n) * objective_scale);
    for (int j = 0; j < m; ++j) {
        raa[j] = std::max(raaeps, (0.1 / n) * raa[j]);
    }

    if (outeriter < 3) {
        for (int i = 0; i < n; ++i) {
            low[i] = xval[i] - 0.5 * xmami[i];
            upp[i] = xval[i] + 0.5 * xmami[i];
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        const double product = (xval[i] - xold1[i]) * (xold1[i] - xold2[i]);
        const double factor = product > 0.0 ? asyminc : (product < 0.0 ? asymdec : 1.0);
        low[i] = xval[i] - factor * (xold1[i] - low[i]);
        upp[i] = xval[i] + factor * (upp[i] - xold1[i]);
        // Official asymp.m v1.5 bounds. The former 100/1e-5 variant was an
        // unqualified project deviation and is intentionally not active here.
        low[i] = std::max(low[i], xval[i] - 10.0 * xmami[i]);
        low[i] = std::min(low[i], xval[i] - 0.01 * xmami[i]);
        upp[i] = std::min(upp[i], xval[i] + 10.0 * xmami[i]);
        upp[i] = std::max(upp[i], xval[i] + 0.01 * xmami[i]);
    }
}

void GCMMASolver::RaaUpdate(const double* xmma, const double* xval,
                            const double f0xnew, const double* fxnew,
                            const double* xmin, const double* xmax) {
    const double raacofmin = 1e-12;
    double raacof = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xmami = std::max(xmax[i] - xmin[i], xmamieps);
        const double xxux = (xmma[i] - xval[i]) / (upp[i] - xmma[i]);
        const double xxxl = (xmma[i] - xval[i]) / (xmma[i] - low[i]);
        raacof += xxux * xxxl * (upp[i] - low[i]) / xmami;
    }
    raacof = std::max(raacof, raacofmin);
    if (f0xnew > f0app + 0.5 * epsimin) {
        const double deltaraa0 = (f0xnew - f0app) / raacof;
        raa0 = std::min(1.1 * (raa0 + deltaraa0), 10.0 * raa0);
    }
    for (int j = 0; j < m; ++j) {
        if (fxnew[j] > fapp[j] + 0.5 * epsimin) {
            const double deltaraa = (fxnew[j] - fapp[j]) / raacof;
            raa[j] = std::min(1.1 * (raa[j] + deltaraa), 10.0 * raa[j]);
        }
    }
}

void GCMMASolver::GenSub(const double* xval, const double f0x,
                         const double* df0dx, const double* fx,
                         const double* dfdx, const double* xmin,
                         const double* xmax) {
    r0 = 0.0;
    std::fill(r.begin(), r.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        alpha[i] = std::max(xmin[i], low[i] + albefa * (xval[i] - low[i]));
        alpha[i] = std::max(alpha[i], xval[i] - move * (xmax[i] - xmin[i]));
        alpha[i] = std::min(alpha[i], xmax[i]);
        beta[i] = std::min(xmax[i], upp[i] - albefa * (upp[i] - xval[i]));
        beta[i] = std::min(beta[i], xval[i] + move * (xmax[i] - xmin[i]));
        beta[i] = std::max(beta[i], xmin[i]);
        const double uxinv = 1.0 / (upp[i] - xval[i]);
        const double xlinv = 1.0 / (xval[i] - low[i]);
        const double xmamiinv = 1.0 / std::max(xmax[i] - xmin[i], xmamieps);
        const double dfp = std::max(0.0, df0dx[i]);
        const double dfm = std::max(0.0, -df0dx[i]);
        const double pq0 = 0.001 * std::abs(df0dx[i]) + raa0 * xmamiinv;
        p0[i] = (dfp + pq0) * (upp[i] - xval[i]) * (upp[i] - xval[i]);
        q0[i] = (dfm + pq0) * (xval[i] - low[i]) * (xval[i] - low[i]);
        r0 += p0[i] * uxinv + q0[i] * xlinv;
        for (int j = 0; j < m; ++j) {
            const double dgp = std::max(0.0, dfdx[i * m + j]);
            const double dgm = std::max(0.0, -dfdx[i * m + j]);
            const double pq = 0.001 * std::abs(dfdx[i * m + j]) + raa[j] * xmamiinv;
            pij[i * m + j] = (dgp + pq) * (upp[i] - xval[i]) * (upp[i] - xval[i]);
            qij[i * m + j] = (dgm + pq) * (xval[i] - low[i]) * (xval[i] - low[i]);
            r[j] += pij[i * m + j] * uxinv + qij[i * m + j] * xlinv;
        }
    }
    r0 = f0x - r0;
    for (int j = 0; j < m; ++j) {
        r[j] = fx[j] - r[j];
        b[j] = -r[j];
    }
}

void GCMMASolver::ComputeApprox(const double* xmma) {
    f0app = 0.0;
    std::fill(fapp.begin(), fapp.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        const double uxinv = 1.0 / (upp[i] - xmma[i]);
        const double xlinv = 1.0 / (xmma[i] - low[i]);
        f0app += p0[i] * uxinv + q0[i] * xlinv;
        for (int j = 0; j < m; ++j) {
            fapp[j] += pij[i * m + j] * uxinv + qij[i * m + j] * xlinv;
        }
    }
    f0app += r0;
    for (int j = 0; j < m; ++j) fapp[j] += r[j];
    m_diagnostics.f0app = f0app;
    m_diagnostics.fapp = fapp;
}

} // namespace mma
