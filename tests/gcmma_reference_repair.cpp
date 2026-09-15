#include <gcmma/GCMMASolver.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Input {
    std::vector<double> x{0.5, 0.5};
    std::vector<double> xmin{0.0, 0.0};
    std::vector<double> xmax{1.0, 1.0};
    double f = 0.0;
    std::vector<double> df{1.0, -2.0};
    std::vector<double> g{-0.2, -0.3};
    // design-major, constraint-minor layout used by the public API
    std::vector<double> dg{1.0, 0.0, 0.0, 2.0};
};

mma::GCMMAStatus Outer(mma::GCMMASolver& solver, const Input& in,
                       std::vector<double>& candidate) {
    candidate.resize(in.x.size());
    return solver.OuterUpdate(candidate.data(), in.x.data(), in.f, in.df.data(),
                              in.g.data(), in.dg.data(), in.xmin.data(),
                              in.xmax.data());
}

void CheckFullResultAndReferenceScaling() {
    Input in;
    mma::GCMMASolver solver(2, 2);
    std::vector<double> candidate;
    Check(Outer(solver, in, candidate) == mma::GCMMAStatus::CandidatePending,
          "reference GCMMA outer solve did not produce a candidate");
    const auto& result = solver.GetSubsolvResult();
    Check(result.x.size() == 2 && result.y.size() == 2 && result.lam.size() == 2,
          "full result did not expose x/y/lambda");
    Check(result.xsi.size() == 2 && result.eta.size() == 2 &&
              result.mu.size() == 2 && result.s.size() == 2,
          "full result did not expose xsi/eta/mu/s");
    Check(std::isfinite(result.z) && std::isfinite(result.zet),
          "full result did not expose finite z/zet");
    Check(std::abs(solver.Raa0() - 0.15) < 1e-14,
          "raa0 is not initialized by the reference formula");
    Check(std::abs(solver.Raa()[0] - 0.05) < 1e-14 &&
              std::abs(solver.Raa()[1] - 0.1) < 1e-14,
          "constraint raa is not initialized independently by the reference formula");

    mma::GCMMASolver objective_scaled(2, 2);
    Input scaled = in;
    scaled.df[0] *= 7.0;
    scaled.df[1] *= 7.0;
    std::vector<double> ignored;
    Check(Outer(objective_scaled, scaled, ignored) ==
              mma::GCMMAStatus::CandidatePending,
          "objective scale fixture did not solve");
    Check(std::abs(objective_scaled.Raa0() / solver.Raa0() - 7.0) < 1e-12,
          "objective scaling did not scale raa0");

    mma::GCMMASolver constraint_scaled(2, 2);
    scaled = in;
    scaled.dg[0] *= 11.0;
    scaled.dg[1] *= 11.0;
    Check(Outer(constraint_scaled, scaled, ignored) ==
              mma::GCMMAStatus::CandidatePending,
          "constraint scale fixture did not solve");
    Check(std::abs(constraint_scaled.Raa()[0] / solver.Raa()[0] - 11.0) <
              1e-12,
          "constraint scaling did not scale only raa_0");
    Check(std::abs(constraint_scaled.Raa()[1] / solver.Raa()[1] - 1.0) <
              1e-12,
          "constraint scaling spuriously changed unrelated raa");
}

void CheckReferenceClamps() {
    Input in;
    mma::GCMMASolver solver(2, 2);
    std::vector<double> candidate;
    for (int iteration = 0; iteration < 3; ++iteration) {
        Check(Outer(solver, in, candidate) == mma::GCMMAStatus::CandidatePending,
              "clamp fixture outer solve failed");
        Check(solver.ConCheck(-1e300, nullptr) == false,
              "null constraints unexpectedly passed with m > 0");
        // Use finite, deliberately conservative values for the two constraints.
        const double gnew[2] = {-1e300, -1e300};
        Check(solver.ConCheck(-1e300, gnew),
              "conservative clamp fixture was rejected");
        solver.CommitUpdate();
        in.x = candidate;
    }
    for (std::size_t i = 0; i < in.x.size(); ++i) {
        Check(solver.Low()[i] >= in.x[i] - 10.0 - 1e-12 &&
                  solver.Low()[i] <= in.x[i] - 0.01 + 1e-12,
              "lower asymptote does not use reference clamp");
        Check(solver.Upp()[i] <= in.x[i] + 10.0 + 1e-12 &&
                  solver.Upp()[i] >= in.x[i] + 0.01 - 1e-12,
              "upper asymptote does not use reference clamp");
    }
}

void CheckNonFiniteAndCapStatuses() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Input in;
    for (double bad : {nan, inf, -inf}) {
        mma::GCMMASolver solver(2, 2);
        std::vector<double> candidate;
        Check(Outer(solver, in, candidate) == mma::GCMMAStatus::CandidatePending,
              "non-finite fixture outer solve failed");
        const std::vector<double> before = in.x;
        const double gnew[2] = {-0.2, -0.3};
        Check(!solver.ConCheck(bad, gnew), "non-finite objective was accepted");
        Check(solver.GetStatus() == mma::GCMMAStatus::NonFiniteEvaluation,
              "non-finite objective did not report status");
        solver.RejectUpdate(candidate.data(), mma::GCMMAStatus::NonFiniteEvaluation);
        Check(candidate == before, "non-finite rejection did not restore candidate");
        Check(solver.GetStatus() == mma::GCMMAStatus::NonFiniteEvaluation,
              "non-finite rejection status was lost");
    }
    for (int bad_index = 0; bad_index < 2; ++bad_index) {
        mma::GCMMASolver solver(2, 2);
        std::vector<double> candidate;
        Check(Outer(solver, in, candidate) == mma::GCMMAStatus::CandidatePending,
              "constraint non-finite fixture outer solve failed");
        double gnew[2] = {-0.2, -0.3};
        gnew[bad_index] = nan;
        Check(!solver.ConCheck(0.0, gnew),
              "non-finite constraint was accepted");
        Check(solver.GetStatus() == mma::GCMMAStatus::NonFiniteEvaluation,
              "non-finite constraint did not report status");
        solver.RejectUpdate(candidate.data(), mma::GCMMAStatus::NonFiniteEvaluation);
    }
    {
        mma::GCMMASolver solver(2, 2);
        std::vector<double> candidate;
        Check(Outer(solver, in, candidate) == mma::GCMMAStatus::CandidatePending,
              "candidate-x non-finite fixture outer solve failed");
        candidate[1] = nan;
        const double gnew[2] = {-0.2, -0.3};
        Check(!solver.ConCheck(0.0, gnew, candidate.data()),
              "non-finite candidate x was accepted");
        Check(solver.GetStatus() == mma::GCMMAStatus::NonFiniteEvaluation,
              "non-finite candidate x did not report status");
        solver.RejectUpdate(candidate.data(), mma::GCMMAStatus::NonFiniteEvaluation);
    }

    mma::GCMMASolver capped(2, 2);
    std::vector<double> candidate;
    Check(Outer(capped, in, candidate) == mma::GCMMAStatus::CandidatePending,
          "cap fixture outer solve failed");
    const std::vector<double> before = in.x;
    const double bad_objective = capped.ApproximateObjective() + 1.0;
    const double bad_constraints[2] = {-0.2, -0.3};
    Check(!capped.ConCheck(bad_objective, bad_constraints),
          "deliberately non-conservative fixture passed");
    capped.RejectUpdate(candidate.data(), mma::GCMMAStatus::NonConservativeAtInnerLimit);
    Check(candidate == before, "inner-cap rejection did not roll back x");
    Check(capped.GetStatus() == mma::GCMMAStatus::NonConservativeAtInnerLimit,
          "inner-cap rejection did not report explicit status");
}

void CheckRollbackReplay() {
    Input in;
    mma::GCMMASolver control(2, 2);
    mma::GCMMASolver replay(2, 2);
    std::vector<double> control_x, replay_x;
    Check(Outer(control, in, control_x) == mma::GCMMAStatus::CandidatePending,
          "control first solve failed");
    const double conservative_g[2] = {-1e300, -1e300};
    Check(control.ConCheck(-1e300, conservative_g), "control was not conservative");
    control.CommitUpdate();
    Input next = in;
    next.x = control_x;
    Check(Outer(control, next, control_x) == mma::GCMMAStatus::CandidatePending,
          "control replay solve failed");

    Check(Outer(replay, in, replay_x) == mma::GCMMAStatus::CandidatePending,
          "replay first solve failed");
    const double nonconservative_g[2] = {-0.2, -0.3};
    Check(!replay.ConCheck(replay.ApproximateObjective() + 1.0,
                           nonconservative_g), "replay failure did not occur");
    replay.RejectUpdate(replay_x.data(),
                        mma::GCMMAStatus::NonConservativeAtInnerLimit);
    Check(Outer(replay, next, replay_x) == mma::GCMMAStatus::CandidatePending,
          "replay post-rollback solve failed");
    Check(control_x == replay_x, "rollback replay was not byte-identical");
    Check(control.Low() == replay.Low() && control.Upp() == replay.Upp() &&
              control.Raa() == replay.Raa(),
          "rollback replay did not restore asymptote/raa history exactly");
}

} // namespace

int main() {
    try {
        CheckFullResultAndReferenceScaling();
        CheckReferenceClamps();
        CheckNonFiniteAndCapStatuses();
        CheckRollbackReplay();
    } catch (const std::exception& error) {
        return (std::fprintf(stderr, "%s\n", error.what()), 1);
    }
    return 0;
}
