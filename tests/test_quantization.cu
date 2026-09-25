#include <quantization_sparse/problem.cuh>
#include <iostream>
#include <iomanip>

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, msg) \
    do { \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++g_pass; } \
        else      { std::cerr << "  FAIL: " << (msg) << "\n"; ++g_fail; } \
    } while (0)

#define EXPECT_NEAR(a, b, tol, msg) \
    EXPECT(std::abs((double)(a)-(double)(b)) < (tol), msg)

// Opt-in, full-size diagnostic; all root runs restart from the same initial
// vertex. No children are searched, so solver completion is NOT an optimality
// claim. The reported certificate is last_gpu_gap(), not the global B&B gap.
__global__ void quantization_component_probe(
    quantization_sparse::QuantizationObjective objective,
    quantization_sparse::SparseInt4LMO lmo, int component,
    const double* x, double* g, const double* lb, const double* ub,
    double* out, unsigned char* workspace)
{
    if (component == 0) objective.grad(threadIdx.x, x, g, workspace);
    else if (component == 1)
        lmo.compute_v(threadIdx.x, quantization_sparse::Vars, g, lb, ub, out);
    else
        lmo.compute_inface_v(threadIdx.x, quantization_sparse::Vars,
                            g, x, lb, ub, out);
}

static void test_quantization_root_diagnostic(int max_iters, bool exact_only = false)
{
    using namespace quantization_sparse;
    std::cout << "\n[Quantization root diagnostic: " << Weights << " weights, 2000 samples, k=" << Weights/2 << ", seed=42]\n"
              << "Each row restarts the root; eps=1e-3 is an ABSOLUTE FW gap.\n"
              << "Maximum iteration budget: " << max_iters << ". Stops early if the FW gap meets eps.\n"
              << "No branching; completion does not certify integer optimality.\n" << std::flush;
    Instance data(42, 2000);
    DeviceBuffer coefficients(data.coefficients);
    QuantizationObjective objective{coefficients.p, data.coefficients.data()};
    SparseInt4LMO lmo{Weights/2};
    Problem problem;
    problem.n_vars = Vars;
    problem.lb.assign(Vars, 0); problem.ub.assign(Vars, 1);
    for (int i = 0; i < Weights; ++i) {
        problem.lb[i] = -8; problem.ub[i] = 7;
    }
    problem.int_vars.resize(Vars);
    std::iota(problem.int_vars.begin(), problem.int_vars.end(), 0);
    problem.f = [objective](const std::vector<double>& x) { return objective.f(x); };
    problem.grad_f = [objective](const std::vector<double>& x, std::vector<double>& g) {
        objective.gradient_host(x, g);
    };

    // Use a nontrivial feasible interior point for independent GPU/CPU checks.
    std::vector<double> x(Vars, 0.25), gradient, expected;
    for (int i = 0; i < Weights; ++i) x[i] = 0.25*data.w[i];
    objective.gradient_host(x, gradient);
    DeviceBuffer dx(x), dg(gradient), dlb(problem.lb), dub(problem.ub), output{size_t(Vars)};
    DeviceBuffer workspace{size_t(2*Weights+4)};
    const char* labels[] = {"gradient dH", "forward LMO", "in-face LMO"};
    for (int component = 0; component < 3; ++component) {
        if (component == 2) {
            // The away oracle minimizes the negative gradient.
            for (double& value : gradient) value = -value;
            DRS_CUDA_CHECK(cudaMemcpy(dg.p, gradient.data(), Vars*sizeof(double),
                                      cudaMemcpyHostToDevice));
        }
        auto launch = [&] {
            quantization_component_probe<<<1, QuantizationObjective::block_threads>>>(
                objective, lmo, component, dx.p, dg.p, dlb.p, dub.p, output.p,
                reinterpret_cast<unsigned char*>(workspace.p));
            DRS_CUDA_CHECK(cudaGetLastError());
        };
        launch(); // Warm up before timing.
        DRS_CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        DRS_CUDA_CHECK(cudaEventCreate(&start));
        DRS_CUDA_CHECK(cudaEventCreate(&stop));
        DRS_CUDA_CHECK(cudaEventRecord(start));
        for (int repeat = 0; repeat < 10; ++repeat) launch();
        DRS_CUDA_CHECK(cudaEventRecord(stop));
        DRS_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        DRS_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        DRS_CUDA_CHECK(cudaEventDestroy(start));
        DRS_CUDA_CHECK(cudaEventDestroy(stop));
        std::cout << "  isolated " << labels[component] << ": " << ms/10 << " ms/call\n";
        std::vector<double> actual(Vars);
        DRS_CUDA_CHECK(cudaMemcpy(actual.data(), component == 0 ? dg.p : output.p,
                                  Vars*sizeof(double), cudaMemcpyDeviceToHost));
        if (component == 0) expected = gradient;
        else lmo.oracle_cpu(gradient, problem.lb, problem.ub, expected,
                            component == 2 ? x.data() : nullptr);
        double error = 0, scale = 1;
        for (int i = 0; i < Vars; ++i) {
            if (!std::isfinite(actual[i])) error = std::numeric_limits<double>::infinity();
            error = std::max(error, std::abs(actual[i]-expected[i]));
            scale = std::max(scale, std::abs(expected[i]));
        }
        EXPECT(error <= 1e-8*scale, std::string(labels[component])+" agrees with CPU");
        std::cout << std::flush;
    }
    // Check the new GPU Gram construction against the original residual loss.
    const double residual = data.residual(x), gram_loss = objective.f(x);
    EXPECT(std::abs(residual-gram_loss) <= 1e-8*(1+residual),
           "GPU Gram objective agrees with direct X residual");

    BranchStrategy no_branch = [](const Node&, int&, const std::vector<double>&,
                                  const std::vector<int>&) { return std::vector<Node>{}; };
    using Mode = DeviceResidentConfig::DICGOracleParallelism;
    const LineSearchOptions defaults;
    std::vector<int> budgets{1, 10, 100, 1000, 10000, max_iters};
    std::sort(budgets.begin(), budgets.end());
    budgets.erase(std::unique(budgets.begin(), budgets.end()), budgets.end());
    // Default diagnostics include all three searches on identical fresh roots.
    // The exact-only CLI keeps long convergence experiments affordable.
    for (int search : {0, 1, 2}) {
        const bool exact = search == 2;
        const bool tight = search == 1;
        if (exact_only && !exact) continue;
        for (Mode mode : {Mode::Serial, Mode::Auto}) {
            for (int budget : budgets) {
                if (budget > max_iters) continue;
                DeviceResidentConfig cfg;
                cfg.gpu_batch_size = 1; cfg.num_streams = 1;
                cfg.gpu_max_iters = budget;
                cfg.gpu_eps = 1e-3;
                cfg.dicg_oracle_parallelism = mode;
                const SecantLS ls = tight ? SecantLS{16, 1e-10, 1e-14}
                    : SecantLS{defaults.secant_iters, defaults.secant_tol, defaults.secant_denom_eps};
                std::cout << "  RUN line_search=" << (exact ? "quadratic-exact" : tight ? "tight" : "default")
                          << " LMO=" << (mode == Mode::Serial ? "serial" : "auto")
                          << " budget=" << budget << std::flush;
                auto run = [&](auto algorithm) {
                    DeviceResidentHybridSolver<QuantizationObjective, decltype(algorithm), SparseInt4LMO>
                        solver(problem, objective, algorithm, no_branch, cfg, lmo);
                    Node root(0, -1, 0, problem.lb, problem.ub);
                    root.dual_bound = 0;
                    solver.push(root);
                    const auto start = std::chrono::steady_clock::now();
                    solver.solve();
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now()-start).count();
                    const int iterations = solver.last_gpu_iterations();
                    const double gap = solver.last_gpu_gap();
                    std::cout << " actual_LMO=" << (solver.paired_gpu_batches() ? "paired" : "serial")
                              << " iters=" << iterations << " gap=" << std::scientific << gap
                              << " converged=" << solver.last_gpu_converged()
                              << " wall_s=" << seconds
                              << " wall_ms/iter=" << 1000*seconds/std::max(1, iterations)
                              << std::defaultfloat << '\n' << std::flush;
                    EXPECT(iterations > 0 && iterations <= budget, "root GPU iteration budget respected");
                    EXPECT(std::isfinite(gap) && gap >= -1e-6, "finite root FW certificate");
                    EXPECT(solver.last_gpu_converged() == (gap >= 0 && gap < cfg.gpu_eps),
                           "convergence flag matches absolute FW gap");
                    const auto solution = solver.incumbent_x();
                    EXPECT(lmo.candidate_feasible(solution, problem.lb, problem.ub, 1e-7),
                           "root produces a feasible integer incumbent");
                };
                if (exact) run(DICGQuadraticExact{});
                else run(DICGSecant{ls});
            }
        }
    }
    std::cout << (exact_only
        ? "Interpretation: compare exact-search gap and time across budgets.\n"
        : "Interpretation: compare gap/time across budgets and default/tight/exact search.\n")

              << "Compare actual serial/paired timings to measure LMO pairing benefit.\n"
              << "Component timings are isolated probes, not a full iteration profile;\n"
              << "wall time also includes transfers and CPU postprocessing.\n";
}

static void test_quantization_round_and_prune() {
    using quantization_sparse::SparseInt4LMO;
    std::vector<double> lb{-8,-8,-8,-8,0,0,0,0};
    std::vector<double> ub{7,7,7,7,1,1,1,1};
    const std::vector<double> x{-6.6,1.2,3.6,-0.2,0.1,0.1,0.1,0.1};
    SparseInt4LMO lmo{2};
    auto candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>({-7,0,4,0,1,0,1,0}),
           "round-and-prune keeps largest absolute rounded weights and rebuilds selectors");
    EXPECT(lmo.candidate_feasible(candidate, lb, ub, 0), "round-and-prune candidate is feasible");
    lb[5] = 1; // Small second weight must stay selected.
    candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>({-7,1,0,0,1,1,0,0}),
           "round-and-prune preserves forced-on selector");
    ub[4] = 0; // Largest weight must be off.
    candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>({0,1,4,0,0,1,1,0}),
           "round-and-prune respects forced-off selector");
    lb[5] = 0; ub[4] = 1;
    candidate = SparseInt4LMO{0}.integer_candidate(x, lb, ub);
    EXPECT(candidate == std::vector<double>(8,0), "zero cardinality yields all-zero candidate");
    const std::vector<double> ties{1.2,-1.2,0,0,0,0,0,0};
    candidate = SparseInt4LMO{1}.integer_candidate(ties, lb, ub);
    EXPECT(candidate == std::vector<double>({0,-1,0,0,0,1,0,0}),
           "equal rounded magnitudes use deterministic index tie break");
    lb[4] = lb[5] = 1;
    EXPECT(SparseInt4LMO{1}.integer_candidate(x, lb, ub).empty(),
           "infeasible mandatory cardinality produces no candidate");
    lb[4] = lb[5] = 0; lb[0] = 2.2; ub[0] = 4.8;
    candidate = lmo.integer_candidate(x, lb, ub);
    EXPECT(candidate.size() == 8 && candidate[0] == 3 && candidate[4] == 1
               && lmo.candidate_feasible(candidate, lb, ub, 0),
           "round-and-prune clamps to integer node bounds and never zeros a mandatory weight");
}

int main(int argc, char** argv) {
    bool diagnostic = false, exact = false;
    int max_iters = 10000;
    try {
        if (argc > 1) {
            const std::string option = argv[1];
            if ((option != "--quantization-root" && option != "--quantization-root-exact") || argc > 3)
                throw std::invalid_argument("Usage: test_quantization [--quantization-root|--quantization-root-exact [max-iters]]");
            diagnostic = true;
            exact = option == "--quantization-root-exact";
            if (argc == 3) {
                size_t end = 0;
                max_iters = std::stoi(argv[2], &end);
                if (argv[2][end] != '\0' || max_iters < 1 || max_iters > 20000)
                    throw std::invalid_argument("max-iters must be in 1..20000");
            }
        }
        test_quantization_round_and_prune();
        if (diagnostic) {
            int count = 0;
            const auto status = cudaGetDeviceCount(&count);
            if (status != cudaSuccess || count == 0) {
                std::cerr << "A CUDA GPU is required for the root diagnostic.\n";
                return 1;
            }
            test_quantization_root_diagnostic(max_iters, exact);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
