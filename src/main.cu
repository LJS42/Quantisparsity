#include <quantization_sparse/problem.cuh>

using namespace quantization_sparse;

int main(int argc, char** argv) {
    try {
        int k = Weights/2, samples = 2000;
        unsigned seed = 42;
        double time_limit = 3600;
        bool verify = false;
        std::string input, output;
        LineSearchKind line_search = LineSearchKind::QuadraticExact;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::printf("Usage: quantization_sparse [--k 0..%d] [--samples N] [--seed N]"
                            " [--time-limit SEC] [--verify] [--line-search secant|quadratic-exact]\n"
                            "Single row w: %d weights, X: %dxN. Signed INT4, scale=1, row nnz<=k.\n"
                            "[--input FILE] [--output FILE] (normalized w and scaled X).\n"
                            "Default line search: quadratic-exact.\n"
                            "--verify enumerates feasible Z (at most 10 million candidates).\n",
                            Weights,Weights,Weights);
                return 0;
            }
            if (arg == "--verify") { verify = true; continue; }
            if (++i >= argc) throw std::runtime_error("missing value: " + arg);
            const std::string value = argv[i];
            if (arg == "--input") { input = value; continue; }
            if (arg == "--output") { output = value; continue; }
            if (arg == "--line-search") {
                if (value == "quadratic-exact") line_search = LineSearchKind::QuadraticExact;
                else if (value == "secant") line_search = LineSearchKind::Secant;
                else throw std::runtime_error("--line-search must be secant or quadratic-exact");
                continue;
            }
            size_t end = 0;
            if (arg == "--time-limit") time_limit = std::stod(value, &end);
            else {
                const long long n = std::stoll(value, &end);
                require(n >= 0 && n <= 1000000, "integer option outside 0..1000000");
                if (arg == "--k") k = static_cast<int>(n);
                else if (arg == "--samples") samples = static_cast<int>(n);
                else if (arg == "--seed") seed = static_cast<unsigned>(n);
                else throw std::runtime_error("unknown option: " + arg);
            }
            require(end == value.size(), "invalid numeric option");
        }
        require(k <= Weights && samples > 0 && std::isfinite(time_limit)
                && time_limit > 0, "require 0<=k<=Weights, samples>0, finite time-limit>0");
        std::printf("Preparing input: X=%dx%d, seed=%u...\n", Weights, samples, seed);
        std::fflush(stdout);
        Instance data(seed, samples, input);
        DeviceBuffer dCoefficients(data.coefficients);
        QuantizationObjective objective{dCoefficients.p,data.coefficients.data()};
        Problem problem;
        problem.n_vars = Vars;
        problem.lb.assign(Vars,0); problem.ub.assign(Vars,1);
        for (int i = 0; i < Weights; ++i) { problem.lb[i] = -8; problem.ub[i] = 7; }
        problem.int_vars.resize(Vars);
        std::iota(problem.int_vars.begin(),problem.int_vars.end(),0);
        problem.f = [objective](const std::vector<double>& x) { return objective.f(x); };
        problem.grad_f = [objective](const std::vector<double>& x, std::vector<double>& g) {
            objective.gradient_host(x,g);
        };
        SparseInt4LMO lmo{k};
        auto settings = gpuboscia::create_default_settings();
        settings.algorithm = FWAlgorithmKind::DICG;
        settings.line_search.kind = line_search;
        settings.time_limit_seconds = time_limit;
        // Squared reconstruction loss is nonnegative, even when the FW
        // linearization certificate f(x)-gap is negative.
        // settings.natural_lower_bound = 0.0;
        settings.device.abs_gap_tol = 1e-6;
        settings.device.rel_gap_tol = 1e-2;
        settings.device.gpu_batch_size = 64;
        settings.device.gpu_max_iters = 2500;
        settings.device.num_streams = 2;
        settings.device.serial_iters = 1000;
        //settings.num_streams = 2;
        std::printf("Single row: %d weights, X=%dx%d, seed=%u, INT4=[-8,7], row nnz<=%d\n",
                    Weights,Weights,samples,seed,k);
        std::printf("Objective: dH matvec, read-only H+w=%.1f KiB (L2 policy), max threads/block=%d\n",
                    objective.l2_window_bytes()/1024.0,QuantizationObjective::block_threads);

        Logger logger(true, 128);
        settings.progress = [&logger](int iter, int open, double bound,
                                      double incumbent, bool fresh) {
            logger.print_row(iter, open, bound, incumbent, fresh);
            std::fflush(stdout);
        };
        LoggerParams params;
        params.traversal_strategy = "best_first (shared queue)";
        params.branching_strategy = "branch_pow2(k=3)";
        params.algorithm = fw_algorithm_name(settings.algorithm);
        params.line_search_method = line_search_name(settings.line_search.kind);
        params.n_vars = problem.n_vars;
        params.n_int_vars = static_cast<int>(problem.int_vars.size());
        params.abs_gap_tol = settings.device.abs_gap_tol;
        params.rel_gap_tol = settings.device.rel_gap_tol;
        params.time_limit_s = settings.time_limit_seconds;
        params.natural_lower_bound = settings.natural_lower_bound;
        params.random_seed = seed;
        params.print_random_seed = true;
        logger.print_header(params);
        std::fflush(stdout);

        const auto result = gpuboscia::solve(problem,objective,lmo,settings);

        SolutionStats stats;
        stats.primal_objective = result.objective;
        stats.dual_bound = result.dual_bound;
        stats.total_nodes = result.nodes_processed;
        stats.total_time_s = result.elapsed_seconds;
        stats.solution = result.x;
        stats.solution_status = result.reached_time_limit ? "Time limit reached"
            : result.reached_tolerance ? "Reached tolerance" : "Finished";
        logger.print_footer(stats);
        std::fflush(stdout);

        require(result.x.size() == Vars, "no incumbent returned");
        require(lmo.candidate_feasible(result.x,problem.lb,problem.ub,1e-7),
                "solver returned an infeasible incumbent");
        int nnz = 0;
        std::printf("w:");
        for (double value : data.w) std::printf(" %8.3f",value);
        std::printf("\nz:");
        for (int i = 0; i < Weights; ++i) {
            std::printf(" %3.0f",result.x[i]);
            nnz += result.x[i] != 0;
        }
        std::puts("");
        const double actual = data.residual(result.x);
        require(std::abs(actual-result.objective) <= 1e-7*(1+actual),
                "reported objective does not match residual");
        std::printf("Reconstruction loss=%.12g, nnz(z)=%d (limit %d)\n",
                    actual, nnz, k);
        if (!output.empty()) {
            std::ofstream stream(output);
            stream << std::setprecision(17);
            // First line: loss, lower bound, solve seconds, nodes, time-limit flag.
            stream << actual << ' ' << result.dual_bound << ' ' << result.elapsed_seconds
                   << ' ' << result.nodes_processed << ' ' << result.reached_time_limit << '\n';
            for (int i = 0; i < Weights; ++i) stream << result.x[i] << ' ';
            stream << '\n';
            stream.flush();
            if (!stream) throw std::runtime_error("could not write output file");
        }
        if (verify) {
            const double exact = data.enumerate(k);
            const double tol = 1e-6*(1+std::abs(exact));
            std::printf("exhaustive optimum=%.12g, incumbent difference=%.12g\n",exact,actual-exact);
            require(result.dual_bound <= exact+tol && actual >= exact-tol,
                    "invalid bound against exhaustive optimum");
            if (!result.reached_time_limit)
                require(actual <= exact+tol, "solver missed exhaustive optimum");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"quantization_sparse: %s\n",e.what());
        return 1;
    }
}
