f = open(join([version_dir, "optim_linesearch_benchmark.csv"], "/"), "w")
write(f, join(["Problem", "Optimizer", "Converged", "Time", "Minimum", "Iterations", "f_calls", "g_calls", "h_call", "f_hat", "f_error", "x_error"], ","))
write(f, "\n")

linesearch_solver_names = ["Accelerated Gradient Descent",
                           "BFGS",
                           "L-BFGS",
                           "Conjugate Gradient",
                           "Momentum Gradient Descent",
                           "Gradient Descent",
                           "Newton"]
linesearch_solver_names = ["Newton",]

linesearch_solvers =[AcceleratedGradientDescent,
                            BFGS,
                            LBFGS,
                            ConjugateGradient,
                            MomentumGradientDescent,
                            GradientDescent,
                            Newton]
linesearch_solvers = [Newton, ]

default_linesearches = [(LineSearches.BackTracking(order = 2), "BackTracking (quadratic)"),
                        (LineSearches.BackTracking(order = 3), "BackTracking (qubic)"),
                        (LineSearches.HagerZhang(), "HagerZhang"),
                        (LineSearches.MoreThuente(), "MoreThuente"),
                        (LineSearches.Static(alpha=0.1), "Static (alpha = 0.1)"),
                        (LineSearches.Static(), "Static (alpha = 1)"),
                        (LineSearches.StrongWolfe(), "StrongWolfe")]

@showprogress 1 "Benchmarking..." for (name, problem) in Optim.UnconstrainedProblems.examples
    for (i,  algorithm) in enumerate(linesearch_solvers)
        for ls in default_linesearches
            print_debuginfo && @show name, problem , algorithm
            try
            # Force compilation and obtain results
            results = optimize(problem.f, problem.g!, problem.h!,
                               problem.initial_x, algorithm(linesearch=ls[1]), Optim.Options(g_tol = 1e-16))
            # Run each algorithm n times
            n = 10
            if algorithm == ParticleSwarm() && name == "Large Polynomial"
                n = 1
            end
            # Estimate run time in seconds
            run_time = minimum([@elapsed optimize(problem.f, problem.g!, problem.h!,
                                   problem.initial_x,
                                   algorithm(linesearch=ls[1]), Optim.Options(g_tol = 1e-16)) for nn = 1:n])

            # Count iterations
            iterations = results.iterations

            # Print out results.
            write(f, join([problem.name,
                           linesearch_solver_names[i]*" with "*ls[2],
                           Optim.converged(results),
                           run_time,
                           Optim.minimum(results),
                           iterations,
                           Optim.f_calls(results),
                           Optim.g_calls(results),
                           Optim.h_calls(results),
                           problem.f(problem.solutions),
                           Optim.minimum(results)-problem.f(problem.solutions),
                           norm(Optim.minimizer(results)-problem.solutions, Inf)], ","))
            write(f, "\n")
            catch m
                println(m)
                write(f, join([problem.name,
                               linesearch_solver_names[i]*" with "*ls[2],
                               "false",
                               "Inf",
                               "Inf",
                               "Inf",
                               "Inf",
                               "Inf",
                               "Inf",
                               "Inf",
                               "Inf",
                               "Inf"], ","))
                write(f, "\n")
            end
        end
    end
end
close(f)
