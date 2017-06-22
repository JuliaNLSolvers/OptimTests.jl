f = open(join([version_dir, "cutest_linesearch_benchmark.csv"], "/"), "w")
write(f, join(["Problem", "Optimizer", "Converged", "Time", "Minimum", "Iterations", "f_calls", "g_calls", "h_calls", "f_hat", "f_error", "x_error"], ","))
write(f, "\n")

options = Optim.Options(show_trace=true)
cutest_problems = CUTEst.select(max_var=100,contype = :unc, custom_filter=x->x["derivative_order"]>=2)

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
                        (LineSearches.BackTracking(order = 3), "BackTracking (cubic)"),
                        (LineSearches.HagerZhang(), "HagerZhang"),
                        (LineSearches.MoreThuente(), "MoreThuente"),
                    #    (LineSearches.Static(alpha=0.1), "Static (alpha = 0.1)"),
                    #    (LineSearches.Static(), "Static (alpha = 1)"),
                        (LineSearches.StrongWolfe(), "StrongWolfe")]

n = length(linesearch_solvers)
m = length(cutest_problems)
@showprogress 1 "Benchmarking..." for prob in cutest_problems
if prob in ("STRATEC", "DMN37142LS", "DMN37143LS") # these take a looong time
    continue
end
    @show prob
    output = []

    nlp = CUTEstModel(prob)
    x0 = initial_x(nlp)

    x_hat = copy(x0)
    f_hat = obj(nlp, x_hat)
    xs = []
    times = []
    for i = 1:n
        for ls in default_linesearches
        	try
                d, constraints = optim_problem(nlp)
        		result = optimize(d, linesearch_solvers[i](linesearch=ls[1]), options)
                mintime = @elapsed optimize(d, x0, linesearch_solvers[i](linesearch=ls[1]), options)
                if f_hat > Optim.minimum(result)
                    f_hat = Optim.minimum(result)
                    x_hat[:] = Optim.minimizer(result)
                end
                push!(output, [prob,
                               linesearch_solver_names[i]*" with "*ls[2],
                               Optim.converged(result),
                               mintime,
                               Optim.minimum(result),
                               Optim.iterations(result),
                               Optim.f_calls(result),
                               Optim.g_calls(result),
                               Optim.h_calls(result)])
                push!(xs, Optim.minimizer(result))
            catch
                push!(output, ([prob,
                               linesearch_solver_names[i]*" with "*ls[2],
                               false,
                               Inf,
                               Inf,
                               Inf,
                               Inf,
                               Inf,
                               Inf]))
                push!(xs, fill(Inf, length(x0)))
        	end
        end
        for i = 1:length(default_linesearches)
            write(f, join(map(x->"$x",output[i]),",")*",")
            write(f, join(map(x->"$x",[f_hat, output[i][5]-f_hat, norm(xs[i]-x_hat, Inf)]),","))
            write(f, "\n")
        end
    end
    finalize(nlp)
end
close(f)
