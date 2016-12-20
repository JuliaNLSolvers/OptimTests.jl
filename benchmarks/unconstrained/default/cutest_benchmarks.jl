##########################################################################
#
# Benchmark optimization algorithms by tracking:
#
# * Number of iterations
# * Number of f_calls
# * Number of g_calls
# * Memory requirements (TODO)
#
####
# unconstrained
options = OptimizationOptions()
cutest_problems = CUTEst.select(max_var=5,contype = :unc, custom_filter=x->x["derivative_order"]>=2)
n = length(default_solvers)
m = length(cutest_problems)
f = open(join([version_dir, "cutest_benchmark.csv"], "/"), "w")
write(f, join(["Problem", "Optimizer", "Converged", "Time", "Minimum", "Iterations", "f_calls", "g_calls", "f_hat", "f_error", "x_error"], ","))
write(f, "\n")
@showprogress 1 "Benchmarking..." for prob in cutest_problems
    @show prob
    output = []

    nlp = CUTEstModel(prob)
    d, constraints = optim_problem(nlp)
    x0 = initial_x(nlp)

    x_hat = copy(x0)
    f_hat = obj(nlp, x_hat)
    xs = []
    times = []
        for i = 1:n
            if !(default_solvers[i] in (Newton(), NewtonTrustRegion()))
            	try
            		result = optimize(x->obj(nlp, x),(x, stor) -> grad!(nlp,x,stor), x0, default_solvers[i], options)
                    mintime = @elapsed optimize(x->obj(nlp, x),(x, stor) -> grad!(nlp,x,stor), x0, default_solvers[i], options)
                    if f_hat > Optim.minimum(result)
                        f_hat = Optim.minimum(result)
                        x_hat[:] = Optim.minimizer(result)
                    end
                    push!(output, [prob,
                                   default_names[i],
                                   Optim.converged(result),
                                   mintime,
                                   Optim.minimum(result),
                                   Optim.iterations(result),
                                   Optim.f_calls(result),
                                   Optim.g_calls(result)])
                    push!(xs, Optim.minimizer(result))
                catch
                    push!(output, ([prob,
                                   default_names[i],
                                   false,
                                   Inf,
                                   Inf,
                                   Inf,
                                   Inf,
                                   Inf]))
                    push!(xs, fill(Inf, length(x0)))
            	end
            end
        end
        for i = 1:n
            if !(default_solvers[i] in (Newton(), NewtonTrustRegion())) # should be typeof-ish instead
                write(f, join(map(x->"$x",output[i]),",")*",")
                write(f, join(map(x->"$x",[f_hat, output[i][5]-f_hat, norm(xs[i]-x_hat, Inf)]),","))
                write(f, "\n")
            end
        end
    finalize(nlp)
end
close(f)
