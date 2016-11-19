using OptimTests, Optim
using Base.Test

# unconstrained
unc_problems = CUTEst.select(max_var=10,contype = :unc, custom_filter=x->x["derivative_order"]>=2)
unc_results = Dict{String,Any}()
for prob in unc_problems
    local result
    @show prob
    try
        result = solve_problem(prob, GradientDescent())
    catch
        unc_results[prob] = "failed"
        continue
    end
    if result == Inf
        println("Initial point is not in the interior")
        continue
    end
    unc_results[prob] = (Optim.converged(result), result.minimum, solution_optimum(prob), result.iterations, result.f_calls)
end

# constrained
con_problems = CUTEst.select(max_var=100,max_con=100,custom_filter=x->x["derivative_order"]>=2)
con_results = Dict{String,Any}()
for prob in con_problems
    local result
    @show prob
    try
        result = solve_problem(prob, IPNewton())
    catch
        con_results[prob] = "failed"
        continue
    end
    if result == Inf
        println("Initial point is not in the interior")
        continue
    end
    con_results[prob] = (Optim.converged(result), result.minimum, solution_optimum(prob), result.iterations, result.f_calls)
end
