using OptimTests, Optim
using Base.Test

options = OptimizationOptions()
problems = CUTEst.select(max_var=100,max_con=100,custom_filter=x->x["derivative_order"]>=2)
results = Dict{String,Any}()
for prob in problems
    local result
    @show prob
    try
        result = solve_problem(prob, options)
    catch
        results[prob] = "failed"
        continue
    end
    if result == Inf
        println("Initial point is not in the interior")
        continue
    end
    results[prob] = (Optim.converged(result), result.minimum, solution_optimum(prob), result.iterations, result.f_calls)
end
