using OptimTests, Optim, CUTEst, NLPModels
using Base.Test

ipoptdata = Float64[]
function intermediate(alg_mod::Int, iter_count::Int,
                      obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
                      d_norm::Float64, regularization_size::Float64, alpha_du::Float64,
                      alpha_pr::Float64, ls_trials::Int)
    push!(ipoptdata, obj_value)
    return true
end

immutable Results
    converged::Bool
    obj_optim::Float64
    obj_ipopt::Float64
    obj_cutest::Float64
    cv_initial::Float64
    cv_optim::Float64
    cv_ipopt::Float64
    iter_optim::Int
    iter_ipopt::Int
    fcalls::Int
    objtrace_optim::Vector{Float64}
    objtrace_ipopt::Vector{Float64}
end

# unconstrained
options = Optim.Options(store_trace=true)
unc_problems = CUTEst.select(max_var=10,contype = :unc, custom_filter=x->x["derivative_order"]>=2)
unc_results = Dict{String,Any}()
for prob in unc_problems
    local result
    @show prob
    nlp = CUTEstModel(prob)
    d, constraints = optim_problem(nlp)
    x0 = initial_x(nlp)
    try
        result = solve_problem(d, constraints, x0, GradientDescent(), options)
    catch
        unc_results[prob] = "failed"
        finalize(nlp)
        continue
    end
    if result == Inf
        println("Initial point is not in the interior")
        continue
    end
    unc_results[prob] = (Optim.converged(result), result.minimum, solution_optimum(prob), result.iterations, result.f_calls)
    finalize(nlp)
end
@test sum(v=="failed" for v in values(unc_results)) < length(unc_results)
#=
# constrained
con_problems = CUTEst.select(max_var=100,max_con=100,custom_filter=x->x["derivative_order"]>=2)
con_results = Dict{String,Any}()
for prob in con_problems
    local result
    @show prob
    nlp = CUTEstModel(prob)
    d, constraints = optim_problem(nlp)
    x0 = initial_x(nlp)
    if !isinterior(constraints, x0)
        println("Initial point is not in the interior")
        finalize(nlp)
        continue
    end
    try
        result = solve_problem(d, constraints, x0, IPNewton(), options)
    catch
        con_results[prob] = "failed"
        finalize(nlp)
        continue
    end
    # Compare against Ipopt
    model = NLPtoMPB(nlp, IpoptSolver())
    setIntermediateCallback(model.inner, intermediate)
    empty!(ipoptdata)
    MathProgBase.optimize!(model)
    # Collect statistics
    niter = min(result.iterations, length(ipoptdata))
    objOptim = [tr.value for tr in result.trace[1:niter]]
    objIpopt = ipoptdata[1:niter]
    con_results[prob] = Results(Optim.converged(result), result.minimum,
                                MathProgBase.getobjval(model), solution_optimum(prob),
                                eqconstraints_violation(nlp, nlp.meta.x0),
                                eqconstraints_violation(nlp, result.minimizer),
                                eqconstraints_violation(nlp, MathProgBase.getsolution(model)),
                                result.iterations, length(ipoptdata), result.f_calls,
                                objOptim, objIpopt)
    finalize(nlp)
end
@test sum(v=="failed" for v in values(con_results)) < length(con_results)
=#
