module OptimTests

using Optim, CUTEst, DataFrames, Plots

export optim_problem, initial_x, solve_problem, solution_optimum, eqconstraints_violation,
       latest_commit, profiles, save_plots

include("benchmarks.jl")

function symmetrize!(h)
    for j = 1:size(h,2)
        for i = 1:j-1
            h[i,j] = h[j,i]
        end
    end
    h
end

function cutest_fg!(nlp, x, g)
    fval,gval = objgrad(nlp, x, true)
    copy!(g, gval)
    fval
end

function cutest_hess!(nlp, x, h)
    symmetrize!(copy!(h, hess(nlp, x)))
end

function cutest_jacobian!(nlp, x, J)
    j = jac(nlp, x)
    copy!(J, j)
end

function cutest_constr_hess!(nlp, x, λ, h)
    hx = symmetrize!(hess(nlp, x; obj_weight=0.0, y=λ))
    h[:,:] += hx
    h
end

"""
    d, constraints = optim_problem(nlp)

Return the objective function and constraint information needed to run
a problem via Optim. If the problem is unconstrained,
`constraints` will be `nothing`.
"""
function optim_problem(nlp::CUTEstModel)
    d = TwiceDifferentiable(x->obj(nlp, x),
                                    (g, x)->copy!(g, grad(nlp, x)),
                                    (g, x)->cutest_fg!(nlp, x, g),
                                    (h, x)->cutest_hess!(nlp, x, h),
                                    initial_x(nlp))
#    if nlp.meta.ncon == 0 && isempty(nlp.meta.ilow) && isempty(nlp.meta.iupp)
        return d, nothing
#    end
#=    constraints = TwiceDifferentiableConstraintsFunction(
        (x,c)->cons!(nlp, x, c),
        (x,J)->cutest_jacobian!(nlp, x, J),
        (x,λ,h)->cutest_constr_hess!(nlp, x, λ, h),
        nlp.meta.lvar, nlp.meta.uvar, nlp.meta.lcon, nlp.meta.ucon)
    d, constraints
=#
end

"""
    initial_x(nlp) -> x0

Return the starting point for an optimization problem `nlp`.
"""
initial_x(nlp::CUTEstModel) = nlp.meta.x0
#=
function solve_problem(d, constraints, x0, method::Optim.ConstrainedOptimizer, options)
    optimize(d, constraints, x0, method, options)
end

function solve_problem(d, ::Void, x0, method::Optim.ConstrainedOptimizer, options)
    constraints = TwiceDifferentiableConstraintsFunction(Float64[],Float64[])
    solve_problem(d, constraints, x0, method, options)
end
=#
function solve_problem(d, ::Void, x0, method::Optim.Optimizer, options)
    optimize(d, x0, method, options)
end
#=
"""
    solve_problem(nlp, method=IPNewton(), options=OptimizationOptions())

Perform optimization on the specified nonlinear problem
`nlp`. Optionally specify the Optim `method` and `options`.
"""
function solve_problem(nlp, method=IPNewton(), options=OptimizationOptions())
    x0 = initial_x(nlp)
    d, constraints = optim_problem(nlp)
    solve_problem(d, constraints, x0, method, options)
end

"""
    eqconstraints_violation(nlp, x)
    eqconstraints_violation(constraints, x)

Return the L1-norm of the violation of the equality constraints at `x`.
"""
function eqconstraints_violation(nlp, x)
    _, constraints = optim_problem(nlp)
    eqconstraints_violation(constraints, x)
end
function eqconstraints_violation(constraints::Optim.AbstractConstraintsFunction, x)
    c = constraints.c!(x, Array{eltype(x)}(nconstraints(constraints)))
    bounds = constraints.bounds
    Δc = [x[bounds.eqx] - bounds.valx; c[bounds.eqc] - constraints.bounds.valc]
    sumabs(Δc)
end
eqconstraints_violation(::Void, x) = zero(eltype(x))
=#
"""
    solution_optimum(probname)

Parse the SIF file for the specified CUTEst problem to extract the
registered minimum. Note there are some problems where this
information has not been stored (will return `NaN`) or is inaccurate.
"""
function solution_optimum(prob::AbstractString)
    str = readstring(joinpath(ENV["MASTSIF"], addsif(prob)))
    m = match(r"SOLTN +([0-9\.\-D]+)", str)
    m == nothing && return NaN
    parse(Float64, replace(m.captures[1], "D", "e"))
end

addsif(prob::AbstractString) = endswith(prob, ".SIF") ? prob : string(prob, ".SIF")

end # module
