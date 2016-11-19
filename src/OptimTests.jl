module OptimTests

using Optim, CUTEst

export solve_problem, solution_optimum

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

function solve_problem(nlp::CUTEstModel, method::Optim.Optimizer, options)
    x0 = nlp.meta.x0
    d = TwiceDifferentiableFunction(x->obj(nlp, x),
                                    (x,g)->copy!(g, grad(nlp, x)),
                                    (x,g)->cutest_fg!(nlp, x, g),
                                    (x,h)->cutest_hess!(nlp, x, h))
    if nlp.meta.ncon == 0 && isempty(nlp.meta.ilow) && isempty(nlp.meta.iupp)
        result = optimize(d, x0, method, options)
    else
        return Inf
    end
    result
end

function solve_problem(nlp::CUTEstModel, method::Optim.IPNewton, options)
    x0 = nlp.meta.x0
    d = TwiceDifferentiableFunction(x->obj(nlp, x),
                                    (x,g)->copy!(g, grad(nlp, x)),
                                    (x,g)->cutest_fg!(nlp, x, g),
                                    (x,h)->cutest_hess!(nlp, x, h))
    if nlp.meta.ncon == 0 && isempty(nlp.meta.ilow) && isempty(nlp.meta.iupp)
        result = optimize(d, TwiceDifferentiableConstraintsFunction(Float64[],Float64[]),
                          x0, method, options)
    else
        constraints = TwiceDifferentiableConstraintsFunction(
            (x,c)->cons!(nlp, x, c),
            (x,J)->cutest_jacobian!(nlp, x, J),
            (x,λ,h)->cutest_constr_hess!(nlp, x, λ, h),
            nlp.meta.lvar, nlp.meta.uvar, nlp.meta.lcon, nlp.meta.ucon)
        if !isinterior(constraints, x0)
            return Inf
        end
        result = optimize(d, constraints, x0, Optim.method, options)
    end
    result
end

function solve_problem(prob::AbstractString, method::Optim.Optimizer, options::Optim.OptimizationOptions = Optim.OptimizationOptions())
    nlp = CUTEstModel(prob)
    local result
    try
        result = solve_problem(nlp, method, options)
    finally
        finalize(nlp)
    end
    result
end

function solution_optimum(prob::AbstractString)
    str = readstring(joinpath(ENV["MASTSIF"], addsif(prob)))
    m = match(r"SOLTN +([0-9\.\-D]+)", str)
    m == nothing && return NaN
    parse(Float64, replace(m.captures[1], "D", "e"))
end

addsif(prob::AbstractString) = endswith(prob, ".SIF") ? prob : string(prob, ".SIF")

end # module
