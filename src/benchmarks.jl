# Get the latest commit as a string due to Ismael Venegas Castelló (@Ismael-VC)
"""
Returns the shortened SHA of the latest commit of a package.
"""
function latest_commit(pkg::String)::String
    original_directory = pwd()
    cd(Pkg.dir(pkg))
    commit = readstring(`git rev-parse --short HEAD`) |> chomp
    cd(original_directory)
    return commit
end


"""
Returns, for each solver, a vector of counts of the number of problems in the DataFrame
that is below the threshold values given.
"""
function profiles(names, df, tau, measure)
    profiles_out = []
    for name in names
        profile = zeros(tau)
        rows =  df[:Optimizer].==name
        for i = 1:length(tau)
            profile[i] = sum(df[rows, measure].<=tau[i])/length(unique(df[:Problem]))
        end
        push!(profiles_out, profile)
    end
    profiles_out
end

"""
Saves two figures showing performance profiles for the objective error and the
error in the minimizer across problems and solvers.
"""
function save_plots(version_dir, testset; tau = 10.0.^(-16:10), legendpos = :bottomright)

    if testset == :cutest
        teststr = "cutest"
    elseif testset == :optim
        teststr = "optim"
    elseif testset == :optim_linesearch
        teststr = "optim_linesearch"
    elseif testset == :cutest_linesearch
        teststr = "cutest_linesearch"
    else
        err("Symbol not supported")
    end

    str = version_dir*"/"*teststr*"_benchmark.csv"
    df = readtable(str)
    names = unique(df[:Optimizer])
    f_profiles = profiles(names, df, tau, :f_error)
    f_err = plot(tau, hcat(f_profiles...),
            label = hcat(names...),
            lc=[:black :red :green :black :red :green :black :red :green :black :red :green],
            ls=[:solid  :solid :solid :dash :dash :dash :dashdot :dashdot :dashdot :dot :dot :dot],
            size =(800,400),
            ylims = (0,1),
            line = :steppre, xscale=:log10, xlabel = "Error level", ylabel = "Proportion of problems",
            title = "Measure: f-f*",
            legend=legendpos)
    savefig("f_err_"*teststr)

    x_profiles = profiles(names, df, tau, :x_error)
    x_err = plot(tau, hcat(x_profiles...),
            label = hcat(names...),
            lc=[:black :red :green :black :red :green :black :red :green :black :red :green],
            ls=[:solid  :solid :solid :dash :dash :dash :dashdot :dashdot :dashdot :dot :dot :dot],
            size =(800,400),
            ylims = (0,1),
            line = :steppre, xscale=:log10, xlabel = "Error level", ylabel = "Proportion of problems",
            title = "Measure: sup-norm of x-x*.",
            legend=legendpos)
    savefig("x_err_"*teststr)
end
