using Optim, OptimTests, CUTEst, ProgressMeter, Plots, DataFrames, BenchmarkTools

do_benchmarks = false
saveplots = false

pkg_dir = Pkg.dir("OptimTests")
version_sha = latest_commit("OptimTests")
benchmark_dir = pkg_dir*"/benchmarks/unconstrained/history/"
version_dir = benchmark_dir*version_sha
try
    run(`mkdir $version_dir`)
catch

end
cd(version_dir)

default_names = ["Accelerated Gradient Descent",
                 "BFGS",
                 "L-BFGS",
                 "Momentum Gradient Descent",
                 "Conjugate Gradient",
                 "Gradient Descent",
                 "Nelder-Mead",
                 "Particle Swarm",
                 "Simulated Annealing",
                 "Newton",
                 "Newton (Trust Region)"]

default_solvers =[AcceleratedGradientDescent(),
                BFGS(),
                LBFGS(),
                ConjugateGradient(),
                GradientDescent(),
                MomentumGradientDescent(),
                NelderMead(),
                ParticleSwarm(),
                SimulatedAnnealing(),
                Newton(),
                NewtonTrustRegion()]

do_benchmarks && include("default/optim_benchmarks.jl")
saveplots && save_plots(version_dir, :optim)

do_benchmarks && include("default/cutest_benchmarks.jl")
saveplots && save_plots(version_dir, :cutest)
