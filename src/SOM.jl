module SOM

using Distances

export Config, State, init

struct Config
    n_data::Int
    dim_data::Int
    dim_latent::Int
    n_units_per_side::Int
    n_units::Int

    rad_init::Float64
    rad_min::Float64
    rad_convergence::Float64

    function Config(;
            n_data::Int,
            dim_data::Int,
            dim_latent::Int = 2,
            n_units_per_side::Int,
            rad_init::Float64 = 1.0,
            rad_min::Float64 = 0.1,
            rad_convergence::Float64 = 0.05)
        return new(n_data, dim_data, dim_latent, n_units_per_side, n_units_per_side^dim_latent, rad_init, rad_min, rad_convergence)
    end
end

struct State
    units::Array{Float64,2}
    bmus::Union{Array{Int,1},Nothing}
    reference_vectors::Union{Array{Float64,2},Nothing}

    function State(;
            units::Array{Float64,2},
            bmus::Union{Array{Int,1},Nothing} = nothing,
            reference_vectors::Union{Array{Float64,2},Nothing} = nothing)
        return new(units, bmus, reference_vectors)
    end
end

function meshgrid(d::Int, n::Int)::Array{Float64,2}
    x = zeros(n^2, d)
    u = repeat(range(0.0, 1.0, length = n), 1, repeat([n], d - 1)...)
    for i = 1:d
        p = [if j == i; 1; elseif j == 1; i; else j; end for j = 1:d]
        x[:,i] .= permutedims(u, p)[:]
    end
    return x
end

function init(config::Config)::State
    return State(units = meshgrid(config.dim_latent, config.n_units_per_side),
        bmus = rand(1:config.n_units, config.n_data))
end

function estep(;data::Array{Float64,2}, reference_vectors::Array{Float64,2})::Array{Int,1}
    d = pairwise(Euclidean(), data, reference_vectors, dims = 1)
    return [ci[2] for ci = findmin(d, dims = 2)[2][:]]
end

function data_weight(bmus::Array{Int,1}, units::Array, neighborhood_radius::Float64)::Array{Float64}
    D = pairwise(Euclidean(), units, units, dims = 1)
    n_units = size(units)[1]
    B = [k == k_ for k = 1:n_units, k_ = bmus]
    R = exp.(-0.5 / neighborhood_radius^2 * (D * B))
    A = R ./ sum(R, dims = 2)
end

function mstep(;
        data::Array{Float64,2},
        units::Array{Float64,2},
        bmus::Array{Int,1},
        neighborhood_radius::Float64)::Array{Float64,2}
    return data_weight(bmus, units, neighborhood_radius) * data
end

function fit(config::Config, state::State, data::Array{Float64,2})::State
    units = state.units
    bmus = something(state.bmus)
    reference_vectors = if state.reference_vectors == nothing
        mstep(data = data, units = data, bmus = bmus, neighborhood_radius = config.rad_init)
    else
        state.reference_vectors
    end

    println(sum(reference_vectors))

    for t = 1:100
        rad = config.rad_min + (config.rad_init - config.rad_min) * exp(-0.5 * t / config.rad_convergence^2)
        bmus = estep(data = data, reference_vectors = reference_vectors)
        reference_vectors = mstep(data = data, units = units, bmus = bmus, neighborhood_radius = rad)
    end

    println(sum(reference_vectors))

    return State(units = units, bmus = bmus, reference_vectors = reference_vectors)
end

end

using PyPlot

function main()
    config = SOM.Config(n_data = 200, dim_data = 3, dim_latent = 2, n_units_per_side = 10)
    X = rand(config.n_data, config.dim_data) .* 2 .- 1
    X[:, 3] = X[:, 1].^2 - X[:, 2].^2

    state = SOM.init(config)
    state = SOM.fit(config, state, X)

    y = state.reference_vectors
    shape = (config.n_units_per_side, config.n_units_per_side)

    plot_wireframe(reshape(y[:,1], shape), reshape(y[:,2], shape), reshape(y[:,3], shape))
    plt.show()
end

main()