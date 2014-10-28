using Distributions
# using MCMC

typealias VectorOrOne{T} Union(Vector{T}, T)

## ---------------------- ##
#- Define block functions -#
## ---------------------- ##

function τ_block!(m::MCGibbsModel)
    # Pull out prior for τ_0. Get starting mean and covariance
    dist = m.dists[:τ_0]
    tau, P = dist.μ, dist.σ

    # pull out current σ_π2 and current σ_τ2 as well as data on π, T
    σ_π2, σ_τ2 = m.state[:σ_π2], m.state[:σ_τ2]
    π = m.data["inflation"]
    T = m.data["T"]

    # Generate arrays to hold mean/variance updates each step
    tau_tt = Array(Float64, T+1)
    P_tt = Array(Float64, T+1)
    P_tp1_t = Array(Float64, T)
    tau_tt[1] = tau
    P_tt[1] = P

    # TODO: abstract this into a forward_filter(s::LinearGuassianSS) function

    # run Kalman Filter forward
    for t=2:T+1
        tau = tau_tt[t - 1]    # predict state
        P = P_tt[t - 1] + σ_τ2 # predicted covariance for state
        y = π[t - 1] - tau     # measurement residual
        S = P + σ_π2           # measurement innovation covariance
        K = P / S              # Kalman gain
        tau_tt[t] = tau + K*y  # updated state estimate
        P_tt[t] = P -  K * P   # updated state covariance estimate

        # also store this... it is P_{t+1|t}
        P_tp1_t[t-1] = P
    end

    # remove τ_0 and P_0 from filtered state
    tau_tt = tau_tt[2:end]
    P_tt = P_tt[2:end]

    # allocate memory for the actual sample and get \tau_T draw
    τ_sample = Array(Float64, T)
    τ_sample[end] = rand(Normal(tau_tt[end], sqrt(P_tt[end])))

    # TODO: abstract into backward_sampler(s::LinearGuassianSS, X_tt, P_tt,
    #                                      P_tp1_t),
    #       where X_tt, P_tt P_tp1_t are arrays of filtered mean/covariance
    #       state estimates

    # run backwards sampler
    for t=T-1:-1:1
        τ_t_tp1 = tau_tt[t] + P_tt[t] / P_tp1_t[t] * (τ_sample[t+1]-tau_tt[t])
        P_t_tp1 = P_tt[t] - P_tt[t]^2 / P_tp1_t[t]
        new_dist = Normal(τ_t_tp1, sqrt(P_t_tp1))
        τ_sample[t] = rand(new_dist)
    end

    m.state[:τ] = copy(τ_sample)
    nothing
end


function σ_π2_block!(m::MCGibbsModel)
    # pull out distribution/parameters
    dist = m.dists[:σ_π2]
    a, b = dist.shape, dist.scale

    T = m.data["T"]  # number of observations

    # innovations in measurement equation
    ζ = m.data["inflation"] - m.state[:τ]

    # update the parameters
    a_new_sigma = a + T / 2
    b_new_sigma = b + dot(ζ, ζ) / 2.0

    # update distribution
    posterior = InverseGamma(a_new_sigma, b_new_sigma)

    # sample from updated distribution
    m.state[:σ_π2] = rand(m.dists[:σ_π2])
    nothing
end


function σ_τ2_block!(m::MCGibbsModel)
    # pull out distribution/parameters
    dist = m.dists[:σ_τ2]
    a, b = dist.shape, dist.scale

    # construct innovations in state equation
    τ = m.state[:τ]
    ν = similar(τ)
    ν[2:end] = τ[2:end] - τ[1:end-1]
    ν[1] = τ[1] - m.data["tau_0"]  # special case for τ₀

    T = m.data["T"]  # number of observations

    # update the parameters
    a_new_tau = a + T / 2
    b_new_tau = b + dot(ν, ν) / 2.0

    # find posterior
    posterior = InverseGamma(a_new_tau, b_new_tau)

    # sample from updated distribution
    m.state[:σ_τ2] = rand(posterior)
    nothing
end


## ------------------------- ##
#- Construct MCGibbsModel objects -#
## ------------------------- ##

# function to easily construct an initial draw, given the priors and length of
# data series
function init_draw(p::Dict, n_τ::Int)
    out = Dict{Symbol, VectorOrOne{Float64}}()
    # fill all values for τ with draw from prior. This is just to make it
    # type stable, we don't actually use anything but the first value
    out[:τ] = fill(rand(p[:τ_0]), n_τ)
    out[:σ_π2] = rand(p[:σ_π2])
    out[:σ_τ2] = rand(p[:σ_τ2])
    return out
end


# This function uses the tools above to construct `n` instances of
# MCGibbsModel.
function prepare_models(n::Int=1)

    infl = readcsv("inflation_data.csv")

    # set up data Dict to attach to model
    data = Dict{Any, Any}()
    data["inflation"] = squeeze(infl, 2)
    data["T"] = length(infl)

    # set up Dict to map block numbers into functions
    block_funcs = Dict{Int, Function}()
    block_funcs[1] = τ_block!
    block_funcs[2] = σ_π2_block!
    block_funcs[3] = σ_τ2_block!

    # set up dict to map paramter symbols into block numbers
    block_sym_to_num = Dict{Symbol, Int}()
    block_sym_to_num[:τ_0] = 1
    block_sym_to_num[:σ_π2] = 2
    block_sym_to_num[:σ_τ2] = 3

    # Dict to hold priors.
    priors = Dict{Symbol, Distribution}()
    priors[:τ_0] = Normal(.05, 3)  # prior for τ₀
    priors[:σ_π2] = InverseGamma(2, 2)
    priors[:σ_τ2] = InverseGamma(2, 2)

    n_blocks = 3

    out = Array(MCGibbsModel, n)

    for i=1:n
        # Dict to hold initial params. Just draws from priors
        state = init_draw(priors, data["T"])
        data["tau_0"] = state[:τ][1]

        out[i] = MCGibbsModel(state, priors, block_sym_to_num,
                                block_funcs, data, n_blocks)
    end

    return out
end

# this is the main script. If we are on the root process it creates a
# new model/system for every worker, then runs them in parallel.
if myid() == 1
    mods = prepare_models(nworkers())
    # systems = MCSystem(mods, Gibbs(), SerialGibbs(steps=100000, burnin=10000))
    # chains = prun(tsks)
end
