## How to run the Gibbs Sampler

Most of the work that needs to be done in order to run a Gibbs sampler
is defining the `MCGibbsModel` object. To do this you need to define a
number of things:

* The initial parameter values. This should be a `Dict{Symbol, Any}`
  where the keys are symbols that provide variable names and the values
  are anything.
* The prior distributions for all variables. This is another `Dict`
  where the keys are symbols corresponding to the parameter values and
  the values are all types of `Distribution`.
* Two more `Dict` objects that specify how the variables should be
  blocked and how the blocks should be updated
  1. A `Dict` with the same symbols for keys and integers for the
     values. These integers should range from 1 to n, where n is the
     number of blocks. The values of these integers determine the order
     in which the block updating functions will be called.
  2. A `Dict` with keys as the integers from the previous `Dict` where
     the values are functions that update the parameters in the given
     block. These functions should all accept a *single argument* that
     is an instance of he MCGibbsModel type. This function *must* update
     the parameters contained in the object `m.state` inplace, where `m` is
     instance of `MCGibbsModel`. This is better understood with an example.
* An optional `Dict` that holds arbitrary data needed to evaluate the
  block functions. This allows you to store arbitrary data within the
  model object itself. This is beneficial for at least two reasons.
  First, the block functions in the previous `Dict` can only accept one
  argument (the model instance) -- this optional `Dict` can serve as a
  way to pass in other arbitrary arguments. Second, storing all the
  information necessary to evaluate the functions defining the model
  will ensure there are no data availability or message passing issues
  if the sampler is to be run in parallel.

This seems like quite a bit of work, but it gives the user full and
complete control over specifying a general model. We allow you to design
your the algorithm however you want -- then we will take care of running
it.

The arguments above are best understood by example.

### Example

Consider the hidden Markov model defined by the measurement equation

$$
\pi_t = \tau_t + \sigma_{\pi} \epsion_{\pi, t}
$$

and the state equation

$$
\tau_t = \tau_{t-1} + \sigma_{\tau} \epsilon_{\tau, t}.
$$

Here $\pi_t$ are observed and $\tau_t$ are the hidden state.
$\sigma_{\pi}$ and $\sigma_{\tau}$ are both unknown variances and the
$\epsilon$ terms are both iid standard normal.

Here the unknown parameters are $\sigma_{\pi}$, $\sigma_{\tau}$, and the
entire history of $\tau_t$, where the history up to time $t$ is denoted
$\tau^t$. Let's break there parameters into 2 + $T$ blocks (where $T$ is
the total number of observations of $\pi$):

1. $\sigma_{\pi}^2 | \sigma_{\tau}, \tau^T \pi^T$
2. $\sigma_{\tau}^2 | \sigma_{\pi}, \tau^T \pi^T$
3. $\tau^t | \sigma_{\pi}, \sigma_{\tau} \pi^t$, $t = 1, 2, \dots, T$

With a little bit of work we can see that after appropriate
conditioning, blocks 1 and 2 each reduce to estimating the variance of a
normally distributed variable with known mean. This is a well known
problem that has a simple closed for solution. After choosing priors
$\sigma_{\pi} \sim InverseGamma(a, b)$ and $\sigma_{\tau} \sim
InverseGamma(c, d)$ we have that the posteriors are given by

$$
\sigma_{\pi}^2: InverseGamma(a+T/2, b+\frac12 \sum_{t=1}^T\nu_{\pi}_t^2)
\sigma_{\tau}^2: InverseGamma(c+T/2, d+\frac12 \sum_{t=1}^T\nu_{\tau}_t^2)
$$

where $\nu_{\pi}_t := \pi_t - \tau_t$ and $\nu_{\tau}_t := \tau_{t+1} -
\tau_t$.

Now, the third block is a bit harder. We suggest that you exploit the
fact that this is a linear, Gaussian model and use a forward filter
backward sampler. This is more complicated that what we need to explain
how this example works so we omit its derivation. If you are interested
in seeing the derivation see [these slides](TODO Add ref to Chase's
slides). From this block we need to know two important things: from the
structure of the problem, we need to specify a prior for only $\tau_0$
*and* we should update all $\tau_t, \; t = 1, 2, \dots, T$ in a single
block.

With that background in place we are ready to show how to write the
model in the appropriate form.

#### Constructing the `MCGibbsModel`

Below we show how to represent the model above in the correct form

```julia
using Distributions
using MCMC

pi_obs = ... # fill this in for how you collect data
T = size(pi_obs, 1)

# define the priors
priors = Dict{Symbol, Distribution}()
priors[:τ_0] = Normal(.05, 3)
priors[:σ_π2] = InverseGamma(2, 2)
priors[:σ_τ2] = InverseGamma(2, 2)

# Set initial values to be draws from the Prior
init_vals = Dict{Symbol, VectorOrOne{Float64}}()
init_vals[:τ] = fill(rand(p[:τ_0]), T)
init_vals[:σ_π2] = rand(p[:σ_π2])
init_vals[:σ_τ2] = rand(p[:σ_τ2])

# set up data Dict to attach to model
data = Dict{Any, Any}()
data["pi_obs"] = pi_obs
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


function τ_block!(m::MCGibbsModel)
    ...
end

function σ_π2_block!(m::MCMCGibbsModel)
    # pull out distribution/parameters
    dist = m.dists[:σ_π2]
    a, b = dist.shape, dist.scale

    T = m.data["T"]  # number of observations

    # innovations in measurement equation
    ζ = m.data["pi_obs"] - m.state[:τ]

    # update the parameters
    a_new_sigma = a + T / 2
    b_new_sigma = b + dot(ζ, ζ) / 2.0

    # update distribution
    posterior = InverseGamma(a_new_sigma, b_new_sigma)

    # sample from updated distribution
    m.state[:σ_π2] = rand(m.dists[:σ_π2])
    nothing
end


function σ_τ2_block!(m::MCMCGibbsModel)
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
```

At this point we can construct the model

```
m = MCGibbsModel(init_vals,       # Dict: symbols to initial values
                priors,           # Dict: symbols to prior Distributions
                block_sym_to_num, # Dict: blocked symbols to Ints
                block_funcs,      # Dict: Int to funcs
                data)             # Dict: arbitrary data
```
