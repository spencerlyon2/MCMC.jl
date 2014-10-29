### MCChain stores the output of a Monte Carlo iteration

type MCChain
  samples::Matrix{Float64}
  logtargets::Vector{Float64}
  gradlogtargets::Matrix{Float64}
  diagnostics::Dict
  runtime::Float64

  function MCChain(s::Matrix{Float64}, l::Vector{Float64}, g::Matrix{Float64}, d::Dict, r::Float64)
    @assert size(s, 1) == length(l) "Number of samples and logtargets values are not equal."
    if size(g) != (0, 0)
      @assert size(s) == size(g) "samples and gradlogtargets do not have the same number of rows and columns."
    end
    new(s, l, g, d, r)
  end
end

MCChain(s::Matrix{Float64}, l::Vector{Float64}, g::Matrix{Float64}, d::Dict) =
  MCChain(s, l, g, d, NaN)
MCChain(s::Matrix{Float64}, l::Vector{Float64}, g::Matrix{Float64}, r::Float64) =
  MCChain(s, l, g, Dict(), r)
MCChain(s::Matrix{Float64}, l::Vector{Float64}, d::Dict, r::Float64) = MCChain(s, l, Array(Float64, 0, 0), d, r)
MCChain(s::Matrix{Float64}, l::Vector{Float64}, g::Matrix{Float64}) = MCChain(s, l, g, Dict(), NaN)
MCChain(s::Matrix{Float64}, l::Vector{Float64}, d::Dict) = MCChain(s, l, Array(Float64, 0, 0), d, NaN)
MCChain(s::Matrix{Float64}, l::Vector{Float64}, r::Float64) = MCChain(s, l, Array(Float64, 0, 0), Dict(), r)
MCChain(s::Matrix{Float64}, l::Vector{Float64}) = MCChain(s, l, Array(Float64, 0, 0), Dict(), NaN)
MCChain() = MCChain(Array(Float64, 0, 0), Float64[], Array(Float64, 0, 0), Dict(), NaN)

function MCChain(npars::Int, nsamples::Int;
  storegradlogtarget::Bool=false, diagnostics::Dict=Dict(), runtime::Float64=NaN)
  if storegradlogtarget
    MCChain(fill(NaN, nsamples, npars), fill(NaN, nsamples), fill(NaN, nsamples, npars), diagnostics, runtime)
  else
    MCChain(fill(NaN, nsamples, npars), fill(NaN, nsamples), Array(Float64, 0, 0), diagnostics, runtime)
  end
end

function show(io::IO, c::MCChain)
  nsamples, npars = size(c.samples)
  println(io, "$npars parameters, $nsamples samples (per parameter), $(round(c.runtime, 1)) sec.")
end


## --------------------------------------------------------- ##
#- Gibbs Chain: stores the result of running a Gibbs sampler -#
## --------------------------------------------------------- ##

type GibbsChain
    samples::Dict{Symbol, Array}
    diagnostics::Dict
    runTime::Float64
end

# no diag
GibbsChain(s::Dict{Symbol, Array}, rt::Float64) = GibbsChain(s, Dict(), rt)

# no rt
GibbsChain(s::Dict{Symbol, Array}, d::Dict) = GibbsChain(s, d, NaN)

# no diag or rt
GibbsChain(s::Dict{Symbol, Array}) = GibbsChain(s, Dict(), NaN)

function show(io::IO, res::GibbsChain)
    npars = length(res.samples)
    # extract a key of this dict
    a_key = collect(keys(res.sample))[1]
    nsamlpes = size(res.samples[a_key], 1)
    msg = "Gibbs chain $npars parameters, $nsamples samples (per parameter)"
    msg *= " $(round(res.runTime, 1)) sec."
    println(io, msg)
end
