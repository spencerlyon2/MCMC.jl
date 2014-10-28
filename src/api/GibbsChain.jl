# GibbsChain, stores the result of running a Gibbs sampler

type GibbsChain
  samples::Dict{Symbol, F64OrVectorF64}
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
  nsamples, npars = size(res.samples)
  msg = "Gibbs chain $npars parameters, $nsamples samples (per parameter)"
  msg *= " $(round(res.runTime, 1)) sec."
  println(io, msg)
end
