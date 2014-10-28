typealias FunctionOrNothing Union(Function, Nothing)
typealias F64OrVectorF64 Union(Float64, Vector{Float64})

abstract MCModel

### Sampler types hold the components that fully specify a Monte Carlo sampler
### The fields of sampler types are user-inputed

abstract MCSampler

abstract MHSampler <: MCSampler # Family of samplers based on Metropolis-Hastings
abstract HMCSampler <: MCSampler # Family of Hamiltonian Monte Carlo samplers
abstract LMCSampler <: MCSampler # Family of Langevin Monte Carlo samplers

### Runner types indicate what type of simulation will be run (ex serial or sequential Monte Carlo)
### Their fields fully specify the simulation details (ex total number or number of burn-in iterations)

abstract ParallelismLevel
type Serial <: ParallelismLevel end
type Parallel <: ParallelismLevel end

abstract MCRunner{P<:ParallelismLevel}

typealias SerialMCRunner MCRunner{Serial}
typealias ParrallelMCRunner MCRunner{Parallel}

### Tuner types

abstract MCTuner

### Monte Carlo sample types hold a single Monte Carlo sample
### A typical MCSample includes at least the vector of parameter values and the respective log-target value

abstract DerivativeOrder
type NullOrder <: DerivativeOrder end
type FirstOrder <: DerivativeOrder end
type SecondOrder <: DerivativeOrder end
type ThirdOrder <: DerivativeOrder end

abstract MCSample{O<:DerivativeOrder}

### Generic MCBaseSample type used by most samplers

type MCBaseSample <: MCSample{NullOrder}
  sample::Vector{Float64}
  logtarget::Float64
end

# TODO: I don't need this type to hold anything. I just have it so I can
#       write GibbsStash <: MCStash{GibbsSample}
type GibbsSample <: MCSample{NullOrder} end

MCBaseSample(s::Vector{Float64}) = MCBaseSample(s, NaN)
MCBaseSample() = MCBaseSample(Float64[], NaN)

MCBaseSample(l::Int) = MCBaseSample(fill(NaN, l), NaN)

logtarget!(s::MCSample, f::Function) = (s.logtarget = f(s.sample))

### MCGradSample is used by samplers that compute (up to) the gradient of the log-target (for ex HMC and MALA)

type MCGradSample <: MCSample{FirstOrder}
  sample::Vector{Float64}
  logtarget::Float64
  gradlogtarget::Vector{Float64}
end

MCGradSample(s::Vector{Float64}) = MCGradSample(s, NaN, Float64[])
MCGradSample() = MCGradSample(Float64[], NaN, Float64[])

MCGradSample(l::Int) = MCGradSample(fill(NaN, l), NaN, fill(NaN, l))

gradlogtargetall!(s::MCSample, f::Function) = ((s.logtarget, s.gradlogtarget) = f(s.sample))

### MCMCTensorSample is used by samplers that compute (up to) the tensor of the log-target (for ex SMMALA)

type MCTensorSample <: MCSample{SecondOrder}
  sample::Vector{Float64}
  logtarget::Float64
  gradlogtarget::Vector{Float64}
  tensorlogtarget::Matrix{Float64}
end

MCTensorSample(s::Vector{Float64}) = MCTensorSample(s, NaN, Float64[], Array(Float64, 0, 0))
MCTensorSample() = MCTensorSample(Float64[], NaN, Float64[], Array(Float64, 0, 0))

MCTensorSample(l::Int) = MCTensorSample(fill(NaN, l), NaN, fill(NaN, l), fill(NaN, l, l))

tensorlogtargetall!(s::MCSample, f::Function) = ((s.logtarget, s.gradlogtarget, s.tensorlogtarget) = f(s.sample))

### MCMCDTensorSample is used by samplers that compute (up to) the derivative of the tensor of the log-target (for ex
### RMHMC and PMALA)

type MCDTensorSample <: MCSample{ThirdOrder}
  sample::Vector{Float64}
  logtarget::Float64
  gradlogtarget::Vector{Float64}
  tensorlogtarget::Matrix{Float64}
  dtensorlogtarget::Array{Float64, 3}
end

MCDTensorSample(s::Vector{Float64}) = MCDTensorSample(s, NaN, Float64[], Array(Float64, 0, 0), Array(Float64, 0, 0, 0))
MCDTensorSample() = MCDTensorSample(Float64[], NaN, Float64[], Array(Float64, 0, 0), Array(Float64, 0, 0, 0))

MCDTensorSample(l::Int) = MCDTensorSample(fill(NaN, l), NaN, fill(NaN, l), fill(NaN, l, l), fill(NaN, l, l, l))

dtensorlogtargetall!(s::MCSample, f::Function) =
  ((s.logtarget, s.gradlogtarget, s.tensorlogtarget, s.dtensorlogtarget) = f(s.sample))

### Tune types hold the temporary output of the sampler that is used for tuning the sampler

abstract MCTune

# MCState holds two samples at each Monte Carlo iteration, the current and the successive one

type MCState{S<:MCSample}
  successive::S # If proposed sample is accepted, then successive = proposed, otherwise successive = current
  current::S
  diagnostics::Dict
end

MCState{S<:MCSample}(p::S, c::S) = MCState(p, c, Dict())

### Stash types hold the temporary components used by a Monte Carlo sampler during its run
### This means that stash types represent the internal state ("local variables") of a Monte Carlo sampler

abstract MCStash{S<:MCSample}

### Monte Carlo Jobs (ex plain jobs, jobs using tasks or MPI jobs)

abstract MCJob

### MCSystem gathers all the components that define a Monte Carlo simulation
### It is in a sense a complete specification of a Monte Carlo simulation based on user input
### At the same time, MCSystem stores vital internal components (ex MCSystem.job.task for task-based jobs)
### The user mainly interacts with the MCSystem type at a higher level via the package's interface
### Users familiar with the package can also interact with the MCSystem type directly

type MCSystem
  model::MCModel
  sampler::MCSampler
  runner::MCRunner
  tuner::MCTuner
  job::MCJob
end

# Convenience constructors for setting up a vector of Monte Carlo simulations

MCSystem(m::MCModel, s::MCSampler, r::MCRunner, t::MCTuner=VanillaMCTuner()) = MCSystem(m, s, r, t)
MCSystem{M<:MCModel, S<:MCSampler, R<:MCRunner, T<:MCTuner}(m::Vector{M}, s::Vector{S}, r::Vector{R}, t::Vector{T}) =
  map(MCSystem, m, s, r, t)
MCSystem{M<:MCModel, S<:MCSampler, R<:MCRunner}(m::Vector{M}, s::Vector{S}, r::Vector{R}) = map(MCSystem, m, s, r)

