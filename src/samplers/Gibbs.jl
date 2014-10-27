"""
  Generic Gibbs sampler

  Parameters:
"""
immutable Gibbs <: MCSampler
end

type GibbsStash
  params::Dict{Symbol, Union(Array{Float64}, Float64)}  # current param values
  tune::MCTune  # tuner
  count::Int  # Current number of iterations
end

function inititialize(m::GibbsModel, s::Gibbs, r::MCRunner, t::MCTuner)
    GibbsStash(deepcopy(m.init), VanillaMCTune(), 0)
end

function reset!(stash::GibbsStash,
                x::Dict{Symbol, Union(Array{Float64}, Float64)})
    stash.params = deepcopy(x)
end

function initialize_task(m::GibbsModel, s::Gibbs, r::MCRunner, t::MCTuner)
    stash::GibbsStash = initialize(m, s, r, t)

    function reset(x::Dict{Symbol, Union(Array{Float64}, Float64)})
        reset!(stash, x)
    end

    task_local_storage(:reset, reset)

    while true
        iterate!(stash, m, s, r, t, produce)
    end
end


function iterate!(stash::GibbsStash, m::GibbsModel, s::Gibbs, r::GibbsRunner,
                  t::MCTuner, send::Function)
    # Localize some parameters
    local next_pars, num_blocks, block

    # initialization
    num_blocks = model.n_block

    # run all blocks
    for block=1:num_blocks
        # updates param value and distribution for each param in  block
        model.block_funcs[block](model)
    end

    # Construct a new GibbsSample
    next_pars = copy(model.curr_params)
    gs = GibbsSample(next_pars, {"accept" => true})

    # Produce the next GibbsSample
    send(gs)
end
