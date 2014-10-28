"""
Generic Gibbs sampler

    Parameters:

## Notes (comments welcome)

* I don't attach much to the stash object. All we need is the state and
  sample count.
    * I don't include a MCTune object because we always accept.


"""
immutable Gibbs <: MCSampler
end

type GibbsStash <: MCStash{GibbsSample}
    state::Dict{Symbol}  # current param values
    count::Int           # Current number of iterations
end


function inititialize(m::MCGibbsModel, s::Gibbs, r::MCRunner, t::MCTuner)
    GibbsStash(GibbsState(deepcopy(m.init), Dict()), 0)
end


function reset!(stash::GibbsStash, x::Dict{Symbol, F64OrVectorF64})
    stash.state.sample = deepcopy(x)
end


function initialize_task(m::MCGibbsModel, s::Gibbs, r::MCRunner, t::MCTuner)
    stash::GibbsStash = initialize(m, s, r, t)

    reset(x::Dict{Symbol, F64OrVectorF64}) = reset!(stash, x)
    task_local_storage(:reset, reset)

    while true
        iterate!(stash, m, s, r, t, produce)
    end
end


function iterate!(stash::GibbsStash, m::MCGibbsModel, s::Gibbs, r::MCRunner,
                  t::MCTuner, send::Function)
    # update all parameters, block by block
    for block=1:model.n_blocks
        model.block_funcs[block](model)
    end

    # extract new parameter values from model
    stash.state = deepcopy(model.state)
    stash.count += 1  # increment counter

    # Produce the next sample
    send(stash.state)
end
