"""
Generic Gibbs sampler

    Parameters:
"""
immutable Gibbs <: MCSampler
end

type GibbsStash <: MCStash{GibbsSample}
    instate::GibbsState{GibbsState}   # current param values
    outstate::GibbsState{GibbsState}  # current param values
    count::Int                        # Current number of iterations
end


function inititialize(m::MCGibbsModel, s::Gibbs, r::MCRunner, t::MCTuner)
    GibbsStash(GibbsState(deepcopy(m.init), Dict()),  # instate
               GibbsState(deepcopy(m.init), Dict()),  # outstate
               0)                                     # count
end


function reset!(stash::GibbsStash, x::Dict{Symbol})
    stash.instate.current = deepcopy(x)
end


function initialize_task(stash::GibbsStash, m::MCGibbsModel, s::Gibbs,
                         r::MCRunner, t::MCTuner)

    # Hook inside Task to allow remote resetting
    task_local_storage(:reset, (x::Dict{Symbol})->reset!(stash, x))

    while true
        iterate!(stash, m, s, r, t, produce)
    end
end


function iterate!(stash::GibbsStash, m::MCGibbsModel, s::Gibbs, r::MCRunner,
                  t::MCTuner, send::Function)
    # update all parameters, block by block. stash.instate should be changed
    for block=1:model.n_blocks
        model.block_funcs[block](model, stash.instate)
    end

    # extract current parameter values and diagnostics from instate
    # TODO: is there any reason not to just use instate and have runner copy?
    stash.outstate.current = deepcopy(stash.instate.current)
    stash.outstate.diagnostics = deepcopy(stash.instate.diagnostics)
    stash.count += 1  # increment counter

    # Produce the next sample
    send(stash.outstate)
end
