"""
  Generic Gibbs sampler

  Parameters:
"""

export Gibbs

immutable Gibbs <: MCMCSampler
end


function SamplerTask(model::MCMCGibbsModel, sampler::Gibbs, runner::MCMCRunner)
    # Localize some parameters
    local next_pars, num_blocks, block

    # initialization
    num_blocks = model.n_block

    while true

        # run all blocks
        for block=1:num_blocks
            # updates param value and distribution for each param in  block
            model.block_funcs[block](model)
        end

        # Construct a new GibbsSample
        next_pars = copy(model.curr_params)
        gs = GibbsSample(next_pars, {"accept" => true})

        # Produce the next GibbsSample
        produce(gs)
    end
end
