"""
Definition of Gibbs-type model
"""

export MCMCGibbsModel

"""
## Fields

* curr_params: Current values of the model parameters
* curr_dists: Current distributions for the model parameters
* block_sym_to_num: Dictionary that maps a Symbol or vector of Symbols
    into integers. This will allow us to control the order of block
    evaluation. Allowing keys to be vectors of Symbols allows us to
    block multiple parameters together
* data: The data needed to evaluate block_funcs. Not used internally,
    but stored here to it is available to all processes with a copy of
    the model. Makes parallelization trivial
* block_funcs: A dict mapping integers specifying block numbers into
    functions that update the parameters/distributions for a particular
    block. The keys here correspond to the values of block_sym_to_num.
    Each function should update curr_params and/or curr_dists inplace.
    Return values will not be used.
* n_block: Number of blocks. Just makes life easier for the sampler to
    iterate over the blocks each scan.
"""
type MCMCGibbsModel <: MCMCModel
    # make these with symbols so we can refer to them by name in the block
    # functions
    curr_params::Dict{Symbol, VectorOrOne{Float64}}
    curr_dists::Dict{Symbol, Distribution}
    block_sym_to_num::Dict{VectorOrOne{Symbol}, Int}
    block_funcs::Dict{Int, Function}
    data::Any
    n_block::Int

    function MCMCGibbsModel(cp, cd, sym2num, funcs, data, n_block)
        if n_block != length(sym2num)
            msg = "block_sym_to_num and block_funcs must have same length"
            throw(ArgumentError(msg))
        end
        new(cp, cd, sym2num, funcs, data, n_block)
    end
end

function MCMCGibbsModel(curr_params::Dict{Symbol, VectorOrOne{Float64}},
                        curr_dists::Dict{Symbol, VectorOrOne{Distribution}},
                        block_sym_to_num::Dict{VectorOrOne{Symbol}, Int},
                        block_funcs::Dict{Int, Function},
                        data::Dict{Any, Any})
    n_block = length(block_funcs)
    return MCMCGibbsModel(curr_params, curr_dists, block_sym_to_num,
                          block_funcs, data, n_blocks)
end

# TODO: write this show method
function show(io::IO, res::MCMCLikelihoodModel)
    nothing
end

# TODO: more constructors. Hook into DSL somehow.
