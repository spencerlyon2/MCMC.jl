"""
Definition of Gibbs-type model
"""

export MCGibbsModel

# just write this out so I don't have to type it a bunch of times
typealias DictKeyVectorOrOne{K, V} Union(Dict{K, V}, Dict{Vector{K}, V})


"""
## Fields

* curr_params: Current values of the model parameters
* dists: Current distributions for the model parameters
* block_sym2num: Dictionary that maps a Symbol or vector of Symbols
    into integers. This will allow us to control the order of block
    evaluation. Allowing keys to be vectors of Symbols allows us to
    block multiple parameters together
* data: The data needed to evaluate block_funcs. Not used internally,
    but stored here to it is available to all processes with a copy of
    the model. Makes parallelization trivial
* block_funcs: A dict mapping integers specifying block numbers into
    functions that update the parameters/distributions for a particular
    block. The keys here correspond to the values of block_sym2num.
    Each function should update curr_params inplace.
    Return values will not be used.
* n_block: Number of blocks. Just makes life easier for the sampler to
    iterate over the blocks each scan.
"""
type MCGibbsModel <: MCModel
    # make these with symbols so we can refer to them by name in the block
    # functions
    init::Dict{Symbol}  # let values of init dict be anything to allow vectors
    dists::Dict{Symbol, Distribution}
    block_sym2num::DictKeyVectorOrOne{Symbol, Int}
    block_funcs::Dict{Int, Function}
    data::Dict
    n_block::Int
    size::Int

    function MCGibbsModel(params::Dict{Symbol},
                          dists::Dict{Symbol, Distribution},
                          sym2num::DictKeyVectorOrOne{Symbol, Int},
                          funcs::Dict{Int, Function},
                          data::Dict{Any, Any}=Dict{Any, Any}(),
                          n_block::Int=length(block_funcs),
                          size::Int=length(init))
        if n_block != length(sym2num)
            msg = "block_sym2num and block_funcs must have same length"
            throw(ArgumentError(msg))
        end
        new(params, dists, sym2num, funcs, data, n_block)
    end
end

function show(io::IO, m::MCGibbsModel)
    msg = "GibbsModel, with $(m.size) in $(m.n_block) blocks"
    print(io, msg)
end

# TODO: more constructors. Hook into DSL somehow.
