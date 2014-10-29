"""
Definition of Gibbs-type model
"""

export MCGibbsModel

# just write this out so I don't have to type it a bunch of times
typealias DictKeyVectorOrOne{K, V} Union(Dict{K, V}, Dict{Vector{K}, V})


"""
## Fields

* init: Initial values for the model parameters
* dists: Current distributions for the model parameters
* block_sym2num: Dictionary that maps a Symbol or vector of Symbols
    into integers. This will allow us to control the order of block
    evaluation. Allowing keys to be vectors of Symbols allows us to
    block multiple parameters together
* block_funcs: A dict mapping integers specifying block numbers into
    functions that update the parameters/distributions for a particular
    block. The keys here correspond to the values of block_sym2num.
    Each function should update the state field on the model inplace.
    Return values will not be used. Each of these functions should have
    the following form::

    ```
    function block_n!(m::MCGibbsModel, s::GibbsState)

        # Do calculations. Using any information in m and s

        s.current[:blockn_param1] = updated_value
        s.current[:blockn_param2] = updated_value2
    end
    ```

    Specifically, each function should accept those two argument types
    and should update the `current` field of the `GibbsState` parameter
    inplace.
* data (optional): The data needed to evaluate block_funcs. Not used
    internally, but stored here to it is available to all processes with
    a copy of the model. Makes parallelization trivial
"""
type MCGibbsModel <: MCModel
    # make these with symbols so we can refer to them by name in the block
    # functions
    init::Dict{Symbol}  # let values of init dict be anything to allow vectors
    dists::Dict{Symbol, Distribution}
    block_sym2num::DictKeyVectorOrOne{Symbol, Int}
    block_funcs::Dict{Int, Function}
    data::Dict
    n_blocks::Int
    size::Int

    function MCGibbsModel(init::Dict{Symbol},
                          dists::Dict{Symbol, Distribution},
                          sym2num::DictKeyVectorOrOne{Symbol, Int},
                          funcs::Dict{Int, Function},
                          data::Dict{Any, Any}=Dict{Any, Any}(),
                          n_blocks::Int=length(block_funcs),
                          size::Int=length(init))
        if n_blocks != length(sym2num)
            msg = "block_sym2num and block_funcs must have same length"
            throw(ArgumentError(msg))
        end
        new(init, dists, sym2num, funcs, data, n_blocks, size)
    end
end

function show(io::IO, m::MCGibbsModel)
    msg = "GibbsModel, with $(m.size) in $(m.n_blocks) blocks"
    print(io, msg)
end

# TODO: more constructors. Hook into DSL somehow.
