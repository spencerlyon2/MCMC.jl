"""
SerialGibbs runner: consumes repeatedly a gibbs sampler and returns a
                    GibbsChain
"""

immutable SerialGibbs <: SerialMCRunner
    burnin::Int
    thinning::Int
    len::Int
    r::Range

    function SerialGibbs(steps::Range{Int})
        r = steps

        burnin = first(r)-1
        thinning = r.step
        len = last(r)

        @assert burnin >= 0 "Burnin rounds ($burnin) should be >= 0"
        @assert len > burnin "Total MC length ($len) should be > to" *
                              " burnin ($burnin)"
        @assert thinning >= 1 "Thinning ($thinning) should be >= 1"

        new(burnin, thinning, len, r)
    end
end

SerialGibbs(steps::Range1{Int}) = SerialGibbs(first(steps):1:last(steps))

function SerialGibbs(;steps::Int=100, burnin::Int=0, thinning::Int=1)
    SerialGibbs((burnin+1):thinning:steps)
end

function run(m::MCGibbsModel, s::Gibbs, r::SerialGibbs, t::MCTuner, job::MCJob)
    tic() # start timer

    param_names = keys(t.model.curr_params)

    # pre-allocate space for storing results
    samples = Dict{Symbol, Array}()
    for nm in param_names
        # This might be a 1 x N Matrix now, but we will resize at the end
        samples[nm] = fill(NaN, length(t.model.curr_params[nm]),
                           length(t.runner.r))
    end

    diags = Dict()
    diags["step"] = collect(t.runner.r)

    # sampling loop
    i::Int = 1

    for j in 1:r.nsteps
        new_state = job.receive()  # this is a Dict{Symbol, F64orVectorF64}

        if j in r.r
            for nm in param_names
                samples[nm][:, j] = new_state[nm]
            end

            # save diagnostics
            for (k,v) in new_state.diagnostics
                # if diag name not seen before, create column
                if !haskey(diags, k)
                    diags[k] = Array(typeof(v), length(diags["step"]))
                end

                diags[k][j] = v
            end

            i += 1
        end
    end

    # make sure each column is a variable. Specifically, if a parameter
    # is draws over time, we want x[i, j] to be x_j on scan i. Also
    # squeeze out singleton dimensions
    for nm in param_names
        s = samples[nm]'
        samples[nm] = squeeze(s, [1:ndims(s)][collect(size(s)) .== 1])
    end

    GibbsChain(samples, diags, toq())
end
