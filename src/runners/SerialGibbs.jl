###########################################################################
#
#  SerialGibbs runner: consumes repeatedly a gibbs sampler and returns a
#                      GibbsChain
#
#
###########################################################################

immutable SerialGibbs <: MCMCRunner
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
        @assert len > burnin "Total MCMC length ($len) should be > to" *
                              " burnin ($burnin)"
        @assert thinning >= 1 "Thinning ($thinning) should be >= 1"

        new(burnin, thinning, len, r)
    end
end

SerialGibbs(steps::Range1{Int}) = SerialGibbs(first(steps):1:last(steps))

function SerialGibbs(;steps::Int=100, burnin::Int=0, thinning::Int=1)
    SerialGibbs((burnin+1):thinning:steps)
end

function run_serialgibbs(t::MCMCTask)
    tic() # start timer

    param_names = keys(t.model.curr_params)

    samples = Dict{Symbol, Array}()
    for nm in param_names
        samples[nm] = fill(NaN, length(t.model.curr_params[nm]),
                           length(t.runner.r))
    end

    diags = Dict()
    diags["step"] = collect(t.runner.r)

    j = 1
    for i in 1:t.runner.len
        newprop = consume(t.task)

        if in(i, t.runner.r)
            for nm in param_names
                samples[nm][j, :] = t.model.curr_params[nm]
            end

            # save diagnostics
            for (k,v) in newprop.diagnostics
                # if diag name not seen before, create column
                if !haskey(diags, k)
                    diags[k] = Array(typeof(v), length(diags["step"]))
                end

                diags[k][j] = v
            end

            j += 1
        end
    end

    GibbsChain(t.runner.r, samples, diags, t, toq())
end


function run_serialgibbs_exit(t::MCMCTask)
  chain = run_serialgibbs(t)
  stop!(chain)
  return chain
end



