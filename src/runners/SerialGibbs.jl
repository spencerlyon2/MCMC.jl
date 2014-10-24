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

    samples = Dict{Symbol, Array}()
