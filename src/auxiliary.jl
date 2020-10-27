export increment!

function sum_mo_counters(c :: MultObjCounters)
  s = 0
  for field in fieldnames(MultObjCounters)
    field == :counters && continue
    s += sum(getfield(c, field))
  end
  return s
end

NLPModels.sum_counters(nlp :: AbstractMultObjModel) = sum_counters(nlp.counters.counters)
sum_mo_counters(nlp :: AbstractMultObjModel) = sum_mo_counters(nlp.counters)

import Base.show
show_header(io :: IO, nlp :: AbstractMultObjModel) = println(io, typeof(nlp))

function show(io :: IO, nlp :: AbstractMultObjModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.counters)
end

for counter in fieldnames(MultObjCounters)
  counter == :counters && continue
  @eval begin
    """
    $($counter)(nlp)

    Get the number of `$(split("$($counter)", "_")[2])` evaluations.
    """
    $counter(nlp :: AbstractMultObjModel) = nlp.counters.$counter
    export $counter
  end
end

for counter in fieldnames(Counters)
  @eval begin
    NLPModels.$counter(nlp :: AbstractMultObjModel) = nlp.counters.counters.$counter
    export $counter
  end
end

import NLPModels.increment!
"""
    increment!(nlp, s, i)

Increment counter `s[i]` of problem `nlp`.
"""
function increment!(nlp :: AbstractMultObjModel, s :: Symbol, i :: Integer)
  getproperty(nlp.counters, s)[i] += 1
end

function NLPModels.reset!(nlp :: AbstractMultObjModel)
  reset!(nlp.counters)
  return nlp
end

# TODO: Make these functions more general inside NLPModels.jl:

# NLPModels.has_bounds(meta::MultObjNLPMeta) = length(meta.ifree) < meta.nvar
# NLPModels.bound_constrained(meta::MultObjNLPMeta) = meta.ncon == 0 && has_bounds(meta)
# NLPModels.unconstrained(meta::MultObjNLPMeta) = meta.ncon == 0 && !has_bounds(meta)
# NLPModels.linearly_constrained(meta::MultObjNLPMeta) = meta.nlin == meta.ncon > 0
# NLPModels.equality_constrained(meta::MultObjNLPMeta) = length(meta.jfix) == meta.ncon > 0
# NLPModels.inequality_constrained(meta::MultObjNLPMeta) = meta.ncon > 0 && length(meta.jfix) == 0