export MultObjCounters

mutable struct MultObjCounters
  counters :: Counters
  neval_obji    :: Vector{Int}  # Number of individual objective evaluations
  neval_gradi   :: Vector{Int}  # Number of individual objective gradient evaluations.
  neval_hessi   :: Vector{Int}  # Number of individual objective Hessian evaluations.
  neval_hiprod  :: Vector{Int}  # Number of individual objective Hessian-vector products.

  function MultObjCounters(nobj)
    return new(Counters(), zeros(Int, nobj), zeros(Int, nobj), zeros(Int, nobj), zeros(Int, nobj))
  end
end

import Base.getproperty, Base.setproperty!

function sum_counters(c :: MultObjCounters)
  s = 0
  for field in fieldnames(Counters)
    s += getfield(c.counters, field)
  end
  for field in fieldnames(MultObjCounters)
    field == :counters && continue
    s += sum(getfield(c, field))
  end
  return s
end

function NLPModels.reset!(c :: MultObjCounters)
  for f in fieldnames(MultObjCounters)
    f == :counters && continue
    fill!(getfield(c, f), 0)
  end
  reset!(c.counters)
  return c
end
