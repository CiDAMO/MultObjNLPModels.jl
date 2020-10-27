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
function getproperty(c :: MultObjCounters, f :: Symbol)
  if f in fieldnames(Counters)
    getfield(c.counters, f)
  else
    getfield(c, f)
  end
end

function setproperty!(c :: MultObjCounters, f :: Symbol, x)
  if f in fieldnames(Counters)
    setfield!(c.counters, f, x)
  else
    setfield!(c, f, x)
  end
end

function NLPModels.reset!(c :: MultObjCounters)
  for f in fieldnames(MultObjCounters)
    f == :counters && continue
    fill!(getfield(c, f), 0)
  end
  reset!(c.counters)
  return c
end
