using NLPModels, MultObjNLPModels

"""
    f(x) = (x₁ - 1)² + 100 (x₂ - x₁²)²
    f₁(x) = (x₁ - 1)²
    f₂(x) = 100 (x₂ - x₁²)²
"""
mutable struct Rosenbrock <: AbstractMultObjModel
  meta :: MultObjNLPMeta
  counters :: MultObjCounters
end

function Rosenbrock()
  meta = MultObjNLPMeta(2, 2, x0 = [-1.2; 1.0], nnzh = 4, nnzhi = [1, 3])
  counters = MultObjCounters(2)

  return Rosenbrock(meta, counters)
end

function MultObjNLPModels.obj(nlp :: Rosenbrock, i :: Integer, x :: AbstractVector)
  NLPModels.@lencheck 2 x
  NLPModels.@rangecheck 1 2 i
  increment!(nlp, :neval_obji, i)
  if i == 1
    return (x[1] - 1)^2
  elseif i == 2
    return 100 * (x[2] - x[1]^2)^2
  end
end

function MultObjNLPModels.grad!(nlp :: Rosenbrock, i :: Integer, x :: AbstractVector, g :: AbstractVector)
  NLPModels.@lencheck 2 x g
  NLPModels.@rangecheck 1 2 i
  increment!(nlp, :neval_gradi, i)
  if i == 1
    g[1] = 2 * (x[1] - 1)
    g[2] = 0
  elseif i == 2
    g[1] = -400 * x[1] * (x[2] - x[1]^2)
    g[2] = 200 * (x[2] - x[1]^2)
  end
  return g
end

function MultObjNLPModels.hess_structure!(nlp :: Rosenbrock, i :: Integer, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  NLPModels.@rangecheck 1 2 i
  if i == 1
    NLPModels.@lencheck 1 rows cols
    rows[1] = 1
    cols[1] = 1
  elseif i == 2
    NLPModels.@lencheck 3 rows cols
    rows .= [1, 2, 2]
    cols .= [1, 1, 2]
  end
  return rows, cols
end

function MultObjNLPModels.hess_coord!(nlp :: Rosenbrock, i :: Integer, x :: AbstractVector, vals :: AbstractVector)
  NLPModels.@lencheck 2 x
  NLPModels.@rangecheck 1 2 i
  if i == 1
    NLPModels.@lencheck 1 vals
    increment!(nlp, :neval_hessi, i)
    vals[1] = 2
  elseif i == 2
    NLPModels.@lencheck 3 vals
    increment!(nlp, :neval_hessi, i)
    vals .= [-400 * (x[2] - x[1]^2) + 800 * x[1]^2, -400 * x[1], 200]
  end
  return vals
end

function MultObjNLPModels.hprod!(nlp :: Rosenbrock, i :: Integer, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector)
  NLPModels.@lencheck 2 x v Hv
  NLPModels.@rangecheck 1 2 i
  increment!(nlp, :neval_hiprod, i)
  if i == 1
    Hv[1] = 2 * v[1]
    Hv[2] = 0
  elseif i == 2
    Hv[1] = (-400 * (x[2] - x[1]^2) + 800 * x[1]^2) * v[1] - 400 * x[1] * v[2]
    Hv[2] = -400 * x[1] * v[1] + 200 * v[2]
  end
  return Hv
end