export LogisticRegressionModel

mutable struct LogisticRegressionModel <: AbstractRegressionModel
  meta :: MultObjNLPMeta
  counters :: MultObjCounters

  X :: Matrix
  y :: Vector
  h    # h(x, w)
  ℓ    # ℓ(y, ŷ)
end

function LogisticRegressionModel(X, y)
  n, p = size(X)

  meta = MultObjNLPMeta(p, n)
  counters = MultObjCounters(n)

  h(x, β) = 1 / (1 + exp(-dot(x, β)))
  ℓ(y, ŷ) = -y * log(ŷ + 1e-8) - (1 - y) * log(1 - ŷ + 1e-8)
  return LogisticRegressionModel(meta, counters, X, y, h, ℓ)
end

function NLPModels.obj(nlp :: LogisticRegressionModel, i :: Integer, β :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_obji, i)
  xi = @view nlp.X[i,:]
  ŷ = 1 / (1 + exp(-dot(xi, β)))
  yi = nlp.y[i]
  return -yi * log(ŷ + 1e-8) - (1 - yi) * log(1 - ŷ + 1e-8)
end

function NLPModels.grad!(nlp :: LogisticRegressionModel, i :: Integer, β :: AbstractVector, g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_gradi, i)
  xi = @view nlp.X[i,:]
  g .= (1 / (1 + exp(-dot(xi, β))) - nlp.y[i]) * xi
  return g
end

function NLPModels.grad!(nlp :: LogisticRegressionModel, J :: AbstractVector{<:Integer}, β :: AbstractVector, g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  for i in J
    increment!(nlp, :neval_gradi, i)
  end
  XJ = @view nlp.X[J,:]
  yJ = @view nlp.y[J]
  g .= XJ' * (1 ./ (1 .+ exp.(-XJ * β)) .- yJ)
  return g
end