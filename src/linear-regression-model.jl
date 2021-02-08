export LinearRegressionModel

mutable struct LinearRegressionModel <: AbstractRegressionModel
  meta :: MultObjNLPMeta
  counters :: MultObjCounters

  X :: Matrix
  y :: Vector
  h    # h(x, w)
  ℓ    # ℓ(y, ŷ)
end

function LinearRegressionModel(X, y)
  n, p = size(X)

  meta = MultObjNLPMeta(p, n)
  counters = MultObjCounters(n)

  h(x, β) = dot(x, β)
  ℓ(y, ŷ) = (y - ŷ)^2 / 2
  return LinearRegressionModel(meta, counters, X, y, h, ℓ)
end

function NLPModels.obj(nlp :: LinearRegressionModel, i :: Integer, β :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_obji, i)
  return @views (nlp.y[i] - dot(nlp.X[i,:], β)) ^ 2 / 2
end

function NLPModels.grad!(nlp :: LinearRegressionModel, i :: Integer, β :: AbstractVector, g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_gradi, i)
  xi = @view nlp.X[i,:]
  g .= (dot(xi, β) - nlp.y[i]) * xi
  return g
end

function NLPModels.grad!(nlp :: LinearRegressionModel, J :: AbstractVector{<:Integer}, β :: AbstractVector, g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  for i in J
    increment!(nlp, :neval_gradi, i)
  end
  XJ = @view nlp.X[J,:]
  yJ = @view nlp.y[J]
  g .= XJ' * (XJ * β - yJ)
  return g
end