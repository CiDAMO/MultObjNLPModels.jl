export RegressionModel, LogisticRegressionModel

using ForwardDiff, LinearAlgebra

abstract type AbstractRegressionModel <: AbstractMultObjModel end

include("linear-regression-model.jl")
include("logistic-regression-model.jl")

mutable struct RegressionModel <: AbstractRegressionModel
  meta :: MultObjNLPMeta
  counters :: MultObjCounters

  X :: Matrix
  y :: Vector
  h    # h(x, w)
  ℓ    # ℓ(y, ŷ)
end

function RegressionModel(X, y, h, ℓ)
  n, p = size(X)

  meta = MultObjNLPMeta(p, n)
  counters = MultObjCounters(n)
  return RegressionModel(meta, counters, X, y, h, ℓ)
end

function NLPModels.obj(nlp :: RegressionModel, i :: Integer, β :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_obji, i)
  return nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β))
end

function NLPModels.grad!(nlp :: RegressionModel, i :: Integer, β :: AbstractVector , g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  increment!(nlp, :neval_gradi, i)
  ForwardDiff.gradient!(g, β->nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β)), β)
  return g
end

function NLPModels.grad!(nlp :: RegressionModel, J :: AbstractVector{<: Integer}, β :: AbstractVector , g :: AbstractVector)
  NLPModels.@lencheck nlp.meta.nvar β g
  for i in J
    increment!(nlp, :neval_gradi, i)
  end
  ForwardDiff.gradient!(g,
    β -> sum(nlp.ℓ(nlp.y[i], nlp.h(nlp.X[i,:], β)) for i in J),
  β)
  return g
end