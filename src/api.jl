export AbstractMultObjModel

"""
    AbstractMultObjModel

`AbstractMultObjModel` is an abstract type for problems with multiple objective.
The default way of looking into these models is by defining the objective

```math
f(x) = ∑ᵢ σᵢ fᵢ(x)
````

An `AbstractMultObjModel` is expected to have:
- `meta :: MultObjNLPMeta`
"""
abstract type AbstractMultObjModel <: AbstractNLPModel end

# New multiple objective functions:

"""
    f = obj(nlp, i, x)

Evaluate ``fᵢ(x)``, the i-th objective function of `nlp` at `x`.
"""
function obj end

"""
    g = grad!(nlp, i, x, g)

Evaluate ``∇fᵢ(x)``, the gradient of the i-th objective function at `x` in place.
"""
function grad! end

"""
    g = grad(nlp, i, x)

Evaluate ``∇fᵢ(x)``, the gradient of the i-th objective function at `x`.
"""
function grad(nlp :: AbstractMultObjModel, i :: Integer, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  g = zeros(eltype(x), nlp.meta.nvar)
  grad!(nlp, i, x, g)
end

"""
    hess_structure!(nlp, i, rows, cols)

Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
function hess_structure! end

"""
    hess_structure!(nlp, i)

Return the structure of the Lagrangian Hessian in sparse coordinate format.
"""
function hess_structure(nlp :: AbstractMultObjModel, i :: Integer)
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  rows = zeros(Int, nlp.meta.nnzhi[i])
  cols = zeros(Int, nlp.meta.nnzhi[i])
  hess_structure!(nlp, i, rows, cols)
end

"""
    vals = hess_coord!(nlp, i, x, vals)

Evaluate the i-th objective Hessian at `x` in sparse coordinate format,
i.e., ``∇²fᵢ(x)``, rewriting `vals`.
Only the lower triangle is returned.
"""
function hess_coord! end

"""
    vals = hess_coord(nlp, i, x)

Evaluate the i-th objective Hessian at `x` in sparse coordinate format,
i.e., ``∇²fᵢ(x)``.
Only the lower triangle is returned.
"""
function hess_coord(nlp :: AbstractMultObjModel, i :: Integer, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  vals = zeros(eltype(x), nlp.meta.nnzhi[i])
  hess_coord!(nlp, i, x, vals)
end

"""
    Hv = hprod!(nlp, i, x, v, Hv)

Evaluate the product of the i-th objective Hessian at `x` with the vector `v` in
place, where the objective Hessian is ``∇²fᵢ(x)``.
"""
function hprod! end

"""
    Hv = hprod(nlp, i, x, v)

Evaluate the product of the i-th objective Hessian at `x` with the vector `v`,
where the objective Hessian is ``∇²fᵢ(x)``.
"""
function hprod(nlp :: AbstractMultObjModel, i :: Integer, x :: AbstractVector, v :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  Hv = zeros(eltype(x), nlp.meta.nvar)
  hprod!(nlp, i, x, v, Hv)
end

"""
    Hx = hess(nlp, i, x)

Evaluate the i-th objective Hessian at `x` as a sparse matrix.
Only the lower triangle is returned.
"""
function hess(nlp :: AbstractMultObjModel, i :: Integer, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  rows, cols = hess_structure(nlp, i)
  vals = hess_coord(nlp, i, x)
  sparse(rows, cols, vals, nlp.meta.nvar, nlp.meta.nvar)
end

"""
    H = hess_op(nlp, i, x)

Return the objective Hessian at `x` as a linear operator.
The resulting object may be used as if it were a
matrix, e.g., `H * v`.
"""
function hess_op(nlp :: AbstractMultObjModel, i :: Integer, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  prod = @closure v -> hprod(nlp, i, x, v)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

"""
    H = hess_op!(nlp, i, x, Hv)

Return the objective Hessian at `x` as a linear operator.
The resulting object may be used as if it were a matrix, e.g., `H * v`.
The vector `Hv` is used as preallocated storage for the operation. 
"""
function hess_op!(nlp :: AbstractMultObjModel, i :: Integer, x :: AbstractVector, Hv :: AbstractVector)
  @lencheck nlp.meta.nvar x Hv
  NLPModels.@rangecheck 1 nlp.meta.nobj i
  prod = @closure v -> hprod!(nlp, i, x, v, Hv)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

# NLPModels functions

function NLPModels.obj(nlp :: AbstractMultObjModel, x :: AbstractVector)
  @lencheck nlp.meta.nvar x
  NLPModels.increment!(nlp, :neval_obj)
  return sum(obj(nlp, i, x) * nlp.meta.weights[i] for i = 1:nlp.meta.nobj)
end

function NLPModels.grad!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, g :: AbstractVector) where T
  @lencheck nlp.meta.nvar x g
  NLPModels.increment!(nlp, :neval_grad)
  fill!(g, zero(T))
  gi = fill!(similar(g), zero(T))
  for i = 1:nlp.meta.nobj
    grad!(nlp, i, x, gi)
    g .+= nlp.meta.weights[i] * gi
  end
  return g
end

# ∇²f(x) = ∑ᵢ σᵢ ∇²fᵢ(x)
# rows = [rows₁, …, rowsₚ]
# cols = [cols₁, …, colsₚ]
function NLPModels.hess_structure!(nlp :: AbstractMultObjModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzh rows cols
  c = 0
  for i = 1:nlp.meta.nobj
    nnzi = nlp.meta.nnzhi[i]
    idx = c .+ (1:nnzi)
    @views hess_structure!(nlp, i, rows[idx], cols[idx])
    c += nnzi
  end
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, vals :: AbstractVector; obj_weight :: Real=one(T)) where T
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  NLPModels.increment!(nlp, :neval_hess)
  c = 0
  for i = 1:nlp.meta.nobj
    nnzi = nlp.meta.nnzhi[i]
    idx = c .+ (1:nnzi)
    @views hess_coord!(nlp, i, x, vals[idx])
    vals[idx] .*= obj_weight * nlp.meta.weights[i]
    c += nnzi
  end
  return vals
end

function NLPModels.hprod!(nlp :: AbstractMultObjModel, x :: AbstractVector{T}, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real=one(T)) where T
  @lencheck nlp.meta.nvar x v Hv
  NLPModels.increment!(nlp, :neval_hprod)
  fill!(Hv, zero(T))
  Hiv = fill!(similar(Hv), zero(T))
  for i = 1:nlp.meta.nobj
    hprod!(nlp, i, x, v, Hiv)
    Hv .+= obj_weight * nlp.meta.weights[i] * Hiv
  end
  return Hv
end