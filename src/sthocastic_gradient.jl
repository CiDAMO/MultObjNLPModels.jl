using LinearAlgebra, Random, SolverTools

export sthocastic_gradient

function sthocastic_gradient(
    nlp::AbstractMultObjModel;
    learning_rate::Symbol = :optimal,
    γ::Float64 = 1e-2,
    α::Float64 = 1e-2, # for penalty
    ρ::Float64 = 0.85, # just for elasticnet
    penalty::Symbol = :l2,
    max_eval::Int = 0,
    max_time::Float64 = 60.0,
    max_iter::Int = 1000,
    atol::Float64 = 1e-8,
    rtol::Float64 = 1e-8,
    power_t::Float64 = 1e-2
  )

  iter = 0
  start_time = time()
  β = nlp.meta.x0
  βavg = similar(β)
  n = nlp.meta.nobj
  g = similar(β)

  f = obj(nlp, β)
  grad!(nlp, β, g)

  Δt = time() - start_time

  P = if penalty == :l2
    β -> β
  elseif penalty == :l1
    β -> sign.(β)
  elseif penalty == :elasticnet
    β -> ρ * β + (1 - ρ) * sign.(β)
  end

  status = :unknown
  tired = Δt > max_time || sum_counters(nlp) > max_eval > 0 || iter > max_iter
  solved = γ < 1e-6

  while !(solved || tired)

    if learning_rate == :optimal
      γ = 1 / (α * (1e3 + iter))
    elseif learning_rate == :invscaling
      γ = 1e-2 / (iter + 1)^power_t
    end

    βavg .= 0
    for i in shuffle(1:n)
      β -= γ * (α * P(β) + grad!(nlp, i, β, g))
      βavg += β
    end
    βavg = βavg / n

    Δt = time() - start_time
    iter += 1
    tired = Δt > max_time || sum_counters(nlp) > max_eval > 0 || iter > max_iter
    solved = γ < 1e-2
  end

  if solved
    :small_step
  elseif tired
    if Δt >: max_time
      :max_time
    elseif sum_counters(nlp) > max_eval
      :max_eval
    elseif iter > max_iter
      :max_iter
    end
  end

  return GenericExecutionStats(status, nlp;
                               solution=β,
                               solver_specific=Dict(:βavg => βavg),
                               elapsed_time=Δt,
                               iter=iter
                               )
end
