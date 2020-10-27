using LinearAlgebra, MultObjNLPModels, Random

function regression_example()
  Random.seed!(0)
  X = randn(10, 3)
  y = X * ones(3) + randn(10) * 0.01

  nlp = LinearRegressionModel(X, y)
  output = sthocastic_gradient(nlp, γ=1e-2, learning_rate=:constant)

  # println("output = $output")
  β = output.solution
  @info("", β)
  @info("", norm(X * β - y))
end

regression_example()