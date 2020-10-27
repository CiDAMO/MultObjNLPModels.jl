include("manual_rosenbr.jl")

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")
include(joinpath(nlpmodels_path, "consistency.jl"))

@testset "Consistency check" begin
  nlp_autodiff = ADNLPModel(
    x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2,
    [-1.2; 1.0]
  )
  nlp_mo = Rosenbrock()
  nlps = [nlp_mo, nlp_autodiff]

  consistent_nlps(nlps)
end