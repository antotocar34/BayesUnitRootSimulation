using HypothesisTests
using LinearAlgebra
using Distributions
using StanSample
using LaTeXStrings
using Plots
using StatsPlots
using StatsBase
using DataFrames


const sigma_ϵ = 1

# Simulate the model
function simulate_AR1(ϕ, T ; sigma_ϵ=1)
    y = zeros(T)
    ϵ = rand(MultivariateNormal(zeros(T), sigma_ϵ^2 * I))
    y[1] = ϵ[1]
    for t=2:T
        y[t] = ϕ*y[t-1] + ϵ[t] 
    end
    return y
end

# Bayesian Inference
standardized_model1 = "
    data {
      int<lower=0> T;
      real<lower=0> sigma;
      vector[T] y;
    }
    transformed data {
      real<lower=0> sigma_std;
      vector[T] y_std;
      y_std = (y - mean(y)) / sd(y);
      sigma_std = sigma / sd(y) ;
    }
    parameters {
      real phi;
    }
    model {
      phi ~ normal(1/2, 1) ;
      y_std[2:T] ~ normal(phi * y_std[1:(T - 1)], sigma_std);
    }
";

standardized_model2 = "
    data {
      int<lower=0> T;
      real<lower=0> sigma;
      vector[T] y;
    }
    transformed data {
      real<lower=0> sigma_std;
      vector[T] y_std;
      y_std = (y - mean(y)) / sd(y);
      sigma_std = sigma / sd(y) ;
    }
    parameters {
      real phi;
    }
    model {
      phi ~ double_exponential(1, 0.1);
      y_std[2:T] ~ normal(phi * y_std[1:(T - 1)], sigma_std);
    }
";


function null_pp(ϕ, T, sm; sigma_ϵ=1 )
    y = simulate_AR1(ϕ, T)
    data = Dict(
                "T" => T,
                "sigma" => sigma_ϵ^2,
                "y" => y
               )
    rc = stan_sample(
                     sm; 
                     data,
                     num_samples=5000
                    )
    if success(rc) == true
        results = read_samples(sm; output_format=:array)
    end
    prob_H1 = mean(map(x -> x < 1, results[:phi]))
    return prob_H1
end

## Frequentist Inference

function DF_pvalue(ϕ, T; sigma_ε=1, num=100)
    ps = Vector{Float64}(undef, num)
    for n=1:num
        y = simulate_AR1(ϕ, T)
        out = ADFTest(y, :none, 0)
        ps[n] = pvalue(out)
    end
    return mean(ps)
end

pvalue_up(p) = begin
               if p == 1
                   return 1
               end
               BFB = 1 / (-exp(1)*p*log(p))
               return BFB / (1 + BFB)
               end

const sm1 = SampleModel("AR1_1", standardized_model1)
const sm2 = SampleModel("AR1_2", standardized_model2)

function posterior_probs(T, sm1, sm2 ; num_reps=10)
    seq = 0.8:0.01:1.01
    freq_results   = Vector{Float64}(undef, length(seq))
    bayes_results1 = Vector{Float64}(undef, length(seq))
    bayes_results2 = Vector{Float64}(undef, length(seq))
    for (i,ϕ) in enumerate(seq)
        println(i / length(seq))
        bayes_results1[i] = mean([null_pp(ϕ, T, sm1) for _=1:num_reps])
        bayes_results2[i] = mean([null_pp(ϕ, T, sm2) for _=1:num_reps])
        freq_results[i] = mean([
                                pvalue_up(DF_pvalue(ϕ, T)) for _=1:num_reps
                               ])
    end
    return bayes_results1, bayes_results2, freq_results, seq
end

function make_first_plot(T)
    out = posterior_probs(T, sm1, sm2)
    seq = out[4]
    plot(seq, out[3] , label=L"$p(H_1 \;|\; y)$", legend=:bottomleft, title="T=$T", dpi=450) 
    plot!(seq, out[1], label=L"$p(H_1 \;|\; y)$, Normal Prior" ) 
    plot!(seq, out[2],label=L"$p(H_1 \;|\; y)$, Laplace Prior" ) 
    savefig("./plots/posterior_probs_$T.png")
end

make_first_plot(50)
make_first_plot(500)
make_first_plot(1000)

function test_decision_rule(ϕ, T ; num=50)
    p = 0
    pp1 = 0
    pp2 = 0

    for i=1:num
      println(i / num)
      bool = (ϕ == 1 )
      p += (DF_pvalue(ϕ, T) < 0.1) != bool
      pp1 += (null_pp(ϕ, T, sm1) < 0.5) == bool
      pp2 += (null_pp(ϕ, T, sm2) < 0.5) == bool
    end
    return p/num, pp1/num, pp2/num
end

# Get values for table
test_decision_rule(1, 2000)
test_decision_rule(0.99, 2000)
test_decision_rule(0.95, 2000)


# Make Plot for prior distribution
plot(Normal(1/2, 1), label="Normal", dpi=400)
plot!(Laplace(1, 0.1), label="Laplace")
xlims!(0, 1.5)
savefig("./plots/priors.png")
