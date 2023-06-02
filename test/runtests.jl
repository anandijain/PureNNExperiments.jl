using CSV, DataFrames, Flux
using StatsBase, DataInterpolations, Plots

df = CSV.read("/Users/anand/.julia/dev/Fin/jxo1d.csv", DataFrame)
t = Dates.value.(df.t)
dt = diff(t)

ns = names(df)[2:end]
# plot(t, df.JPYUSD)
function lirp(x, t)
    eqt = t[1]:minimum(diff(t)):t[end]
    itp = LinearInterpolation(x, t)
    eqt, itp(eqt)
end

plot(eqt, xs)

tups = [lirp(x, t) for x in eachcol(df)[2:end]]
cols = last.(tups)
pushfirst!(cols, collect(eqt))
df = DataFrame(cols, names(df))


N_X = 100
PAST = 1
FUTURE = 1
data = [(Float32.(collect(df[i-1, 2:end])), Float32.(collect(df[i, 2:end]))) for i in 2:nrow(df)]

# arch = [N_X, 50, 10, 10, N_X]

m = model = Chain(Dense(N_X => 5, tanh), Dense(5 => 5, tanh), Dense(5 => N_X))
optim = Flux.setup(Adam(), model)

myloss(m, x, y) = sum((m(x) .- y) .^ 2)

all_ls = []
for epoch in 1:50
    Flux.train!(myloss, model, data, optim)
    ls = [myloss(m, d...) for d in data]
    tot_l = sum(ls)
    push!(all_ls, tot_l)
    display(plot(all_ls))
    @info epoch, tot_l
    @info epoch
end
