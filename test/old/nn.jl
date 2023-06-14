using Flux, ImageIO, FileIO, ImageView, ImageShow,  Colors 
fn = _data("10057.png")
img = load(fn)
fimg = Float32.(img)
fimg16 = Float16.(img)

data = []
for i in 1:28
    for j in 1:28
        push!(data, (Float32.([i / 28, j / 28]), fimg[i, j]))
    end
end
xs, ys = unzip(data)

m = model = Chain(Dense(2 => 20, Flux.sigmoid), Dense(20 => 8, Flux.sigmoid), Dense(8 => 1))
# m2 = model = f16(Chain(Dense(2 => 10, Flux.tanh), Dense(10 => 4, Flux.sigmoid), Dense(4 => 1)))

myloss(m, x, y) = (m(x)[1] - y)^2
optim = Flux.setup(Adam(), model)
# Data
all_ls = []
for epoch in 1:1000
    Flux.train!(myloss, model, data, optim)
    if epoch % 10 == 0
        ls = [myloss(m, d...) for d in data]
        tot_l = sum(ls)
        push!(all_ls, tot_l)
        @info epoch, tot_l
    else 
        # @info epoch
    end
    # display(plot(all_ls))
    display(Gray.(reshape(stack(m.(xs)), 28, 28)))
    # @info epoch
end

using BenchmarkTools
@benchmark m(xs[1]) 
@benchmark m2(xs[1]) 