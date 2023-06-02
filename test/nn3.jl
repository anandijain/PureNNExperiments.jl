using SimpleChains, Images, VideoIO
using ProgressMeter, BenchmarkTools
fn = _data("short_downscaled_fluid.mp4")
vid = load(fn)

# skip = 8
# downscaled_vid = map(x->x[1:8:end, 1:8:end], vid)
# VideoIO.save(_data("short_downscaled_fluid.mp4"), downscaled_vid)


# collect(vid)

# for later when we do rgb video
# v = stack(channelview.(vid))

# gv = 
v = stack(map(x -> Gray.(x), load(fn))) # to save memory

ci = CartesianIndices(v)
x = stack(reshape(map(x -> collect(x.I), ci), :))
y = Float64.(reshape(vcat([v[x] for x in ci]), 1, :))

in_size = size(x, 1)
act = SimpleChains.σ
# act = tanh
model = SimpleChain(
    static(in_size),
    TurboDense{true}(act, 20),
    TurboDense{true}(act, 20),
    TurboDense{true}(act, 20),
    TurboDense{false}(identity, 1),
    SquaredLoss(y)
)

p = SimpleChains.init_params(model; rng=SimpleChains.local_rng())
g = SimpleChains.alloc_threaded_grad(model)

fwd = SimpleChains.remove_loss(model)
opt = SimpleChains.ADAM(1e-3)

@benchmark valgrad!($g, $model, $x, $p)

# N = 100000
# out = Vector(undef, N,);

SimpleChains.train_batched!(
    g,
    p,
    model,
    x,
    opt,
    1000;
    # batchsize=10
)

outi = fwd.(eachcol(x), (p,))

o = map(only, reshape(, chimgsize...))

@showprogress for i in 1:N

    # clamp01!(o)

    # imgi = Gray{N0f8}.(o)
    # # imgi = RGB{N0f8}.(colorview(RGB, o))

    # out[i] = imgi
    # if i % 250 == 0
    #     display(imgi)
    # end

end
