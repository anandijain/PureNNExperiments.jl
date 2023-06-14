using SimpleChains, Images, VideoIO, WAV, Distributions
using ProgressMeter, BenchmarkTools

# bn = "short_downscaled_fluid.mp4"
# fn = _data("10057.png")
bn = "81.jpeg"
# bn = "010323.wav"
_bn, ext = splitext(bn)
fn = _data(bn)
vid = load(fn)

if ext == "mp4"
    orig_elt = eltype(vid[1]) # only if its a video 
    vid2 = channelview.(vid)
    v = stack(vid2)
    v[:, :, :, 1] |> colorview(orig_elt)
else
    orig_elt = eltype(vid)
    new_elt = RGB{Float32}
    v = channelview(vid[1:8:end, 1:8:end])
    
end

dims = size(v)
ci = CartesianIndices(v)
x = stack(reshape(map(x -> collect(x.I), ci), :))
y = Float64.(reshape(vcat([v[x] for x in ci]), 1, :))

in_size = size(x, 1)
act = SimpleChains.σ
# N = 4000
Ls = [20, 20, 20]
layers = [TurboDense{true}(act, l) for l in Ls]
# act = tanh
model = SimpleChain(
    static(in_size),
    layers...,
    # TurboDense{true}(act, 20),
    # TurboDense{true}(act, 20),
    # TurboDense{true}(act, 20),
    TurboDense{false}(identity, 1),
    SquaredLoss(y)
)

p = SimpleChains.init_params(model; rng=SimpleChains.local_rng())
g = SimpleChains.alloc_threaded_grad(model)

fwd = SimpleChains.remove_loss(model)
opt = SimpleChains.ADAM(1e-3)

@benchmark valgrad!($g, $model, $x, $p)

N = 100

@showprogress for i in 1:N

    SimpleChains.train_batched!(
        g,
        p,
        model,
        x,
        opt,
        1;
    )
    
    o = only.(fwd.(eachcol(x), (p,)))
    clamp01!(o)
    fs = reshape(o, dims) |> colorview(new_elt)
    display(fs)
end

fs = new_elt.(reshape(o, dims))

fv = eachslice(fs;dims=3)
VideoIO.save("vidtest2.mp4", fv, framerate=60)

# VideoIO.save("$(_bn)_$(act)_$(lr)_$(skip)_$(N)_$(Ls).mp4", imgs, framerate=60)
