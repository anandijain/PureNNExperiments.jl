using SimpleChains, Images, VideoIO, WAV, Distributions
using ProgressMeter, BenchmarkTools

# bn = "short_downscaled_fluid.mp4"
# fn = _data("10057.png")
bn, skip = "10057.png", false;
bn, skip = "81.jpeg", true;
# bn = "010323.wav"
_bn, ext = splitext(bn)
fn = _data(bn)
orig_vid = vid = load(fn)

if ext == "mp4"
    orig_elt = eltype(vid[1]) # only if its a video 
    vid2 = channelview.(vid)
    v = stack(vid2)
    v[:, :, :, 1] |> colorview(orig_elt)
else
    orig_elt = eltype(vid)
    new_elt = RGB{Float32}
    if skip
        vid = vid[1:16:end, 1:8:end]
        @info size(vid)
        display(vid)
        # vid = orig_vid[2000:4:3500, 1000:8:2000]
    end
    v = channelview(vid)

end

img_dims = size(vid)
dims = size(v)
ci = CartesianIndices(v)
ci = CartesianIndices(vid)
x = stack(reshape(map(x -> collect(x.I), ci), :))
# x = hcat(eachrow(x) ./ x[:, end]...)'
# y = Float64.(reshape(vcat([v[x] for x in ci]), 1, :))
y = reduce(hcat, reshape(vcat([Float32.(v[:, x]) for x in ci]), 1, :)) # testing a network that outputs all 3 colors instead of having channel be a dimension


in_size = size(x, 1)
out_size = size(y, 1)
act = SimpleChains.σ
act = tanh
act = relu
# N = 4000
# Ls = fill(20, 2)
# Ls = fill(20, 3)
Ls = [100, 25, 25, 25]
layers = [TurboDense{true}(act, l) for l in Ls]
model = SimpleChain(
    static(in_size),
    layers...,
    TurboDense{true}(SimpleChains.σ, out_size),
    SquaredLoss(y)
)

p = SimpleChains.init_params(model; rng=SimpleChains.local_rng())
g = SimpleChains.alloc_threaded_grad(model)

fwd = SimpleChains.remove_loss(model)
opt = SimpleChains.ADAM(0.0001)
# opt = SimpleChains.ADAM()

# @benchmark valgrad!($g, $model, $x, $p)

N = 1000

@showprogress for i in 1:N

    SimpleChains.train_batched!(
        g,
        p,
        model,
        x,
        opt,
        10;
    )
    _fwd = collect.(fwd.(eachcol(x), (p,)))
    clamp01!.(_fwd)

    _fs = reshape(map(x -> RGB(x...), _fwd), img_dims)
    # @info "i: $i, loss: $l"
    # o = only.()
    # clamp01!(o)
    # fs = _fs |> colorview(new_elt)

    if (i % 1 == 0)
        display(_fs)
    end
end

# fs = new_elt.(reshape(o, dims))

# fv = eachslice(fs; dims=3)
# VideoIO.save("vidtest2.mp4", fv, framerate=60)

# VideoIO.save("$(_bn)_$(act)_$(lr)_$(skip)_$(N)_$(Ls).mp4", imgs, framerate=60)
