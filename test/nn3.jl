using SimpleChains, Images, VideoIO, WAV
using ProgressMeter, BenchmarkTools

bn = "short_downscaled_fluid.mp4"
_bn, ext = splitext(bn)
fn = _data(bn)
vid = load(fn)

# for later when we do rgb video
# v = stack(channelview.(vid))
v = stack(vid) # Gray to save memory
orig_elt = eltype(v)
new_elt = Gray{N0f8}
v = new_elt.(v)

# v = v .* 3

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

outi = only.(fwd.(eachcol(x), (p,)))
fs = mapslices(x -> Gray.(x), reshape(outi, dims); dims=3)
display(fs[:, :, 1])

# o = map(only, reshape(, chimgsize...))
N = 10

@showprogress for i in 1:N

    SimpleChains.train_batched!(
        g,
        p,
        model,
        x,
        opt,
        1;
        # batchsize=10
    )
    o = only.(fwd.(eachcol(x), (p,)))
    clamp01!(o)
    fs = new_elt.(reshape(o, dims))

    # imgi = Gray{N0f8}.(o)
    # # imgi = RGB{N0f8}.(colorview(RGB, o))

    # out[i] = imgi
    # if i % 250 == 0
    #     display(imgi)
    # end

end
fv = eachslice(fs;dims=3)
VideoIO.save("vidtest.mp4", fv, framerate=60)
VideoIO.save("$(_bn)_$(act)_$(lr)_$(skip)_$(N)_$(Ls).mp4", imgs, framerate=60)
