using SimpleChains, Images, VideoIO, WAV

# load the image
# fn = _data("10057.png")
# bn = "81.jpeg"
bn = "010323.wav"
_bn, ext = splitext(bn)
fn = _data(bn)

# fn = _data("fluid.mp4")
# fn = _data("green.mp4")
# img = Gray.(load(fn))
skip = 16
lrs = [1e-4, 1e-3]
N = 4000
Ls = [100, 100, 100]
# vid = load(fn)
# img = vid[1]
if ext == ".wav"
    s, sr = load(fn)
    s = s[1:(44100*3)]
else
    img = load(fn)
end
# img = load(fn)
# img = Gray.(load(fn))#[1:skip:end, 1:skip:end]
img = img[1:skip:end, 1:skip:end]#[:, 1:end-1]
chimg = channelview(img)
imgsize = size(img)
chimgsize = size(chimg)

ci = CartesianIndices(chimg)
x = stack(reshape(map(x -> collect(x.I), ci), :))
y = Float64.(reshape(vcat([chimg[x] for x in ci]), 1, :))

# act = SimpleChains.σ
act = tanh
for lr in lrs
    @show lr
    in_size = ndims(chimg)
    model = SimpleChain(
        static(in_size),
        TurboDense{true}(act, Ls[1]),
        TurboDense{true}(act, Ls[2]),
        TurboDense{true}(act, Ls[3]),
        TurboDense{false}(identity, 1),
        SquaredLoss(y)
    )

    p = SimpleChains.init_params(model; rng=SimpleChains.local_rng())
    g = SimpleChains.alloc_threaded_grad(model)

    fwd = SimpleChains.remove_loss(model)
    opt = SimpleChains.ADAM(lr)
    # imgs = Array{Gray{N0f8}}(undef, N, chimgsize...);
    imgs = Vector(undef, N,)
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

        o = map(only, reshape(fwd.(eachcol(x), (p,)), chimgsize...))
        clamp01!(o)

        # imgi = Gray{N0f8}.(o)
        imgi = RGB{N0f8}.(colorview(RGB, o))

        imgs[i] = imgi
        if i % 250 == 0
            display(imgi)
        end

    end

    # imgsb = imgs[:, :, 1:end-1]
    # imgstack = eachslice(imgsb;dims=1)
    # # mean(imgstack[end-100:end];dims=2)[1]

    # VideoIO.save("video.mp4", imgstack[1:100:end], framerate=60)
    VideoIO.save("$(splitext(bn)[1])_$(act)_$(lr)_$(skip)_$(N)_$(Ls).mp4", imgs, framerate=60)
end