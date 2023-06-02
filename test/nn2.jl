using SimpleChains, Images, VideoIO

# load the image
# fn = _data("10057.png")
# fn = _data("81.jpeg")

fn = _data("fluid.mp4")
# img = Gray.(load(fn))
# skip = 16
vid = load(fn)
img = Gray.(load(fn))#[1:skip:end, 1:skip:end]
# img = load(fn)[1:skip:end, 1:skip:end]#[:, 1:end-1]
chimg = channelview(img)
imgsize = size(img)
chimgsize = size(chimg)

x = stack(map(x -> Float64.(collect(x.I)), reshape(collect(CartesianIndices(chimg)), :)))
xi = Int.(x)
y = Float64.(reshape(vcat([chimg[x...] for x in eachcol(xi)]), 1, :))

in_size = ndims(chimg)
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
N = 100000
# imgs = Array{Gray{N0f8}}(undef, N, chimgsize...);
imgs = Vector(undef, N,);
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

    imgi = Gray{N0f8}.(o)
    # imgi = RGB{N0f8}.(colorview(RGB, o))

    imgs[i] = imgi
    if i % 250 == 0
        display(imgi)
    end

end

# imgsb = imgs[:, :, 1:end-1]
# imgstack = eachslice(imgsb;dims=1)
# # mean(imgstack[end-100:end];dims=2)[1]

# VideoIO.save("video.mp4", imgstack[1:100:end], framerate=60)
VideoIO.save("video_rgb_4.mp4", [x[:, 1:end-1] for x in imgs], framerate=60)
