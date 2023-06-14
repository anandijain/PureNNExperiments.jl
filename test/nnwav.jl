using SimpleChains, Images, VideoIO, WAV, Distributions

bn = "010323.wav"
_bn, ext = splitext(bn)
fn = _data(bn)

lrs = [1e-4, 1e-3]
N = 10000
Ls = [100, 100, 100]
s, sr = load(fn)
sri = Int(sr)
s = s[(sri*3):round(Int, (sri*(4))), :]
heatmap()
img = reshape(s[1:44100, 1] ,210,210)
sr = 44100
freq = 261.63
duration = 1  # duration of the sound in seconds

t = 0:1/sr:duration
s = sin.(2pi*freq*t)
wavplay(s, sr)
s_size = size(s)

ci = CartesianIndices(s)
x = stack(reshape(map(x -> collect(x.I), ci), :))
y = Float64.(reshape(vcat([s[x] for x in ci]), 1, :))

# act = SimpleChains.σ
act = tanh

lr = first(lrs)
lr = 0.01
# for lr in lrs
# @show lr
in_size = ndims(s)
model = SimpleChain(
    static(in_size),
    TurboDense{true}(act, Ls[1]),
    TurboDense{true}(act, Ls[2]),
    TurboDense{true}(sin, Ls[3]),
    TurboDense{false}(identity, 1),
    SquaredLoss(y)
)

p = SimpleChains.init_params(model; rng=SimpleChains.local_rng())
g = SimpleChains.alloc_threaded_grad(model)

fwd = SimpleChains.remove_loss(model)
opt = SimpleChains.ADAM(lr)
# imgs = Array{Gray{N0f8}}(undef, N, chimgsize...);
imgs = Vector(undef, N,)
os = []
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

    # o = map(only, reshape(fwd.(eachcol(x), (p,)), s_size...))
    # clamp!(o, -1, 1)
    # imgs[i] = o

    if i % 250 == 0
        o = map(only, reshape(fwd.(eachcol(x), (p,)), s_size...))
        clamp!(o, -1, 1)
        push!(os, o)
        wavplay(o, sr)
    end

end

o = map(only, reshape(fwd.(eachcol(x), (p,)), s_size...))
clamp!(o, -1, 1)
wavplay(o, sr) # chekc

wavwrite(o, "wav_wav.wav"; Fs=sr)
# VideoIO.save("$(splitext(bn)[1])_$(act)_$(lr)_$(skip)_$(N)_$(Ls).mp4", imgs, framerate=60)
# end