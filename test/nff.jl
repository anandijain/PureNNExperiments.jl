function fourier_feature_mapping(x, B)
    return [cos.(x*B); sin.(x*B)]
end

# Set up the number of Fourier features and their frequencies
num_fourier_features = 256
frequencies = rand(Distributions.Normal(0, 1), in_size, num_fourier_features)

# Generate Fourier features for the inputs
x_fourier_features = fourier_feature_mapping(x, frequencies)
