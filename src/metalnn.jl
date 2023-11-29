using Pkg
metal_dir = dirname(@__DIR__)
Pkg.activate(; temp=true)
Pkg.add(["Metal", "Flux", "DataFrames", "OneHotArrays"])
Pkg.develop(path=metal_dir)

using Flux
using Metal
using CSV
using DataFrames
using OneHotArrays
using Statistics

Metal.functional()

device = Flux.get_device(; verbose=true)

device.deviceID

# model = Dense(2 => 3); # the model initially lives in CPU
# model.weight
# model = model |> device # transfer model to GPU
# model.weight


# load breast cancer dataset
csv_file_path = metal_dir * "/data/breast-cancer-wisconsin.data"
df_orig = CSV.File(csv_file_path) |> DataFrame;
column_names = [:SampleCodeNumber, :ClumpThickness, :UniformityOfCellSize,
:UniformityOfCellShape, :MarginalAdhesion, :SingleEpithelialCellSize,
:BareNuclei, :BlandChromatin, :NormalNucleoli, :Mitoses, :Class];
rename!(df_orig, column_names);

# # Remove unwanted columns
df = df_orig[:, Not(Cols(:BareNuclei, :SampleCodeNumber))]

# Define a dictionary to map values
class_mapping = Dict(2 => 0, 4 => 1)
# Use the map function to apply the mapping
df[!, :Class] = map(
    x -> class_mapping[x], 
    df[:, :Class]
);

# features
x = df[:, Not(Cols(:Class))] |> Matrix{Float32}
# labels
y = df[:, :Class]

const classes = [0, 1]
y_onehot = onehotbatch(y, [0, 1])


# store in gpu
x |> device
y |> device


# define simple neural network
model = Chain(
    Dense(8 => 2), # model with 8 features and 2 classes
    softmax
)

# model[1].weight
# model[1].bias

model |> device;

function loss(model, x, y)
    ŷ = model(x)
    Flux.logitcrossentropy(ŷ, y)
end

loss(model, x', y_onehot)

# calculate the accuracy
accuracy(x, y) = mean(onecold(model(x), classes) .== y);
accuracy(x', y)