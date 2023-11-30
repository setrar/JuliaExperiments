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

model = Chain(
    Dense(8 => 2),  # model with 8 features and 2 classes
    σ               # sigmoid activation
) |> device         # output model to GPU

classes = [0, 1]
accuracy(x, y) = mean(onecold(model(x), classes) .== y);
# accuracy(x, y_orig)

# function loss(ŷ, y)
#     Flux.logitcrossentropy(ŷ, y)
# end

function loss(ŷ, y)
    Flux.binarycrossentropy(ŷ, y)
end


X_train = Matrix{Float64}(df[:, Not(:Class)])'
y_train = vec(df[:, :Class])
train_loader = Flux.DataLoader((X_train, y_train), batchsize=1, shuffle=false)
epochs = 1
optimizer = Flux.setup(Adam(), model)

for epoch in 1:epochs
    total_accuracy = 0.
    total_loss = 0.
    for (x_cpu, y_cpu) in train_loader
        # pass the input and label to the GPU
        input = reshape(x_cpu, :) |> device
        label = y_cpu[1] |> device
        # calculate the gradient
        ∂f∂x = gradient(m -> loss(m(input), label), model)
        Flux.update!(optimizer, model, ∂f∂x[1])
        # calculate accuracy and loss
        ŷ = model(input)
        total_accuracy += accuracy(input, label)
        total_loss += loss(ŷ, label)
    end
    s = length(train_loader)
    avg_accuracy = total_accuracy / s
    avg_loss = total_loss / s
    println("Epoch $epoch: Accuracy=$avg_accuracy, Loss=$avg_loss")
end
