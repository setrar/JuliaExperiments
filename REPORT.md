# REPORT

The goal of the fork is to try METAL experiment through Jupyter Notebooks

The 3 source files have been mimicked in a Notebook file:

- [ ] [fluxnn.ipynb](fluxnn.ipynb)
- [ ] [kernetfunction.ipynb](kernetfunction.ipynb)
- [ ] [profilekf.ipynb](profilekf.ipynb)

&#x1F516; Note: Packages have to be installed first before running `Jupyter lab`

```julia
julia> using Metal

julia> Metal.versioninfo()
macOS 14.2.1, Darwin 23.2.0

Toolchain:
- Julia: 1.10.1
- LLVM: 15.0.7

Julia packages: 
- Metal.jl: 0.5.1
- Metal_LLVM_Tools_jll: 0.5.1+0

1 device:
- Apple M2 (64.000 KiB allocated)

julia> dev = MTLDevice(1)
<AGXG14GDevice: 0x12e274600>
    name = Apple M2

julia> dev.name
NSString("Apple M2")
```

&#x1F516; Note: Observe the `NSString` type which comes from `ObjectiveC.jl` package also used by `Metal.jl`
