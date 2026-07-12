# Bundled reference datasets (shipped with the package; no external download).
# Same panels the paper applications use and the parity harness asserts against.

const _DATADIR = normpath(joinpath(@__DIR__, "..", "data"))

"""
    datasets() -> Vector{String}

Names of the reference datasets bundled with PanelAdequacy.
"""
datasets() = sort!([replace(f, ".csv" => "")
                    for f in readdir(_DATADIR) if endswith(f, ".csv")])

"""
    datapath(name) -> String

Absolute path to a bundled reference-dataset CSV (with or without the `.csv`
suffix). Ships in the package, so it is available offline.
"""
function datapath(name::AbstractString)
    fname = endswith(name, ".csv") ? name : name * ".csv"
    path = joinpath(_DATADIR, fname)
    isfile(path) || throw(ArgumentError(
        "no bundled dataset '$name'; available: $(join(datasets(), ", "))"))
    return path
end

"""
    load_dataset(name) -> NamedTuple

Load a bundled reference dataset as a `NamedTuple` of column vectors. Columns
that parse fully as numbers become `Vector{Union{Missing,Float64}}` (blank
cells → `missing`); the rest stay `Vector{String}`. Dependency-free.

# Example
```julia
using PanelAdequacy
d = load_dataset("eiv_vdem_panel")
keep = .!ismissing.(d.ly) .& .!ismissing.(d.v2x_polyarchy) .& .!ismissing.(d.v2x_polyarchy_sd)
eiv_adequacy(collect(skipmissing(d.ly[keep])), collect(skipmissing(d.v2x_polyarchy[keep])),
             d.iso[keep], d.year[keep]; sigma_nu=collect(skipmissing(d.v2x_polyarchy_sd[keep])))
```
"""
function load_dataset(name::AbstractString)
    lines = readlines(datapath(name))
    isempty(lines) && throw(ArgumentError("empty dataset '$name'"))
    header = Symbol.(split(lines[1], ','))
    ncol = length(header)
    raw = [String[] for _ in 1:ncol]
    for ln in @view lines[2:end]
        isempty(strip(ln)) && continue
        vals = split(ln, ',', limit=ncol)
        for j in 1:ncol
            push!(raw[j], j <= length(vals) ? String(vals[j]) : "")
        end
    end
    cols = map(raw) do col
        parsed = Vector{Union{Missing,Float64}}(undef, length(col))
        ok = true
        for (i, s) in enumerate(col)
            if isempty(s)
                parsed[i] = missing
            else
                v = tryparse(Float64, s)
                v === nothing && (ok = false; break)
                parsed[i] = v
            end
        end
        ok ? parsed : col
    end
    return NamedTuple{Tuple(header)}(Tuple(cols))
end
