using PanelAdequacy
using Printf

const FLOAT_FIELDS = (:rho, :Vn, :lambda_n, :n_eff,
                      :kappa_greedy, :kappa_designed)
const INT_FIELDS = (:n, :N, :T, :d_K, :C, :effective_C)
const ALL_FIELDS = (:n, :N, :T, :d_K, :rho, :Vn, :lambda_n, :n_eff,
                    :kappa_greedy, :kappa_designed, :C, :effective_C,
                    :verdict)

json_string(s) = "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"") * "\""

function read_design(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    header in (["unit", "time", "x", "y"],
               ["unit", "time", "x", "y", "control"]) ||
        error("unexpected CSV schema in $path")
    controlled = length(header) == 5
    unit = String[]; time = String[]; x = Float64[]; y = Float64[]
    control = Float64[]
    for line in lines[2:end]
        isempty(line) && continue
        values = split(line, ',')
        length(values) == length(header) || error("malformed row in $path")
        push!(unit, values[1]); push!(time, values[2])
        push!(x, parse(Float64, values[3])); push!(y, parse(Float64, values[4]))
        controlled && push!(control, parse(Float64, values[5]))
    end
    return (; unit, time, x, y, controls=controlled ? control : nothing)
end

function method_for(name)
    startswith(name, "sparse") ? :sparse : :structured
end

function record(data, method)
    report = cycle_report(data.y, data.x, data.unit, data.time;
                          method=method, controls=data.controls, interval=false)
    greedy = cycle_contrasts(data.x, data.unit, data.time; method=:greedy,
                             controls=data.controls)
    designed = cycle_contrasts(data.x, data.unit, data.time; method=method,
                               controls=data.controls)
    d = report.design; s = report.statistic
    return (n=d.n, N=d.N, T=d.T, d_K=d.d_K, rho=d.rho,
            Vn=designed.V_n, lambda_n=s.lambda_n, n_eff=s.n_eff,
            kappa_greedy=greedy.kappa, kappa_designed=designed.kappa,
            C=s.C, effective_C=s.effective_C, verdict=String(report.verdict)),
           designed
end

function packing_signature(cs)
    signatures = Tuple{Int,String}[]
    for c in eachindex(cs.rows)
        pairs = sort(collect(zip(cs.rows[c], cs.signs[c])); by=first)
        orientation = last(first(pairs)) < 0 ? -1.0 : 1.0
        signature = join((string(row, sign * orientation > 0 ? ":+" : ":-")
                          for (row, sign) in pairs), ",")
        push!(signatures, (first(first(pairs)), signature))
    end
    sort!(signatures; by=first)
    return last.(signatures)
end

function write_records(path, records)
    open(path, "w") do io
        print(io, "{\n")
        names = sort(collect(keys(records)))
        for (j, name) in enumerate(names)
            r = records[name]
            print(io, "  ", json_string(name), ": {")
            for (k, field) in enumerate(ALL_FIELDS)
                k > 1 && print(io, ", ")
                print(io, json_string(field), ": ")
                value = getproperty(r, field)
                if field in FLOAT_FIELDS
                    @printf(io, "%.17g", value)
                elseif field in INT_FIELDS
                    print(io, value)
                else
                    print(io, json_string(value))
                end
            end
            print(io, "}", j == length(names) ? "\n" : ",\n")
        end
        print(io, "}\n")
    end
end

function write_packing(path, signatures)
    open(path, "w") do io
        print(io, "{\n")
        names = sort(collect(keys(signatures)))
        for (j, name) in enumerate(names)
            values = join(json_string.(signatures[name]), ", ")
            print(io, "  ", json_string(name), ": [", values, "]",
                  j == length(names) ? "\n" : ",\n")
        end
        print(io, "}\n")
    end
end

root = @__DIR__
out = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(root, "julia.json")
packing_out = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(root, "julia_packing.json")
records = Dict{String,NamedTuple}()
signatures = Dict{String,Vector{String}}()
for path in sort(readdir(joinpath(root, "designs"); join=true))
    endswith(path, ".csv") || continue
    name = splitext(basename(path))[1]
    rec, cs = record(read_design(path), method_for(name))
    records[name] = rec
    signatures[name] = packing_signature(cs)
end
write_records(out, records)
write_packing(packing_out, signatures)
println("wrote ", out, " and ", packing_out)
