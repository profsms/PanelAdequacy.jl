# Stepped-wedge parity emitter (Julia). Evaluates every case of sw_reference.csv
# and writes case,quantity,value. Compare with sw_emit.R / the Python reference via
# sw_compare.py (tolerance 1e-10).
#   julia --project=. parity/sw_emit.jl test/fixtures/sw_reference.csv parity/sw_julia.csv
using PanelAdequacy

function main(ref, out)
    lines = readlines(ref)
    hdr = split(lines[1], ',')
    col(r, name) = r[findfirst(==(name), hdr)]
    num(s) = isempty(s) ? nothing : parse(Float64, s)
    open(out, "w") do io
        println(io, "case,quantity,value")
        cache = Dict{String,Any}()
        for ln in lines[2:end]
            r = split(ln, ','; limit=length(hdr))
            kind = col(r, "kind"); q = col(r, "quantity"); cid = col(r, "case")
            kind in ("balanced", "integer", "nested", "haines") || continue
            if q == "phi_prime"
                v = sw_phi_nested(num(col(r, "icc")), num(col(r, "cac")), num(col(r, "m")))
                println(io, "$cid,$q,$(repr(v))"); continue
            end
            kind == "haines" && continue   # cell sizes come from the bundled data; covered by the test suite
            d = parse.(Float64, split(col(r, "design"), ';'))
            est = Symbol(col(r, "estimator"))
            kw = Dict{Symbol,Any}(:estimator => est)
            if est === :GLS
                num(col(r, "phi")) !== nothing && (kw[:phi] = num(col(r, "phi")))
                num(col(r, "icc")) !== nothing && (kw[:icc] = num(col(r, "icc")); kw[:cac] = num(col(r, "cac")); kw[:m] = num(col(r, "m")))
            end
            W, ec = get!(cache, cid) do
                (sw_exposure_weights(d; kw...), sw_eti_cost(d; kw...))
            end
            v = q == "Gamma" ? sw_gamma(W) : q == "kappa" ? sw_radius(W) :
                q == "it_ratio" ? ec.it_ratio : q == "eti_cost" ? ec.eti_cost :
                q == "var_it" ? ec.var_it : q == "var_eti" ? ec.var_eti : W[parse(Int, q[3:end])]
            println(io, "$cid,$q,$(repr(v))")
        end
    end
end

main(ARGS[1], ARGS[2])
