# Release checklist

1. Confirm CI and cross-language parity are green on `main`.
2. Confirm `Manifest.toml` is untracked and `[compat]` supports Julia 1.6, 1.10, and stable.
3. Tag and push `v0.5.1`; create a GitHub release from `CHANGELOG.md`.
4. Comment `@JuliaRegistrator register` on the release commit after the tag is public.
5. Open the resulting General-registry pull request and wait for AutoMerge.
6. Replace the README's URL install as the primary command after registration.
