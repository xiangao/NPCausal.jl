using Documenter
using NPCausal

makedocs(
    sitename = "NPCausal.jl",
    modules = [NPCausal],
    pages = [
        "Home" => "index.md",
        "Vignettes" => [
            "Getting Started" => "vignettes/01_getting_started.md",
            "Advanced Estimators" => "vignettes/02_advanced_estimators.md",
        ],
    ],
    warnonly = true,
)
